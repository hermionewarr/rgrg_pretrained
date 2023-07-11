from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

import sys
sys.path.append("/home/hermione/Documents/VLP/TUM/rgrg_pretrained/")
from src.binary_classifier.binary_classifier_region_abnormal import BinaryClassifierRegionAbnormal
from src.binary_classifier.binary_classifier_region_selection import BinaryClassifierRegionSelection
from src.object_detector.object_detector import ObjectDetector
from src.language_model.language_model import LanguageModel
from src.full_model.run_configurations import NUM_BEAMS

class ReportGenerationModel(nn.Module):
    """
    Full model consisting of:
        - object detector encoder
        - binary classifier for selecting regions for sentence genneration
        - binary classifier for detecting if a region is abnormal or normal (to encode this information in the region feature vectors)
        - language model decoder
    """

    def __init__(self, pretrain_without_lm_model=False):
        super().__init__()
        self.pretrain_without_lm_model = pretrain_without_lm_model

        self.object_detector = ObjectDetector()
        # Load the best object detector from the 1st training stage here when starting the 2nd training stage
        # path_to_best_object_detector_weights = "/u/home/tanida/runs/object_detector/run_10/weights/val_loss_13.482_epoch_6.pth"
        # self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        self.binary_classifier_region_selection = BinaryClassifierRegionSelection()
        self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal()

        self.language_model = LanguageModel()

    def forward(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        #image_targets: List[Dict],  # contains a dict for every image with keys "boxes" and "labels"
        input_ids: torch.LongTensor,  # shape [(batch_size * 29) x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
        attention_mask: torch.FloatTensor,  # shape [(batch_size * 29) x seq_len]
        #region_has_sentence: torch.BoolTensor,  # shape [batch_size x 29], ground truth boolean mask that indicates if a region has a sentence or not
        #region_is_abnormal: torch.BoolTensor,  # shape [batch_size x 29], ground truth boolean mask that indicates if a region has is abnormal or not
        return_loss: bool = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Forward method is used for training and evaluation of model.
        Generate method is used for inference.
        """
        features = self.object_detector(images)
        del images

        language_model_loss = self.language_model(
            input_ids,
            attention_mask,
            features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache,
        )

        del input_ids
        del attention_mask
        del features
        return language_model_loss
               

    def get_valid_decoder_input_for_training(
        self,
        class_detected,  # shape [batch_size x 29]
        region_has_sentence,  # shape [batch_size x 29]
        input_ids,  # shape [(batch_size * 29) x seq_len]
        attention_mask,  # shape [(batch_size * 29) x seq_len]
        region_features,  # shape [batch_size x 29 x 1024]
    ):
        """
        We want to train the decoder only on region features (and corresponding input_ids/attention_mask) whose corresponding sentences are non-empty and
        that were detected by the object detector.
        """
        # valid is of shape [batch_size x 29]
        valid = torch.logical_and(class_detected, region_has_sentence)

        # reshape to [(batch_size * 29)], such that we can apply the mask to input_ids and attention_mask
        valid_reshaped = valid.reshape(-1)

        valid_input_ids = input_ids[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_attention_mask = attention_mask[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_region_features = region_features[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x 1024]

        return valid_input_ids, valid_attention_mask, valid_region_features

    def get_valid_decoder_input_for_evaluation(
        self,
        selected_regions,  # shape [batch_size x 29]
        input_ids,  # shape [(batch_size * 29) x seq_len]
        attention_mask  # shape [(batch_size * 29) x seq_len]
    ):
        """
        For evaluation, we want to evaluate the decoder on the top_region_features selected by the classifier to get a sentence generated.
        We also have to get the corresponding input_ids and attention_mask accordingly.
        """
        # reshape to [(batch_size * 29)]
        selected_regions = selected_regions.reshape(-1)

        valid_input_ids = input_ids[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]
        valid_attention_mask = attention_mask[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]

        return valid_input_ids, valid_attention_mask

    @torch.no_grad()
    def generate(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
    ):
        """
        In inference mode, we usually input 1 image (with 29 regions) at a time.

        The object detector first finds the region features for all 29 regions.

        The binary classifier takes the region_features of shape [batch_size=1, 29, 1024] and returns:
            - selected_region_features: shape [num_regions_selected_in_batch, 1024],
            all region_features which were selected by the classifier to get a sentence generated (and which were also detected by the object detector)

            - selected_regions: shape [batch_size x 29], boolean matrix that indicates which regions were selected to get a sentences generated
            (these regions must also have been detected by the object detector).
            This is needed in case we want to find the corresponding reference sentences to compute scores for metrics such as BertScore or BLEU.

        The decoder then takes the selected_region_features and generates output ids for the batch.
        These output ids can then be decoded by the tokenizer to get the generated sentences.

        We also return selected_regions, such that we can map each generated sentence to a selected region.
        We also return detections, such that we can map each generated sentence to a bounding box.
        We also return class_detected to know which regions were not detected by the object detector (can be plotted).
        """
        # top_region_features of shape [batch_size x 29 x 1024]
        features = self.object_detector(images)

        del images

        # selected_region_features is of shape [num_regions_selected_in_batch x 1024]
        # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
        # (it has exactly num_regions_selected_in_batch True values)
        """ selected_regions, selected_region_features = self.binary_classifier_region_selection(
            top_region_features, class_detected, return_loss=False
        )

        del top_region_features
 
        # selected_region_features can be empty if no region was both detected by the object detector and selected
        # by the binary classifier to get a sentence generated. This can happen especially early on in training
        # Since this would throw an exception in the language model, we return early
        if features.shape[0] == 0:
            return -1
"""
        # output_ids of shape (num_regions_selected_in_batch x longest_generated_sequence_length)
        output_ids = self.language_model.genlanguage_model_losserate(
            features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping,
        )

        del features

        return output_ids
