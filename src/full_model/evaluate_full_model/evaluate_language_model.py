"""
This module contains all functions used to evaluate the language model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - METEOR for:
        - all generated sentences
        - generated sentences for each region
        - generated sentences with gt = normal region (i.e. the region was considered normal by the radiologist)
        - generated sentences with gt = abnormal region (i.e. the region was considered abnormal by the radiologist)

    - BLEU 1-4, METEOR, ROUGE-L, CIDEr-D for all generated reports
    - Clinical efficacy metrics for all generated reports:
        - micro-averaged over 5 observations
        - exampled-based averaged over all 14 observations
        - computed for each observation individually

It also calls subfunctions which:
    - save NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences as a txt file
    (for manual verification what the model generates)
    - save NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated reports as a txt file
    (for manual verification what the model generates)
    - save NUM_IMAGES_TO_PLOT (see run_configurations.py) images to tensorboard where gt and predicted bboxes for every region are depicted,
    as well as the generated sentences (if they exist) and reference sentences for every region
"""
from collections import defaultdict
import csv
import io
import os
import re
import tempfile

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.cider.cider import Cider
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
)
from src.path_datasets_and_weights import path_chexbert_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_NLG_scores(nlg_metrics: list[str], gen_sents_or_reports: list[str], ref_sents_or_reports: list[str]) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        """
        The compute_score methods of the scorer objects require the input not to be list[str],
        but of the form:
        generated_reports =
        {
            "image_id_0" = ["1st generated report"],
            "image_id_1" = ["2nd generated report"],
            ...
        }

        Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
        following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
        see lines 132 and 133
        """
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-L
        - Cider-D

    Returns a dict that maps from the metrics specified to the corresponding scores.
    """
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores


def compute_clinical_efficacy_scores(language_model_scores: dict, gen_reports: list[str], ref_reports: list[str]):
    """
    This function computes:
        - micro average CE scores over all 14 conditions
        - micro average CE scores over 5 conditions ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")
        -> this is done following Miura (https://arxiv.org/pdf/2010.10042.pdf)
        - (micro) average CE scores of each condition
        - example-based CE scores over all 14 conditions
        -> this is done following Nicolson (https://arxiv.org/pdf/2201.09405.pdf)

    To compute these scores, we first need to get the disease labels extracted by CheXbert for both the generated and reference reports.
    This is done by the (nested) function "get_chexbert_labels_for_gen_and_ref_reports". Inside this function, there is another function
    called "label" from the module src/CheXbert/src/label.py that extracts these labels requiring 2 input arguments:
        1. chexbert (nn.Module): instantiated chexbert model
        2. csv_path (str): path to the csv file with the reports. The csv file has to have 1 column titled "Report Impression"
        under which the reports can be found

    We use a temporary directory to create the csv files for the generated and reference reports.

    The function label returns preds_gen_reports and preds_ref_reports respectively, which are List[List[int]],
    with the outer list always having len=14 (for 14 conditions, specified in CheXbert/src/constants.py),
    and the inner list has len=num_reports.

    E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
    the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

    There are 4 possible labels:
        0: blank/NaN (i.e. no prediction could be made about a condition, because it was no mentioned in a report)
        1: positive (condition was mentioned as present in a report)
        2: negative (condition was mentioned as not present in a report)
        3: uncertain (condition was mentioned as possibly present in a report)

    To compute the micro average scores (i.e. all the scores except of the example-based scores), we follow the implementation of the paper
    by Miura et. al., who considered the negative and blank/NaN to be one whole negative class, and positive and uncertain to be one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.

    To compute the example-based scores, we follow Nicolson's implementation, who considered blank/NaN, negative and uncertain to be the negative class,
    and only positive to be the positive class. Meaning labels 2 and 3 are converted to label 0.
    """

    def get_chexbert():
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model

    def get_chexbert_labels_for_gen_and_ref_reports():
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        return preds_gen_reports, preds_ref_reports

    def compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports):
        def convert_labels_like_miura(preds_reports: list[list[int]]):
            """
            See doc string of update_clinical_efficacy_scores function for more details.
            Miura (https://arxiv.org/pdf/2010.10042.pdf) considers blank/NaN (label 0) and negative (label 2) to be the negative class,
            and positive (label 1) and uncertain (label 3) to be the positive class.

            Thus we convert label 2 -> label 0 and label 3 -> label 1.
            """
            def convert_label(label: int):
                if label == 2:
                    return 0
                elif label == 3:
                    return 1
                else:
                    return label

            preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports_converted

        preds_gen_reports_converted = convert_labels_like_miura(preds_gen_reports)
        preds_ref_reports_converted = convert_labels_like_miura(preds_ref_reports)

        # for the CE scores, we follow Miura (https://arxiv.org/pdf/2010.10042.pdf) in micro averaging them over these 5 conditions:
        """five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = [] """

        # we also compute the micro average over all 14 conditions:
        total_preds_gen_reports_14_conditions = []
        total_preds_ref_reports_14_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports_converted, preds_ref_reports_converted, CONDITIONS):
            """ if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition) """

            total_preds_gen_reports_14_conditions.extend(preds_gen_reports_condition)
            total_preds_ref_reports_14_conditions.extend(preds_ref_reports_condition)

            # compute and save scores for the given condition
            precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
            acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

            language_model_scores["report"]["CE"][condition]["precision"] = precision
            language_model_scores["report"]["CE"][condition]["recall"] = recall
            language_model_scores["report"]["CE"][condition]["f1"] = f1
            language_model_scores["report"]["CE"][condition]["acc"] = acc

        # compute and save scores for all 14 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions)

        language_model_scores["report"]["CE"]["precision_micro_all"] = precision
        language_model_scores["report"]["CE"]["recall_micro_all"] = recall
        language_model_scores["report"]["CE"]["f1_micro_all"] = f1
        language_model_scores["report"]["CE"]["acc_all"] = acc

        # compute and save scores for the 5 conditions
        """ precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        language_model_scores["report"]["CE"]["precision_micro_5"] = precision
        language_model_scores["report"]["CE"]["recall_micro_5"] = recall
        language_model_scores["report"]["CE"]["f1_micro_5"] = f1
        language_model_scores["report"]["CE"]["acc_5"] = acc """

    def compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports):
        """
        example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        """
        preds_gen_reports_np = np.array(preds_gen_reports)  # array of shape (14 x num_reports), 14 for 14 conditions
        preds_ref_reports_np = np.array(preds_ref_reports)  # array of shape (14 x num_reports)

        # convert label 1 to True and everything else (i.e. labels 0, 2, 3) to False
        # (effectively doing the label conversion as done by Nicolson, see doc string of compute_clinical_efficacy_scores for more details)
        preds_gen_reports_np = preds_gen_reports_np == 1
        preds_ref_reports_np = preds_ref_reports_np == 1

        tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)

        # sum up the TP, FP, FN and TN for each report (i.e. for each column)
        tp_example = tp.sum(axis=0)  # int array of shape (num_reports)
        fp_example = fp.sum(axis=0)  # int array of shape (num_reports)
        fn_example = fn.sum(axis=0)  # int array of shape (num_reports)
        tn_example = tn.sum(axis=0)  # int array of shape (num_reports)

        # compute the scores for each report
        precision_example = tp_example / (tp_example + fp_example)  # float array of shape (num_reports)
        recall_example = tp_example / (tp_example + fn_example)  # float array of shape (num_reports)
        f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)  # float array of shape (num_reports)
        acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)  # float array of shape (num_reports)

        # since there can be cases of zero division, we have to replace the resulting nan values with 0.0
        precision_example[np.isnan(precision_example)] = 0.0
        recall_example[np.isnan(recall_example)] = 0.0
        f1_example[np.isnan(f1_example)] = 0.0
        acc_example[np.isnan(acc_example)] = 0.0

        # finally, take the mean over the scores for all reports
        precision_example = float(precision_example.mean())
        recall_example = float(recall_example.mean())
        f1_example = float(f1_example.mean())
        acc_example = float(acc_example.mean())

        language_model_scores["report"]["CE"]["precision_example_all"] = precision_example
        language_model_scores["report"]["CE"]["recall_example_all"] = recall_example
        language_model_scores["report"]["CE"]["f1_example_all"] = f1_example
        language_model_scores["report"]["CE"]["acc_example_all"] = acc_example

    chexbert = get_chexbert()
    print("Get chexpert labels.")
    preds_gen_reports, preds_ref_reports = get_chexbert_labels_for_gen_and_ref_reports()

    print("Compute CE scores.")
    compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports)
    compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports)


def compute_language_model_scores(gen_and_ref_reports):
    print("Computing language model scores..")
    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
        nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

        for nlg_metric_name, score in nlg_scores.items():
            language_model_scores["report"][nlg_metric_name] = score

        compute_clinical_efficacy_scores(language_model_scores, gen_reports, ref_reports)

    def create_language_model_scores_dict():
        language_model_scores = {}

        # on report-level, we evalute on:
        # BLEU 1-4
        # METEOR
        # ROUGE-L
        # Cider-D
        # CE scores (P, R, F1, acc)
        language_model_scores["report"] = {f"bleu_{i}": None for i in range(1, 5)}
        language_model_scores["report"]["meteor"] = None
        language_model_scores["report"]["rouge"] = None
        language_model_scores["report"]["cider"] = None
        language_model_scores["report"]["CE"] = {
            # following Miura (https://arxiv.org/pdf/2010.10042.pdf), we evaluate the micro average CE scores over these 5 diseases/conditions:
            # "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"
            "precision_micro_5": None,
            "recall_micro_5": None,
            "f1_micro_5": None,
            "acc_5": None,

            # we additionally compute the micro average CE scores over all conditions
            "precision_micro_all": None,
            "recall_micro_all": None,
            "acc_all": None
        }

        # we also compute the CE scores for each of the 14 conditions individually
        for condition in CONDITIONS:
            language_model_scores["report"]["CE"][condition] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # following Nicolson (https://arxiv.org/pdf/2201.09405.pdf), we evaluate the example-based CE scores over all conditions
        # example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        language_model_scores["report"]["CE"]["precision_example_all"] = None
        language_model_scores["report"]["CE"]["recall_example_all"] = None
        language_model_scores["report"]["CE"]["f1_example_all"] = None
        language_model_scores["report"]["CE"]["acc_example_all"] = None

        return language_model_scores

    language_model_scores = create_language_model_scores_dict()

    compute_report_level_scores()

    return language_model_scores


def write_sentences_and_reports_to_file(
    gen_and_ref_reports,
    generated_sentences_and_reports_folder_path,
    overall_steps_taken,
):
    print("Writing example reports..")
    def write_reports():
        txt_file_name = os.path.join(
            generated_sentences_and_reports_folder_path,
            "generated_reports",
            f"generated_reports_step_{overall_steps_taken}",
        )

        with open(txt_file_name, "w") as f:
            for gen_report, ref_report in zip(
                generated_reports,
                reference_reports,
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("=" * 30)
                f.write("\n\n")

    
    num_generated_reports_to_save = NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"][:num_generated_reports_to_save]
    reference_reports = gen_and_ref_reports["reference_reports"][:num_generated_reports_to_save]
    
    write_reports()


def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    """
    Get a plot title like in the below example.
    1 region_set always contains 6 regions (except for region_set_5, which has 5 regions).
    The characters in the brackets represent the colors of the corresponding bboxes (e.g. b = blue),
    "nd" stands for "not detected" in case the region was not detected by the object detector.

    right lung (b), right costophrenic angle (g, nd), left lung (r)
    left costophrenic angle (c), cardiac silhouette (m), spine (y, nd)
    """
    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [
        region + f" ({color})" if cls_detect else region + f" ({color}, nd)"
        for region, color, cls_detect in zip(region_set, region_colors, class_detected)
    ]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): holds the generated sentences for all regions that were selected in the batch, i.e. of length "num_regions_selected_in_batch"
        selected_regions (Tensor[bool]): of shape [batch_size x 29], specifies for each region if it was selected to get a sentences generated (True) or not by the binary classifier for region selection.
        Ergo has exactly "num_regions_selected_in_batch" True values.
        num_img (int): specifies the image we are currently processing in the batch, its value is in the range [0, batch_size-1]
        region_index (int): specifies the region we are currently processing of a single image, its value is in the range [0, 28]

    Returns:
        str: generated sentence for region specified by num_img and region_index

    Implementation is not too easy to understand, so here is a toy example with some toy values to explain.

    generated_sentences_for_selected_regions = ["Heart is ok.", "Spine is ok."]
    selected_regions = [
        [False, False, True],
        [True, False, False]
    ]
    num_img = 0
    region_index = 2

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 29).
    The generated_sentences_for_selected_regions is of len 2, meaning num_regions_selected_in_batch = 2.
    Therefore, the selected_regions boolean tensor also has exactly 2 True values.

    (1) Flatten selected_regions:
        selected_regions_flat = [False, False, True, True, False, False]

    (2) Compute cumsum (to get an incrementation each time there is a True value):
        cum_sum_true_values = [0, 0, 1, 2, 2, 2]

    (3) Reshape cum_sum_true_values to shape of selected_regions
        cum_sum_true_values = [
            [0, 0, 1],
            [2, 2, 2]
        ]

    (4) Subtract 1 from tensor, such that 1st True value in selected_regions has the index value 0 in cum_sum_true_values,
        the 2nd True value has index value 1 and so on.
        cum_sum_true_values = [
            [-1, -1, 0],
            [1, 1, 1]
        ]

    (5) Index cum_sum_true_values with num_img and region_index to get the final index for the generated sentence list
        index = cum_sum_true_values[num_img][region_index] = cum_sum_true_values[0][2] = 0

    (6) Get generated sentence:
        generated_sentences_for_selected_regions[index] = "Heart is ok."
    """
    selected_regions_flat = selected_regions.reshape(-1)
    cum_sum_true_values = np.cumsum(selected_regions_flat)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


def get_generated_reports(generated_sentences_for_selected_regions, selected_regions, sentence_tokenizer, bertscore_threshold):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in remove_duplicate_generated_sentences to separate the generated sentences

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
        removed_similar_generated_sentences (List[Dict[str, List]): list of length batch_size containing dicts that map from one generated sentence to a list
        of other generated sentences that were removed because they were too similar. Useful for manually verifying if removing similar generated sentences was successful
    """
    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        # TODO:
        # the nested for loops below check each generated sentence with every other generated sentence
        # this is not particularly efficient, since e.g. generated sentences for the region "right lung" most likely
        # will never be similar to generated sentences for the region "abdomen"
        # thus, one could speed up these checks by only checking anatomical regions that are similar to each other

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

    bert_score = evaluate.load("bertscore")

    generated_reports = []
    removed_similar_generated_sentences = []
    curr_index = 0

    for selected_regions_single_image in selected_regions:
        # sum up all True values for a single row in the array (corresponing to a single image)
        num_selected_regions_single_image = np.sum(selected_regions_single_image)

        # use curr_index and num_selected_regions_single_image to index all generated sentences corresponding to a single image
        gen_sents_single_image = generated_sentences_for_selected_regions[
            curr_index: curr_index + num_selected_regions_single_image
        ]

        # update curr_index for next image
        curr_index += num_selected_regions_single_image

        # concatenate generated sentences of a single image to a continuous string gen_report_single_image
        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return generated_reports, removed_similar_generated_sentences


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 29 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 29]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()

def evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path):
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    # whilst iterating over the validation loader, save the (all, normal, abnormal) generated and reference sentences in the respective lists
    # the list under the key "num_generated_sentences_per_image" will hold integers that represent how many sentences were generated for each image
    # this is useful to be able to get all generated and reference sentences that correspond to the same image
    # (since we append all generated and reference sentences to the "generated_sentences" and "reference_sentences" lists indiscriminately, this information would be lost otherwise)
    """gen_and_ref_sentences = { #dunno if need this but maybe if we want to remove identicle sentences though given we are not genrerating on a per region basis we should be ok. 
        "generated_sentences": [],
        "reference_sentences": [],
        "num_generated_sentences_per_image": []
    }"""

    # and of course the generated and reference reports, and additionally keep track of the generated sentences
    # that were removed because they were too similar to other generated sentences (only as a sanity-check/for writing to file)
    gen_and_ref_reports = {
        "scan_ids" : [],
        "generated_reports": [],
        #"removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    # gen_sentences_with_corresponding_regions will be a list[list[tuple[str, str]]],
    # where len(outer_list) will be NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE
    # the inner list will correspond to a single generated report / single image and hold tuples of (region_name, generated_sentence),
    # i.e. each region that was selected for that single image, with the corresponding generated sentence (without any removal of similar generated sentences)
    #
    # gen_sentences_with_corresponding_regions will be used such that each generated sentences in a generated report can be directly attributed to a region
    # because this information gets lost when we concatenated generated sentences
    # this is only used to get more insights into the generated reports that are written to file
    #gen_sentences_with_corresponding_regions = []

    # we also want to plot a couple of images
    #num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    # used in function get_generated_reports
    sentence_tokenizer = spacy.load("en_core_web_trf")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            # since generating sentences takes some time, we limit the number of batches used to compute bleu/rouge-l/meteor
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            #image_targets = batch["image_targets"]
            #region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 29]
            #scan_ids = batch["mimic_image_file_path"]
            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            #reference_sentences = batch["reference_sentences"]

            # List[str] that holds the reference report for the images in the batch
            reference_reports = batch["reference_reports"]

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.generate(
                        images.to(device, non_blocking=True),
                        max_length=MAX_NUM_TOKENS_GENERATE,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Generation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
            beam_search_output = output
            #selected_regions = selected_regions.detach().cpu().numpy()

            # generated_sents_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_reports = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sents_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sents_for_selected_regions)
           
            """generated_reports, removed_similar_generated_sentences = get_generated_reports(
                generated_reports
                sentence_tokenizer,
                BERTSCORE_SIMILARITY_THRESHOLD
            )""" #come back to this
            #gen_and_ref_reports["scan_id"].extend(scan_ids)
            gen_and_ref_reports["generated_reports"].extend(generated_reports)
            gen_and_ref_reports["reference_reports"].extend(reference_reports)
            #gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

            #if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
            #    update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions)

            """ if num_batch < num_batches_to_process_for_image_plotting:
                plot_detections_and_sentences_to_tensorboard(
                    writer,
                    num_batch,
                    overall_steps_taken,
                    images,
                    image_targets,
                    reference_reports,
                    generated_reports,
                ) """

    write_sentences_and_reports_to_file(
        gen_and_ref_reports,
        generated_sentences_and_reports_folder_path,
        overall_steps_taken,
    )

    language_model_scores = compute_language_model_scores(gen_and_ref_reports)

    return language_model_scores
