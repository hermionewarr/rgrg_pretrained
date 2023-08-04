from ast import literal_eval
import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm

import sys
sys.path.append("/home/hermione/Documents/VLP/TUM/rgrg_pretrained/")
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset

from src.full_model.evaluate_full_model.evaluate_language_model import (
    compute_language_model_scores
)
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer
from src.path_datasets_and_weights import path_full_dataset, path_runs_full_model, path_test_set_evaluation_scores_txt_files

# specify the checkpoint you want to evaluate by setting "RUN" and "CHECKPOINT"
RUN = 21
CHECKPOINT = "checkpoint_val_loss_2.060_overall_steps_258410.pt"#"checkpoint_val_loss_1.778_overall_steps_501150.pt"
BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 10
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 100
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 100
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

"""
Folder specified by path_test_set_evaluation_scores_txt_files will have these files after test set evaluation:

    - final_scores.txt
    - generated_sentences.txt
    - generated_abnormal_sentences.txt
    - generated_reports.txt
"""
final_scores_txt_file = os.path.join(path_test_set_evaluation_scores_txt_files, "final_scores.txt")


def write_all_scores_to_file(
    language_model_scores
):

    def write_clinical_efficacy_scores(ce_score_dict):
        """
        ce_score_dict is of the structure:

        {
            precision_micro_5: ...,
            precision_micro_all: ...,
            precision_example_all: ...,
            recall_micro_5: ...,
            recall_micro_all: ...,
            recall_example_all: ...,
            f1_micro_5: ...,
            f1_micro_all: ...,
            f1_example_all: ...,
            acc_micro_5: ...,
            acc_micro_all: ...,
            acc_example_all: ...,
            condition_1 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            condition_2 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            ...,
            condition_14 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            }
        }

        where the "..." after the 4 metrics are the corresponding scores,
        and condition_* are from the 14 conditions in src/CheXbert/src/constants.py
        """
        for k, v in ce_score_dict.items():
            if k.startswith("precision") or k.startswith("recall") or k.startswith("f1") or k.startswith("acc"):
                with open(final_scores_txt_file, "a") as f:
                    f.write(f"report_CE_{k}: {v:.5f}\n")
            else:
                # k is a condition
                condition_name = "_".join(k.lower().split())
                for metric, score in ce_score_dict[k].items():
                    with open(final_scores_txt_file, "a") as f:
                        f.write(f"report_CE_{condition_name}_{metric}: {score:.5f}\n")

    def write_language_model_scores():
        """
        language_model_scores is a dict with keys:
            - all: for all generated sentences
            - normal: for all generated sentences corresponding to normal regions
            - abnormal: for all generated sentences corresponding to abnormal regions
            - report: for all generated reports
            - region: for generated sentences per region
        """
        #for subset in language_model_scores:
        for metric, score in language_model_scores["report"].items():
            if metric == "CE":
                ce_score_dict = language_model_scores["report"]["CE"]
                write_clinical_efficacy_scores(ce_score_dict)
            else:
                with open(final_scores_txt_file, "a") as f:
                    f.write(f"{metric}: {score:.5f}\n")

    with open(final_scores_txt_file, "a") as f:
        f.write(f"Run: {RUN}\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"BertScore for removing similar generated sentences: {BERTSCORE_SIMILARITY_THRESHOLD}\n")
        f.write(f"Num beams: {NUM_BEAMS}\n")

    write_language_model_scores()


def write_sentences_and_reports_to_file_for_test_set(
    gen_and_ref_reports,
):
    def write_reports():
        txt_file_name = os.path.join(path_test_set_evaluation_scores_txt_files, "generated_reports.txt")

        with open(txt_file_name, "a") as f:
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


def evaluate_language_model_on_test_set(model, test_loader, tokenizer):
    def iterate_over_test_loader(test_loader):
        # to recover from potential OOMs
        oom = False

        # used in function get_generated_reports
        sentence_tokenizer = spacy.load("en_core_web_trf")

        with torch.no_grad():
            for num_batch, batch in tqdm(enumerate(test_loader)):

                images = batch["images"]  # shape [batch_size x 1 x 512 x 512]

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

                        with open(final_scores_txt_file, "a") as f:
                            f.write("Generation:\n")
                            f.write(f"OOM at batch number {num_batch}.\n")
                            f.write(f"Error message: {str(e)}\n\n")
                    else:
                        raise e

                if oom:
                    # free up memory
                    torch.cuda.empty_cache()
                    oom = False
                    continue

                beam_search_output = output
                    
                generated_reports = tokenizer.batch_decode(
                    beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                gen_and_ref_reports["generated_reports"].extend(generated_reports)
                gen_and_ref_reports["reference_reports"].extend(reference_reports)

    # and of course the generated and reference reports, and additionally keep track of the generated sentences
    # that were removed because they were too similar to other generated sentences (only as a sanity-check/for writing to file)
    gen_and_ref_reports = {
        "generated_reports": [],
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
    gen_sentences_with_corresponding_regions = []

    log.info("Test loader: generating sentences/reports...")
    iterate_over_test_loader(test_loader)
    log.info("Test loader: generating sentences/reports... DONE.")

    """ log.info("Test loader 2: generating sentences/reports...")
    iterate_over_test_loader(test_2_loader)
    log.info("Test loader 2: generating sentences/reports... DONE.") """

    write_sentences_and_reports_to_file_for_test_set(
        gen_and_ref_reports
    )

    with open(final_scores_txt_file, "a") as f:
        f.write(f"Num generated reports: {len(gen_and_ref_reports['generated_reports'])}\n")

    log.info("Computing language_model_scores...")
    language_model_scores = compute_language_model_scores(gen_and_ref_reports)
    log.info("Computing language_model_scores... DONE.")

    return language_model_scores


def evaluate_model_on_test_set(model, test_loader, tokenizer):
    language_model_scores = evaluate_language_model_on_test_set(model, test_loader, tokenizer)
    write_all_scores_to_file(
        language_model_scores
    )
    return


def get_model():
    checkpoint = torch.load(
        os.path.join(path_runs_full_model, f"run_{RUN}", "checkpoints", f"{CHECKPOINT}"),
        map_location=torch.device("cpu"),
    )

    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

    # pretrain_without_lm_model=True, since we don't need to compute the language model loss (see forward method of full model)
    # we evaluate the language model in function evaluate_language_model_on_test_set by generating sentences/reports based on input images
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint

    return model


def get_data_loaders(tokenizer, test_dataset_complete):
    custom_collate_test = CustomCollator(
        tokenizer=tokenizer, is_val_or_test=True, pretrain_without_lm_model=False
    )

    test_loader = DataLoader(
        test_dataset_complete,
        collate_fn=custom_collate_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
    )

    return test_loader


def get_transforms():
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # don't apply data augmentations to test set
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return test_transforms


def get_tokenized_dataset(tokenizer, raw_test_dataset):
    def tokenize_function(example):
        phrases = example["reference_report"]  # (str)  - make List[str]?
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrases + eos_token]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_test_dataset = raw_test_dataset.map(tokenize_function)
    #tokenized_test_2_dataset = raw_test_2_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #   - reference_report (str)

    return tokenized_test_dataset #, tokenized_test_2_dataset


def get_dataset():
    usecols = [
        "mimic_image_file_path",
        "reference_report"
    ]

    datasets_as_dfs = {}
    datasets_as_dfs["test"] = pd.read_csv(os.path.join(path_full_dataset, "test.csv"), usecols=usecols)
    #datasets_as_dfs["test-2"] = pd.read_csv(os.path.join(path_full_dataset, "test-2.csv"), usecols=usecols, converters=converters)
    total_num_samples_test = len(datasets_as_dfs["test"])
    
    #raw_test_2_dataset = Dataset.from_pandas(datasets_as_dfs["test-2"])
    new_num_samples_test = int(0.01* total_num_samples_test)

    log.info(f"Test: {new_num_samples_test} images")

    datasets_as_dfs["test"] = datasets_as_dfs["test"].sample(n=new_num_samples_test, random_state=SEED) #[:new_num_samples_train]
 
    raw_test_dataset = Dataset.from_pandas(datasets_as_dfs["test"])
    return raw_test_dataset  #, raw_test_2_dataset


def main():
    raw_test_dataset= get_dataset()

    # note that we don't actually need to tokenize anything (i.e. we don't need the input ids and attention mask),
    # because we evaluate the language model on it's generation capabilities (for which we only need the input images)
    # but since the custom dataset and collator are build in a way that they expect input ids and attention mask
    # (as they were originally made for training the model),
    # it's better to just leave it as it is instead of adding unnecessary complexity
    tokenizer = get_tokenizer()
    tokenized_test_dataset = get_tokenized_dataset(tokenizer, raw_test_dataset)

    test_transforms = get_transforms()

    test_dataset_complete = CustomDataset("test", tokenized_test_dataset, test_transforms, log)
    #test_2_dataset_complete = CustomDataset("test", tokenized_test_2_dataset, test_transforms, log)

    test_loader= get_data_loaders(tokenizer, test_dataset_complete)

    model = get_model()

    evaluate_model_on_test_set(model, test_loader, tokenizer)


if __name__ == "__main__":
    main()
