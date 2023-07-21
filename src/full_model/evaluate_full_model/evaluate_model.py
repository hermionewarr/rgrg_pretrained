"""
This module contains all functions used to evaluate the full model.

The (main) function evaluate_model of this module is called by the function train_model in train_full_model.py
every K steps and also at the end of every epoch.

The K is specified by the EVALUATE_EVERY_K_STEPS variable in run_configurations.py

evaluate_model and its sub-functions evaluate among other things:

    - total val loss as well as the val losses of each individual module (i.e. model component)
    - object detector:
        - average IoU of region (ideally 1.0 for every region)
        - average num detected regions per image (ideally 29.0)
        - average num each region is detected in an image (ideally 1.0 for every region)
    - binary classifier region selection:
        - precision, recall, f1 for all regions, regions that have gt = normal (i.e. the region was considered normal by the radiologist),
        regions that have gt = abnormal (i.e. the region was considered abnormal by the radiologist)
    - binary classifier region abnormal detection:
        - precision, recall, f1 for all regions
    - language model (is evaluated in separate evaluate_language_model.py module):
        - see doc string of evaluate_language_model.py for information on metrics
"""

import os

import torch
import torchmetrics
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.evaluate_language_model import evaluate_language_model
from src.full_model.run_configurations import EPOCH_TO_EVAL_LANG_ON, PRETRAIN_WITHOUT_LM_MODEL

cuda_device_to_see = 0
os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_device_to_see}'
device = torch.device(f"cuda:{cuda_device_to_see}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device_to_see)
print("device: ", device)

def write_all_losses_and_scores_to_tensorboard(
    writer,
    overall_steps_taken,
    train_losses_dict,
    val_losses_dict,
    language_model_scores,
    current_lr,
    epoch
):
    def write_losses():
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "loss",
                {f"{loss_type}_train": train_losses_dict[loss_type], f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )

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
                writer.add_scalar(f"CE/{k}", v, overall_steps_taken)
            else:
                # k is a condition
                condition_name = "_".join(k.lower().split())
                for metric, score in ce_score_dict[k].items():
                    writer.add_scalar(f"CE/{condition_name}/{metric}", score, overall_steps_taken)

    def write_language_model_scores():
        """
        language_model_scores is a dict with keys:
            - all: for all generated sentences
            - normal: for all generated sentences corresponding to normal regions
            - abnormal: for all generated sentences corresponding to abnormal regions
            - report: for all generated reports
            - region: for generated sentences per region
        """
        print(language_model_scores)
        #print(language_model_scores.keys())
        #for subset in language_model_scores:
        for metric, score in language_model_scores["report"].items(): #[subset]
                if metric == "CE":
                    ce_score_dict = language_model_scores["report"]["CE"]
                    write_clinical_efficacy_scores(ce_score_dict)
                else:
                    writer.add_scalar(f"{metric}", score, overall_steps_taken)

    write_losses()

    if epoch >= EPOCH_TO_EVAL_LANG_ON:
        write_language_model_scores()

    writer.add_scalar("lr", current_lr, overall_steps_taken)

def get_val_losses(model, val_dl,
                   log_file,
                   epoch):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_losses_dict (Dict): holds different val losses of the different modules as well as the total val loss
        obj_detector_scores (Dict): holds scores of the average IoU per Region, average number of detected regions per image,
        average number each region is detected in an image
        region_selection_scores (Dict): holds precision and recall scores for all, normal and abnormal sentences
        region_abnormal_scores (Dict): holds precision and recall scores for all sentences
    """
    val_losses_dict = {
        "total_loss": 0.0
    }

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    num_images = 0

    # for normalizing the val losses
    steps_taken = 0

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl)):
            images = batch["images"]
            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=False)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            input_ids = input_ids.to(device, non_blocking=False)
            attention_mask = attention_mask.to(device, non_blocking=False)

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images, input_ids, attention_mask)
            
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True
                    
                    with open(log_file, "a") as f:
                        f.write("Evaluation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                    
                else:
                    raise e
            
            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                num_images -= batch_size
                continue
            
            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:

                with open(log_file, "a") as f:
                    f.write("Evaluation:\n")
                    f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")
                
                num_images -= batch_size

                continue

            val_losses_dict["total_loss"] += output.item()
            steps_taken += 1
            
    # normalize the val losses by steps_taken
    val_losses_dict["total_loss"] /= steps_taken

    return val_losses_dict


def evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, optimizer, scaler, writer, tokenizer, run_params, generated_sentences_and_reports_folder_path):

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_device_to_see}'
    device = torch.device(f"cuda:{cuda_device_to_see}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(cuda_device_to_see)
    print("device: ", device)

    epoch = run_params["epoch"]
    steps_taken = run_params["steps_taken"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    # normalize all train losses by steps_taken
    train_losses_dict["total_loss"] /= steps_taken

    val_losses_dict = get_val_losses(model, val_dl, log_file, epoch)
    
    # the language model will generate gibberish in the beginning, so no need to evaluate it for first 100000 steps
    # (you may need to change this number based on the batch size you use, we used a small batch size of 2 for resource constraints)
    if epoch >= EPOCH_TO_EVAL_LANG_ON: #100000
        print("steps:", overall_steps_taken)
        language_model_scores = evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path)
    else:
        language_model_scores = None

    current_lr = float(optimizer.param_groups[0]["lr"])

    write_all_losses_and_scores_to_tensorboard(
        writer,
        overall_steps_taken,
        train_losses_dict,
        val_losses_dict,
        language_model_scores,
        current_lr,
        epoch
    )
    total_val_loss = val_losses_dict["total_loss"]

    # decrease lr if total_val_loss has not decreased after certain number of evaluations
    lr_scheduler.step(total_val_loss)

    # save model every time the val loss has decreased
    if total_val_loss < run_params["lowest_val_loss"]:
        run_params["lowest_val_loss"] = total_val_loss
        run_params["best_epoch"] = epoch
        
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_val_loss_{total_val_loss:.3f}_overall_steps_{overall_steps_taken}.pt")
        
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "current_epoch": epoch,
            "overall_steps_taken": overall_steps_taken,
            "lowest_val_loss": total_val_loss,
        }

        torch.save(checkpoint, save_path)
    
