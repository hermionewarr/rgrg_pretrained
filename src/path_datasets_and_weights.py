"""
Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.

Note that we only need the report txt files from MIMIC-CXR, which are in the file mimic-cxr-report.zip at
https://physionet.org/content/mimic-cxr/2.0.0/.

path_full_dataset specifies the path where the folder will be created (by module src/dataset/create_dataset.py) that will hold the
train, valid, test files, which will be used for training, evaluation and testing. See doc string of create_dataset.py for more information.
That means in my case, "/u/home/tanida/datasets/" should already exist as a directory, and the folder "dataset-with-reference-reports" would
be created in that directory by the module.

path_chexbert_weights specifies the path to the weights of the CheXbert labeler needed to extract the disease labels from the generated and reference reports.
The weights can be downloaded here: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download

path_runs_* specify the directories where all the run folders (containing checkpoints, tensorboard files etc.) will be created
when training the object detector and full model (with and without the language model), respectively.
That means the directories specified by path_runs_* should already exist before starting the training.

path_test_set_evaluation_scores_txt_files will be the path where the txt files will be stored which contain the test set scores.
"""

path_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr"
path_mimic_cxr_jpg = "/u/home/tanida/datasets/mimic-cxr-jpg"
path_full_dataset = "/u/home/tanida/datasets/dataset-with-reference-reports"
path_chexbert_weights = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/CheXbert/src/models/chexbert.pth"
path_runs_object_detector = "/u/home/tanida/runs/object_detector"
path_runs_full_model = "/u/home/tanida/runs/full_model"
path_test_set_evaluation_scores_txt_files = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src"

DATA_PATH = "/home/hermione/Documents/data/"
MODEL_INFO_PATH = "/home/hermione/Documents/VLP/TUM/rgrg_pretrained/" #/home/hermione/Documents/VLP/rgrg/src/runs
path_chest_imagenome = DATA_PATH+"rgrg/chest-imagenome-dataset-1.0.0/" #hmm do we need this?
path_mimic_cxr = DATA_PATH+"physionet.org/files/mimic-cxr/2.0.0/"
path_mimic_cxr_jpg = DATA_PATH+"physionet.org/files/mimic-cxr-jpg/2.0.0/"
path_full_dataset = DATA_PATH+"rgrg/dataset-with-reference-reports_small"
path_chexbert_weights = DATA_PATH+"rgrg/chexbert.pth"
#path_runs_object_detector = MODEL_INFO_PATH+"src/runs/object_detector"
path_runs_full_model = MODEL_INFO_PATH+"src/runs/full_model"
path_test_set_evaluation_scores_txt_files = MODEL_INFO_PATH+"src/region-guided-chest-x-ray-report-generation"
model_trained_checkpoint = "/home/hermione/Documents/VLP/TUM/rgrg/full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt"