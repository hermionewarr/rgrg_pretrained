import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            # mimic_image_file_path is the 1st column of the dataframes
            #image_path = self.dataset_df.iloc[index, 0]
            image_path = self.dataset_df[index]["mimic_image_file_path"]

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # bbox_coordinates (List[List[int]]) is the 2nd column of the dataframes
            #bbox_coordinates = self.dataset_df.iloc[index, 1]

            # bbox_labels (List[int]) is the 3rd column of the dataframes
            report_labels = self.dataset_df[index]["reference_report"]
            #condition = torch.tensor([50256, 47, 293, 1523, 27848, 4241, 50256], device=self.device) #plueral effusion
        
            labels = np.where(report_labels=='Pleural Effusion', 1,0)

            # apply transformations to image, bboxes and label
            transformed = self.transforms(image=image)

            transformed_image = transformed["image"]
            #transformed_bboxes = transformed["bboxes"]
            #transformed_bbox_labels = transformed["class_labels"]

            sample = {
                "image": transformed_image,
                "labels": torch.tensor(labels, dtype=torch.int64),
            }

        except Exception:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample
