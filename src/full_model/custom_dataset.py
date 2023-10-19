#pairing this down to remove boxes

import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms, log):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms
        self.log = log

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # get the image_path for potential logging in except block
        image_path = self.tokenized_dataset[index]["mimic_image_file_path"]

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            input_ids = self.tokenized_dataset[index]["input_ids"]  # List[List[int]]
            attention_mask = self.tokenized_dataset[index]["attention_mask"]  # List[List[int]]

                # same for the reference_report
            reference_report = self.tokenized_dataset[index]["reference_report"]  # str

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # apply transformations to image, bbox_coordinates and bbox_labels
            transformed = self.transforms(image=image)

            transformed_image = transformed["image"]

            sample = {
                "image": transformed_image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "reference_report" : reference_report,
            }

   

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None
        #print(sample)
        return sample
