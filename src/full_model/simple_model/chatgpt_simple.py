
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # Choose the GPT-2 model size
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

def load_resnet():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
    resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # use only the feature extractor of the pre-trained classification model
    # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    #self.backbone_end = nn.Sequential(*list(resnet.children())[-2:])
    # FasterRCNN needs to know the number of output channels of the backbone
    # for ResNet-50, it's 2048 (with feature maps of size 16x16)
    backbone.out_channels = 2048
    return backbone

# Define a custom dataset to load image features and tokenized captions
class ImageCaptionDataset(Dataset):
    def __init__(self, image_features, captions, tokenizer, max_length):
        self.image_features = image_features
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        inputs = self.tokenizer.encode(" ".join(caption), add_special_tokens=True, max_length=self.max_length, truncation=True)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "images": torch.tensor(self.image_features[idx], dtype=torch.float32),
        }

# Example usage:
# Replace 'image_features' and 'captions' with your own data
# 'image_features' should be a list of image features (e.g., numpy arrays)
# 'captions' should be a list of tokenized captions
data = pd.read_csv("/home/hermione/Documents/data/rgrg/dataset-with-reference-reports/train_white_square.csv")
image_paths = data["mimic_image_file_path"]
ref_reports = data["reference_report"]

def load_images(image_paths):
    images = []
    IMAGE_INPUT_SIZE = 512
    mean = 0.471
    std = 0.302
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    for i in range(10):
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        images.append(transforms(image))
    return images

images = load_images(image_paths) #[torch.randn(2048), torch.randn(2048)]  # Example image features
captions = [["a", "cat", "on", "the", "mat"], ["a", "dog", "in", "the", "park"]]  # Example tokenized captions

# Create a dataset instance
dataset = ImageCaptionDataset(image_features=images, captions=captions, tokenizer=tokenizer, max_length=32)

# Create a data loader for training
batch_size = 2  # Adjust as needed
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a loss function (e.g., cross-entropy) and an optimizer for fine-tuning
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=1e-5)
image_encoder = load_resnet()

num_epochs = 10
# Fine-tune the model
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs = batch["input_ids"]
        image_features = batch["images"]
        
        # Forward pass
        image_features = image_encoder(images)
        outputs = gpt2_model(input_ids=inputs, inputs_embeds=image_features)
        logits = outputs.logits
        
        # Calculate the loss
        loss = loss_fn(logits.view(-1, logits.shape[-1]), inputs.view(-1))
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

