from collections import OrderedDict
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
# from torchinfo import summary

import sys
sys.path.append("/home/hermione/Documents/VLP/TUM/rgrg_pretrained/")
from src.object_detector.custom_roi_heads import CustomRoIHeads
from src.object_detector.custom_rpn import CustomRegionProposalNetwork
from src.object_detector.image_list import ImageList


class ObjectDetector(nn.Module):
    """
    Implements Faster R-CNN with a classifier pre-trained on chest x-rays as the backbone.
    The implementation differs slightly from the PyTorch one.

    During training, the model expects both the input image tensor as well as the targets.

    The input image tensor is expected to be a tensor of shape [batch_size, 1, H, W], with H = W (which will most likely be 512).
    This differs form the PyTorch implementation, where the input images are expected to be a list of tensors (of different shapes).
    We apply transformations before inputting the images into the model, whereas the PyTorch implementation applies transformations
    after the images were inputted into the model.

    The targets is expected to be a list of dictionaries, with each dict containing:
        - boxes (FloatTensor[N, 4]): the gt boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The PyTorch implementation returns a Dict[Tensor] containing the 4 losses in train mode, and a List[Dict[Tensor]] containing
    the detections for each image in eval mode.

    My implementation returns different things depending on if the object detector is trained/evaluated in isolation,
    or if it's trained/evaluated as part of the full model.

    Please check the doc string of the forward method for more details.
    """

    def __init__(self, return_feature_vectors=False):
        super().__init__()
        # boolean to specify if feature vectors should be returned after roi pooling inside RoIHeads
        self.return_feature_vectors = return_feature_vectors

        # 29 classes for 29 anatomical regions + background (defined as class 0)
        self.num_classes = 30

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # use only the feature extractor of the pre-trained classification model
        # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        #self.backbone_end = nn.Sequential(*list(resnet.children())[-2:])
        # FasterRCNN needs to know the number of output channels of the backbone
        # for ResNet-50, it's 2048 (with feature maps of size 16x16)
        self.backbone.out_channels = 2048

        #for name, param in self.backbone.named_parameters(): #check trainable
        #    print(f"Parameter: {name}, Requires grad: {param.requires_grad}") #yep all true

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        Args:
            images (Tensor): images to be processed of shape [batch_size, 1, 512, 512] (gray-scale images of size 512 x 512)
            targets (List[Dict[str, Tensor]]): list of batch_size dicts, where a single dict contains the fields:
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format
                - labels (Int64Tensor[N]): the class label for each ground-truth box

        Returns:
            (1) If object detector is trained/evaluated in isolation, then self.return_feature_vectors should be False and it returns
                (I) in train mode:
                    - losses (Dict[Tensor]), which contains the 4 object detector losses
                (II) in eval mode:
                    - losses (Dict[Tensor]). If targets == None (i.e. during inference), then (val) losses will be an empty dict
                    - detections (List[Dict[str, Tensor]]), which are the predictions for each input image.

            (2) If object detector is trained/evaluated as part of the full model, then self.return_feature_vectors should be True and it returns
                (I) in train mode:
                    - losses
                    - top_region_features (FloatTensor(batch_size, 29, 1024)):
                        - the region visual features with the highest scores for each region and for each image in the batch
                        - these are needed to train the binary classifiers and language model
                    - class_detected (BoolTensor(batch_size, 29)):
                        - boolean is True if a region/class had the highest score (i.e. was top-1) for at least 1 RoI box
                        - if the value is False for any class, then this means the object detector effectively did not detect the region,
                        and it is thus filtered out from the next modules in the full model
                (II) in eval mode:
                    - losses. If targets == None (i.e. during inference), then (val) losses will be an empty dict
                    - detections
                    - top_region_features
                    - class_detected
        
        if targets is not None:
            self._check_targets(targets)
        """
        features = self.backbone(images)

        # if we return region features, then we train/evaluate the full model (with object detector as one part of it)
        return features
    
