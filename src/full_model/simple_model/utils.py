import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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


