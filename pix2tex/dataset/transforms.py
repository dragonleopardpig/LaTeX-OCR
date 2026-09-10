import albumentations as alb
import cv2
from albumentations.pytorch import ToTensorV2

train_transform = alb.Compose(
    [
        alb.Compose(
            [
                alb.Affine(
                    scale=(0.85, 1.0),
                    translate_percent=0,
                    rotate=(-1, 1),
                    mode=cv2.BORDER_CONSTANT,
                    cval=(255, 255, 255),
                    interpolation=cv2.INTER_CUBIC,
                    p=1,
                ),
                alb.GridDistortion(
                    distort_limit=0.1,
                    border_mode=cv2.BORDER_CONSTANT,
                    interpolation=cv2.INTER_CUBIC,
                    value=(255, 255, 255),
                    p=.5,
                ),
            ],
            p=.15,
        ),
        # alb.InvertImg(p=.15),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.3),
        alb.GaussNoise(var_limit=(0, 10), p=.2),
        alb.RandomBrightnessContrast(
            brightness_limit=.05,
            contrast_limit=(-.2, 0),
            brightness_by_max=True,
            p=0.2,
        ),
        alb.ImageCompression(quality_range=(95, 100), p=.3),
        alb.ToGray(p=1),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)
test_transform = alb.Compose(
    [
        alb.ToGray(p=1),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)
