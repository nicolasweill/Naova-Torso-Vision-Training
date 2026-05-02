from __future__ import annotations

from dataclasses import dataclass

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class AugConfig:
    enabled: bool = True
    horizontal_flip: float = 0.5
    random_brightness_contrast: float = 0.3
    hue_saturation_value: float = 0.3
    random_scale_min: float = 0.5
    random_scale_max: float = 1.5


_BBOX_PARAMS = A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.1,
    clip=True,
)


def get_train_transforms(cfg: AugConfig, input_size: tuple[int, int]) -> A.Compose:
    h, w = input_size
    transforms: list = []

    if cfg.enabled:
        transforms += [
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.RandomBrightnessContrast(p=cfg.random_brightness_contrast),
            A.HueSaturationValue(p=cfg.hue_saturation_value),
        ]

    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(transforms, bbox_params=_BBOX_PARAMS)


def get_val_transforms(input_size: tuple[int, int]) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )
