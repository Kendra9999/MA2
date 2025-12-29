import numpy as np
from monai.transforms import (
    ScaleIntensityd,
    SpatialPadd,
    Orientationd,
    Spacingd,
    Compose,
    LoadImaged,
    RandGaussianNoised,
    Rand3DElasticd,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandSpatialCropd,
    RandAffined,
    EnsureTyped,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
)

from .transforms import RandSimulateLowResolutiond_per_axis


def get_mae_train_transforms(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"], pixdim=spacing, mode=("bilinear")
            ),
            ScaleIntensityd(keys="image"),
            SpatialPadd(keys=["image"], 
                        spatial_size=patch_size,
                        mode="constant"),
            RandSpatialCropd(
                keys=["image"],
                roi_size=patch_size,
                random_size=False,
            ),
            RandGaussianNoised(keys=["image"], prob=0.33),
            RandBiasFieldd(
                keys=["image"], prob=0.33, coeff_range=(0.0, 0.075)
            ),
            RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
            RandAdjustContrastd(keys=["image"], prob=0.33),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.33,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),
            RandGaussianSharpend(
                keys=["image"], 
                prob=0.33,
                sigma1_x=(0.0, 3.0), sigma1_y=(0.0, 3.0), sigma1_z=(0.0, 3.0),
                sigma2_x=(0.0, 1.0), sigma2_y=(0.0, 1.0), sigma2_z=(0.0, 1.0),
                alpha=(1, 30),
            ),
            # Simulate low resolution (some MRI cases):
            RandSimulateLowResolutiond_per_axis(keys=["image"]),
            # Apply elastic deformation:
            Rand3DElasticd(
                keys=["image"],
                prob=0.333,
                mode=['bilinear'],
                sigma_range=(5.0, 8.0),
                magnitude_range=(10, 50),
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            ),
            RandAffined(
                keys=["image"],
                prob=0.98,
                mode=("bilinear"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.25, 0.25, 0.25),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=patch_size,
                padding_mode='zeros',
            ),
            RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            ScaleIntensityd(keys="image"),
        ]
    )

    return train_transforms