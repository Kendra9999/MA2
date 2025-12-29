import copy
import numpy as np
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform
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

from .simulate_noncontrast import simulate_noncontrast
from .intensity_transform import intensity_transform, inverse_intensity
from .synthetic_data import generate_volume

class RandSimulateLowResolutiond_per_axis(Transform):
    """
    Simulate low resolution by zooming in the image along each axis.
    """
    def __init__(self, keys: list[str], prob: float = 0.5, zoom_range: tuple = (0.25, 1.0)):
        self.keys = keys
        self.prob = min(max(prob, 0.0), 1.0)
        self.zoom_range = zoom_range

    def __call__(self, data):
        d = dict(data)

        if np.random.uniform() < self.prob:
            scale = tuple(np.random.uniform(self.zoom_range[0], self.zoom_range[1], size=3).tolist())

            for key in self.keys:
                ori_img = d[key].unsqueeze(0)
                down_img = F.interpolate(ori_img, scale_factor=scale, mode='nearest')
                up_img = F.interpolate(down_img, size=ori_img.shape[2:], mode='trilinear', align_corners=False)
                d[key] = up_img.squeeze(0)

        return d


class RandIntensityTransformd(Transform):
    def __init__(self, keys: list[str], prob: list[float] = [0.5,0.5,0.0,0.0]):
        self.keys = keys
        self.prob = [min(max(p, 0.0), 1.0) for p in prob]
        assert sum(self.prob) == 1.0

    def __call__(self, data):
        d = dict(data)

        p = np.random.uniform()

        for key in self.keys:
            if p < self.prob[0]:
                pass
            elif p < self.prob[0]+self.prob[1]:
                d[key] = simulate_noncontrast(d[key], d["label"])
            elif p < self.prob[0]+self.prob[1]+self.prob[2]:
                d[key] = intensity_transform(d[key])
            else:
                d[key] = inverse_intensity(d[key])

        return d


class GenerateVolumed(Transform):
    def __init__(self, source_key: str, target_key: str):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        d = dict(data)

        source = d[self.source_key]
        
        target = generate_volume(source.squeeze(0)).unsqueeze(0)

        d[self.target_key] = MetaTensor.ensure_torch_and_prune_meta(
                                    target, copy.deepcopy(source.meta))

        del d[self.source_key]

        return d


def get_train_transforms(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            ScaleIntensityd(keys="image"),
            SpatialPadd(keys=["image", "label"], 
                        spatial_size=patch_size,
                        mode="constant"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=patch_size,
                random_size=False,
            ),
            RandIntensityTransformd(keys=["image"], prob=[0.5,0.5,0.0,0.0]),
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
                keys=["image", "label"],
                prob=0.333,
                mode=['bilinear', 'nearest'],
                sigma_range=(5.0, 8.0),
                magnitude_range=(10, 50),
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.25, 0.25, 0.25),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=patch_size,
                padding_mode='zeros',
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            ScaleIntensityd(keys="image"),
        ]
    )

    return train_transforms



def get_train_transforms_nointensity(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            ScaleIntensityd(keys="image"),
            SpatialPadd(keys=["image", "label"], 
                        spatial_size=patch_size,
                        mode="constant"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=patch_size,
                random_size=False,
            ),
            # RandIntensityTransformd(keys=["image"], prob=[0.5,0.5,0.0,0.0]),
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
                keys=["image", "label"],
                prob=0.333,
                mode=['bilinear', 'nearest'],
                sigma_range=(5.0, 8.0),
                magnitude_range=(10, 50),
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.25, 0.25, 0.25),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=patch_size,
                padding_mode='zeros',
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            ScaleIntensityd(keys="image"),
        ]
    )

    return train_transforms



def get_supervoxel_transforms(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    supervoxel_transforms = Compose(
        [
            LoadImaged(keys=["supervoxel", "label"]),
            EnsureChannelFirstd(keys=["supervoxel", "label"]),
            EnsureTyped(keys=["supervoxel", "label"]),
            Orientationd(keys=["supervoxel", "label"], axcodes="RAS"),
            Spacingd(
                keys=["supervoxel", "label"], pixdim=spacing, mode=("nearest", "nearest")
            ),
            SpatialPadd(keys=["supervoxel", "label"], 
                        spatial_size=patch_size,
                        mode="constant"),
            RandSpatialCropd(
                keys=["supervoxel", "label"],
                roi_size=patch_size,
                random_size=False,
            ),
            GenerateVolumed(source_key="supervoxel", target_key="image"),
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
                keys=["image", "label"],
                prob=0.333,
                mode=['bilinear', 'nearest'],
                sigma_range=(5.0, 8.0),
                magnitude_range=(10, 50),
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.25, 0.25, 0.25),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=patch_size,
                padding_mode='zeros',
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            ScaleIntensityd(keys="image"),
        ]
    )

    return supervoxel_transforms



def get_val_transforms(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            ScaleIntensityd(keys="image"),
            SpatialPadd(keys=["image", "label"], 
                        spatial_size=patch_size,
                        mode="constant"),
        ]
    )

    return val_transforms



def get_test_transforms(patch_size=(128, 128, 128), spacing=(2.0, 2.0, 2.0)):

    test_transforms = Compose(
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
        ]
    )

    return test_transforms