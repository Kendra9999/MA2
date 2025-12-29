"""
Modified from Dey et al.
https://github.com/neel-dey/anatomix/
"""

import numpy as np
import torch

from monai.transforms import (
    ScaleIntensityd,
    Compose,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandSimulateLowResolutiond,
    ThresholdIntensityd
)

# -----------------------------------------------------------------------------
# Helpers for volume generation:
# -----------------------------------------------------------------------------


def get_transforms():
    """
    Generates a MONAI composed transformation set for augmenting the GMM
    sampled intensities. We sample one view/volume per each 3D labelmap ("view" below).
    These are then augmented as below.

    See the comments below to walk through the transforms.

    Returns
    -------
    train_transforms : monai.transforms.Compose
        A MONAI Compose object containing the specified sequence of transforms.

    Notes
    -----
    The specific probabilities and parameter ranges for each transformation are
    based on the empirical settings used in the paper. Play around with it!
    """
    
    train_transforms = Compose(
        [
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view"]),
            # Apply bias fields:
            RandBiasFieldd(
                keys=["view"], prob=0.98, coeff_range=(0.0, 0.075),
            ),
            # Apply K-spikes:
            RandKSpaceSpikeNoised(keys=["view"], prob=0.2),
            # Apply gamma transforms:
            RandAdjustContrastd(keys=["view"], prob=0.5, gamma=(0.5, 2.)),
            # Apply smoothing:
            RandGaussianSmoothd(
                keys=["view"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            RandGaussianSmoothd(
                keys=["view"],
                prob=0.98,
                sigma_x=(1.5, 2.5),
                sigma_y=(1.5, 2.5),
                sigma_z=(1.5, 2.5),
            ),
            # Apply gibbs ringing (applies a box mask to kspace. alpha=0, box
            # width=1, i.e. no masking. alpha=1, boxwidth=0, i.e. all masked):
            RandGibbsNoised(keys=["view"], prob=0.5, alpha=(0.0, 0.333)),
            # Apply sharpening:
            RandGaussianSharpend(keys=["view"], prob=0.25),
            # Simulate much bigger voxels. MONAI does it nnUNet style, as
            # opposed to TorchIO's (IMO better) per-axis anisotropic style:
            RandSimulateLowResolutiond(keys=["view"], prob=0.5, zoom_range=(0.25, 1.0)),
            # Clip out negative values:
            ThresholdIntensityd(
                keys=["view"], above=True, threshold=0.,
            ),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view"]),
        ]
    )
    return train_transforms


def draw_perlin_volume(
    out_shape,
    scales,
    min_std=0,
    max_std=1,
    dtype=torch.float32,
    device="cpu",
):
    """
    #TODO: merge draw_perlin_volume and draw_perlin_deformation
    
    Generates a 3D tensor with Perlin noise as defined in
    https://arxiv.org/abs/2004.10282

    Parameters
    ----------
    out_shape : tuple of int
        Shape of the output tensor (e.g., (D, H, W)).
    scales : float or list of float
        List of scales at which to generate the noise. 
        A single float can also be provided.
    min_std : float, optional
        Minimum standard deviation of the Gaussian noise. Default is 0.
    max_std : float, optional
        Maximum standard deviation of the Gaussian noise. Default is 1.
    dtype : torch.dtype, optional
        Data type of the output tensor. Default is torch.float32.
    device : str or torch.device, optional
        Device on which to create the tensor. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape `out_shape` with generated Perlin noise.
    """
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    out = torch.zeros(tuple(out_shape), dtype=dtype, device=device)

    for scale in scales:
        sample_shape = np.ceil(out_shape / scale).astype(np.uint8)
    
        std = (max_std - min_std) * torch.rand(
            (1,), dtype=torch.float32, device=device
        )
        std = std + min_std
        gauss = std * torch.randn(
            tuple(sample_shape), dtype=torch.float32, device=device
        )
    
        zoom = [o // s for o, s in zip(out_shape, sample_shape)]
        if scale == 1:
            out += gauss
        else:
            out += torch.nn.functional.interpolate(
                gauss[None, None, ...],
                size=out.size(),
                # scale_factor=scale,
                mode='trilinear'
            )[0, 0, ...]

    return out


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def sample_gmm(means, stds, label_map, zero_bckgnd=0.8, device="cuda"):
    """
    Generate a synthetic image using a Gaussian Mixture Model (GMM).
    
    This function creates a synthetic image where each region corresponding 
    to a unique label in the 3D synthetic `label_map` is filled with values
    from a Gaussian distribution characterized by the specified means and 
    standard deviations (`stds`). 
    
    100*zero_bckgnd % of the time, fill background label with zeros.

    Parameters
    ----------
    means : list or np.ndarray
        A list or array of means for the Gaussians, one for each label.
    stds : list or np.ndarray
        A list or array of std devs for the Gaussians, one for each label.
    label_map : np.ndarray
        A 3D array where each element corresponds to a label indicating the
        region in the synthetic image.
    zero_bckgnd : float
        Probability of filling background with zeros instead of intensities.

    Returns
    -------
    torch.Tensor
        A synthetic image/torch Tensor with values generated from the Gaussian 
        distributions, with values clipped to a minimum of 0 and scaled using 
        min-max normalization.
        
    """
    labels = np.unique(label_map)
    synthimage = torch.zeros(label_map.shape, requires_grad=False, device=device)

    for i, label in enumerate(labels):
        if (i == 0) and (torch.rand(1) < zero_bckgnd):
            continue
        indices = label_map==label
        synthimage[indices] = stds[i] * torch.randn(indices.sum(), device=device) + means[i]

    synthimage = torch.clip(synthimage, min=0)
    synthimage = minmax(synthimage)

    return synthimage


def transform_uniform(arr, minval, maxval):
    """
    Transform arr from a uniform distribution in [0, 1] to [minval, maxval].
    """
    assert arr.min() >= 0
    assert arr.max() <= 1
    return (maxval - minval) * arr + minval


# -----------------------------------------------------------------------------
# Volume generation:
# -----------------------------------------------------------------------------


def generate_volume(
        label_map, 
        means_range=(25, 225, 255),
        stds_range=(5, 20),
        perl_scales=(4, 8, 16, 32),
        perl_max_std=5.,
        perl_mult_factor=0.02,
        ):
    random_means_low = np.random.randint(means_range[0], means_range[1])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Initialize MONAI augmentation pipeline:
    transforms = get_transforms()

    labels = np.unique(label_map)

    # Sample random means and std devs for both paired volumes:
    means = transform_uniform(
        torch.rand(len(labels)), random_means_low, means_range[2],
    )
    stds = transform_uniform(
        torch.rand(len(labels)), stds_range[0], stds_range[1],
    )

    # Sample volume from the specified GMMs:
    synthview = sample_gmm(means, stds, label_map, device=device)

    # Sample Perlin-like noise to simulate spatial structure in texture:
    randperl_view = draw_perlin_volume(
        out_shape=label_map.shape,
        scales=perl_scales,
        max_std=perl_max_std,
        device=device,
    )
    # Pointwise multiply with Perlin-like noise and downscale intensities 
    # by `perl_mult_factor`:
    synthperl = synthview * (1 + perl_mult_factor * randperl_view)

    # Create data dict and send to MONAI augmentation pipeline:
    inputimgs = {"view": synthperl, "label": label_map}
    outputs = transforms(inputimgs)

    # torch.cuda.empty_cache()

    return outputs["view"].detach()


if __name__ == "__main__":
    import SimpleITK as sitk
    
    supervoxel_path = "../../../Data_preprocess/CARE-Liver/LiQA_training_data/GED4_supervoxel/Vendor_B2/1075-B2-S4/GED4.nii.gz"
    output_path = "volume.nii.gz"

    supervoxel_obj = sitk.ReadImage(supervoxel_path)
    supervoxel = sitk.GetArrayFromImage(supervoxel_obj)

    volume = generate_volume(supervoxel)

    volume_obj = sitk.GetImageFromArray(volume.cpu().numpy())
    volume_obj.CopyInformation(supervoxel_obj)
    sitk.WriteImage(volume_obj, output_path)
