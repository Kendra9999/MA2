"""
Modified from https://github.com/sha168/ADNet
"""

import os
import SimpleITK as sitk
import glob
from skimage.measure import label
import scipy.ndimage.morphology as snm
from felzenszwalb_3d import *

data_dir = "../Data_preprocess/CHAOS_MRI/"
save_dir = os.path.join(data_dir, "supervoxelTr")
os.makedirs(save_dir, exist_ok=True)

MODE = 'MIDDLE' # minimum size of pesudolabels. 'MIDDLE' is the default setting
n_sv = 1000
fg_thresh = 5


def read_nii_bysitk(input_fid):
    """ read nii to numpy through simpleitk
        peelinfo: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    return img_np, img_obj


# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh):
    mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask


# remove supervoxels within the empty regions
def supervox_masking(seg, mask):
    seg[seg == 0] = seg.max() + 1
    seg = np.int32(seg)
    seg[mask == 0] = 0

    return seg


imagesTr_dir = os.path.join(data_dir, "imagesTr")

patients = sorted(os.listdir(imagesTr_dir))

for patient in patients:
    
    img_path = os.path.join(imagesTr_dir, patient)
    
    # make supervoxels
    img, img_obj = read_nii_bysitk(img_path)
    img = img.astype(np.float32)
    img = 255 * (img - img.min()) / img.ptp()

    reader = sitk.ImageFileReader()
    reader.SetFileName(img_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    x = float(reader.GetMetaData('pixdim[1]'))
    y = float(reader.GetMetaData('pixdim[2]'))
    z = float(reader.GetMetaData('pixdim[3]'))

    segments_felzenszwalb = felzenszwalb_3d(img, min_size=n_sv, sigma=0, spacing=(z, x, y))

    # post processing: remove bg (low intensity regions)
    fg_mask_vol = np.zeros(segments_felzenszwalb.shape)
    for ii in range(segments_felzenszwalb.shape[0]):
        _fgm = fg_mask2d(img[ii, ...], fg_thresh)
        fg_mask_vol[ii] = _fgm
    processed_seg_vol = supervox_masking(segments_felzenszwalb, fg_mask_vol)

    # write to nii.gz
    out_seg = sitk.GetImageFromArray(processed_seg_vol)
    out_seg = sitk.Cast(out_seg, sitk.sitkUInt8)
    out_seg.CopyInformation(img_obj)

    sitk.WriteImage(out_seg, os.path.join(save_dir, patient))

    print(f'Image {os.path.join(save_dir, patient)} has finished')
