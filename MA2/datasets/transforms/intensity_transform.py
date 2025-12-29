import random
import copy
from .synthetic_data import minmax

def inverse_intensity(image):
    image = minmax(image)
    new_image = copy.deepcopy(image)

    new_image = (1.0 - new_image) ** 3
    new_image[image<0.015] = 0

    return new_image

def intensity_transform(
        image,
        bin_num = 5,
        ):
    image = minmax(image)
    new_image = copy.deepcopy(image)

    target_bs = random.sample([b for b in range(bin_num)], k=bin_num)

    for idx, b in enumerate(range(bin_num)):
        ori_vmin = image.min() + b * (image.max() - image.min()) / bin_num
        ori_vmax = image.min() + (b + 1) * (image.max() - image.min()) / bin_num

        target_b = target_bs[idx]
        target_vmin = image.min() + target_b * (image.max() - image.min()) / bin_num
        target_vmax = image.min() + (target_b + 1) * (image.max() - image.min()) / bin_num

        new_image[(image >= ori_vmin) & (image <= ori_vmax)] = image[(image >= ori_vmin) & (image <= ori_vmax)] * \
                                    (target_vmax - target_vmin) / (ori_vmax - ori_vmin) + target_vmin - ori_vmin

    new_image[image<0.015] = 0
    
    return new_image


if __name__ == "__main__":
    import SimpleITK as sitk
    
    image_path = "../../../Data/CARE-Liver/LiQA_training_data/Vendor_B2/1075-B2-S4/GED4.nii.gz"
    output_path = "intensity_transform.nii.gz"

    image_obj = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_obj)

    volume = intensity_transform(image)

    volume_obj = sitk.GetImageFromArray(volume)
    volume_obj.CopyInformation(image_obj)
    sitk.WriteImage(volume_obj, output_path)


    output_path = "inverse_intensity.nii.gz"

    image_obj = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_obj)

    volume = inverse_intensity(image)

    volume_obj = sitk.GetImageFromArray(volume)
    volume_obj.CopyInformation(image_obj)
    sitk.WriteImage(volume_obj, output_path)