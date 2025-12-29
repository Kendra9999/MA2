import random
from .synthetic_data import minmax

def simulate_noncontrast(image, label):
    image = minmax(image)
    
    sub_value = random.uniform(0.0, min(image[label==1].mean() + image[label==1].std(), 0.8))

    image[label==1] -= sub_value
    image[image<0] = 0

    return image


if __name__ == "__main__":
    import SimpleITK as sitk
    
    image_path = "../../../Data/CARE-Liver/LiQA_training_data/Vendor_B2/1075-B2-S4/GED4.nii.gz"
    label_path = "../../../Data/CARE-Liver/LiQA_training_data/Vendor_B2/1075-B2-S4/mask_GED4.nii.gz"
    output_path = "noncontrast.nii.gz"

    image_obj = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_obj)

    label_obj = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label_obj)

    volume = simulate_noncontrast(image, label)

    volume_obj = sitk.GetImageFromArray(volume)
    volume_obj.CopyInformation(image_obj)
    sitk.WriteImage(volume_obj, output_path)