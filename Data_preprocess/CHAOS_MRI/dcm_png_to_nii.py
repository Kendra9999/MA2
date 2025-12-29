"""
Modified from Ouyang et al.
https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
"""

import os
import glob
import json
import numpy as np
from PIL import Image
import SimpleITK as sitk

target_labels = ["liver"]
original_labels = {0: "background", 1: "liver", 2: "kidney_right", 3: "kidney_left", 4: "spleen"}


def convert_dcm_to_nii(dcmdirectory, savepath):
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(dcmdirectory)
    reader.SetFileNames(img_names)
    dcm = reader.Execute()         
    sitk.WriteImage(dcm, savepath)

data_dir = 'Data/CHAOS_MRI/Train_Sets/MR/'
save_dir = 'Data_preprocess/CHAOS_MRI/'
image_save_dir = os.path.join(save_dir, 'imagesTr')
label_save_dir = os.path.join(save_dir, 'labelsTr')
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(label_save_dir, exist_ok=True)

modality = ["T1DUAL_InPhase", "T1DUAL_OutPhase", "T2SPIR"]


label_mapping = {}
invert_label_dict = {v: k for k, v in original_labels.items()}
target_labels_dict = {0: "background"}
for idx, label in enumerate(target_labels):
    label_mapping[int(invert_label_dict[label])] = idx + 1
    target_labels_dict[idx + 1] = label
print (label_mapping)


patients_dir = sorted(os.listdir(data_dir))
for patient_dir in patients_dir:
    patient_path = os.path.join(data_dir, patient_dir)
    for mod in modality:
        # convert dcm to nii
        if mod == "T2SPIR":
            dcm_path = os.path.join(patient_path, mod, "DICOM_anon")
        else:
            dcm_path = os.path.join(patient_path, mod.split("_")[0], "DICOM_anon", mod.split("_")[1])
            
        image_save_path = os.path.join(image_save_dir, patient_dir + "_" + mod + "_0000.nii.gz")
        
        print ('Convert {} to {}'.format(dcm_path, image_save_path))
        convert_dcm_to_nii(dcm_path, image_save_path)

        # convert png to nii
        pngs = glob.glob(os.path.join(patient_path, mod.split("_")[0], "Ground/*.png"))
        pngs = sorted(pngs, key = lambda x: int(os.path.basename(x).split("-")[-1].split(".png")[0]))
        buffer = []

        for fid in pngs:
            buffer.append(Image.open(fid))

        vol = np.stack(buffer, axis = 0)
        # # flip correction
        # vol = np.flip(vol, axis = 1).copy()
        # remap values
        for new_val, old_val in enumerate(sorted(np.unique(vol))):
            vol[vol == old_val] = new_val

        # convert label map
        new_label_data = np.zeros_like(vol)
        for key, value in label_mapping.items():
            new_label_data[vol == key] = value

        # get reference    
        img_o = sitk.ReadImage(image_save_path)
        vol_o = sitk.GetImageFromArray(new_label_data)
        vol_o.SetSpacing(img_o.GetSpacing())
        vol_o.SetOrigin(img_o.GetOrigin())
        vol_o.SetDirection(img_o.GetDirection())

        label_save_path = os.path.join(label_save_dir, patient_dir + "_" + mod + ".nii.gz")
        sitk.WriteImage(vol_o, label_save_path)
        print(f'{label_save_path} has been saved!')