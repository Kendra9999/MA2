import os
import json
from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Invertd,
    AsDiscreted,
    SaveImaged,
)

from .transforms.transforms import get_val_transforms, get_test_transforms

class LiverValDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, transform_kwargs):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)
        
        self.target_modalities = ["DWI_800", "T2", "GED4"]
        
        self.val_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Val"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Val"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Val"]
        self.val_files = [(f, m) for f in self.val_files for m in self.target_modalities]
        self.val_files = [(f, m) for f, m in self.val_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        
        self.val_transforms = get_val_transforms(**transform_kwargs)

    def __len__(self):
        return len(self.val_files)

    def __getitem__(self, index):
        patient, modality = self.val_files[index]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.data_dir, vendor, patient, modality + ".nii.gz")
        label_path = os.path.join(self.process_data_dir, modality + "_aligned", vendor, patient, "mask_GED4.nii.gz")
        
        data = {"image": image_path, "label": label_path}
        data = self.val_transforms(data)

        return data


class LiverTestforSemiSupDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, transform_kwargs):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)
        
        self.same_patient_target_modalities = ["DWI_800", "T2", "T1"]
        
        self.same_patient_test_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Train"]
        self.same_patient_test_files = [(f, m) for f in self.same_patient_test_files for m in self.same_patient_target_modalities]
        self.same_patient_test_files = [(f, m) for f, m in self.same_patient_test_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.other_patient_target_modalities = ["DWI_800", "T2", "GED4"]
        
        self.other_patient_test_files = self.info_json["Patient_wo_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B2"]["Train"]
        self.other_patient_test_files = [(f, m) for f in self.other_patient_test_files for m in self.other_patient_target_modalities]
        self.other_patient_test_files = [(f, m) for f, m in self.other_patient_test_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.test_files = self.same_patient_test_files + self.other_patient_test_files

        self.test_transforms = get_test_transforms(**transform_kwargs)

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        patient, modality = self.test_files[index]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.data_dir, vendor, patient, modality + ".nii.gz")
        
        data = {"image": image_path, "vendor": vendor, "patient": patient}
        data = self.test_transforms(data)

        return data       

    def save_image(self, data, output_dir):

        post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=self.test_transforms,
                                       orig_keys="image",
                                       meta_keys="pred_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               AsDiscreted(keys="pred", argmax=False, to_onehot=None),
                               SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir,
                                          separate_folder=False, output_postfix="mask",
                                          resample=False),
                               ])
        
        [post_transforms(i) for i in decollate_batch(data)]  