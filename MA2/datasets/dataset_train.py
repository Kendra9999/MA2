import os
import json
import random
import numpy as np
from torch.utils.data import Dataset

from .transforms.transforms import get_train_transforms, get_supervoxel_transforms, get_train_transforms_nointensity

class LiverTrainDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, transform_kwargs, multisets=False, set_length=100):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)
        
        self.train_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Train"]
        
        self.train_transforms = get_train_transforms(**transform_kwargs)
        self.supervoxel_transforms = get_supervoxel_transforms(**transform_kwargs)
        
        self.multisets = multisets
        if self.multisets:
            self.set_length = set_length

    def __len__(self):
        if self.multisets:
            return self.set_length
        else:
            return len(self.train_files)

    def __getitem__(self, index):
        if self.multisets:
            loc = np.random.randint(0, len(self.train_files))
        else:
            loc = index

        patient = self.train_files[loc]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.data_dir, vendor, patient, "GED4.nii.gz")
        label_path = os.path.join(self.data_dir, vendor, patient, "mask_GED4.nii.gz")
        supervoxel_path = os.path.join(self.process_data_dir, "GED4_supervoxel", vendor, patient, "GED4.nii.gz")

        if random.random() < 0.65:
            data = {"supervoxel": supervoxel_path, "label": label_path}
            data = self.supervoxel_transforms(data)
        else:
            data = {"image": image_path, "label": label_path}
            data = self.train_transforms(data)

        return data


class OtherTrainDataset(Dataset):
    def __init__(self, other_data_dirs, transform_kwargs, multisets=False, set_length=100):
        self.data_dirs = other_data_dirs
        
        self.train_files = []
        for data_dir in self.data_dirs:            
            self.train_files += [(f, data_dir) for f in sorted(os.listdir(os.path.join(data_dir, "imagesTr")))]
        
        self.train_transforms = get_train_transforms(**transform_kwargs)
        self.supervoxel_transforms = get_supervoxel_transforms(**transform_kwargs)

        self.multisets = multisets
        if self.multisets:
            self.set_length = set_length

    def __len__(self):
        if self.multisets:
            return self.set_length
        else:
            return len(self.train_files)

    def __getitem__(self, index):
        if self.multisets:
            loc = np.random.randint(0, len(self.train_files))
        else:
            loc = index

        patient, data_dir = self.train_files[loc]
        
        image_path = os.path.join(data_dir, "imagesTr", patient)
        label_path = os.path.join(data_dir, "labelsTr", patient.replace('_0000.nii.gz', '.nii.gz'))
        supervoxel_path = os.path.join(data_dir, "supervoxelTr", patient)

        if random.random() < 0.65:
            data = {"supervoxel": supervoxel_path, "label": label_path}
            data = self.supervoxel_transforms(data)
        else:
            data = {"image": image_path, "label": label_path}
            data = self.train_transforms(data)

        return data
    


class LiverSemiSupDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, predict_data_dir,
                  transform_kwargs, multisets=False, set_length=100):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        self.predict_data_dir = predict_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)
        
        self.target_modalities = ["DWI_800", "T2", "T1", "GED4"]
        
        self.same_patient_train_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Train"]
        self.same_patient_train_files = [(f, m) for f in self.same_patient_train_files for m in self.target_modalities]
        self.same_patient_train_files = [(f, m) for f, m in self.same_patient_train_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.other_patient_train_files = self.info_json["Patient_wo_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B2"]["Train"]
        self.other_patient_train_files = [(f, m) for f in self.other_patient_train_files for m in self.target_modalities]
        self.other_patient_train_files = [(f, m) for f, m in self.other_patient_train_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.train_files = [(f, m, "same") for f, m in self.same_patient_train_files] + \
                            [(f, m, "other") for f, m in self.other_patient_train_files]
        
        self.train_transforms = get_train_transforms_nointensity(**transform_kwargs)
        
        self.multisets = multisets
        if self.multisets:
            self.set_length = set_length

    def __len__(self):
        if self.multisets:
            return self.set_length
        else:
            return len(self.train_files)

    def __getitem__(self, index):
        if self.multisets:
            loc = np.random.randint(0, len(self.train_files))
        else:
            loc = index

        patient, modality, tag = self.train_files[loc]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.data_dir, vendor, patient, modality + ".nii.gz")
        if modality == "GED4" and tag == "same":
            label_path = os.path.join(self.data_dir, vendor, patient, "mask_GED4.nii.gz")
        else:
            label_path = os.path.join(self.predict_data_dir, vendor, patient, f"{modality}_mask.nii.gz")
        
        data = {"image": image_path, "label": label_path}
        data = self.train_transforms(data)

        return data