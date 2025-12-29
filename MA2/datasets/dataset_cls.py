import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms.transforms_cls import get_cls_train_transforms, get_cls_val_transforms


def sliding_window(tensor, kernel_size, overlap=0.5):
    stride = [int(kernel_size[i] * overlap) for i in range(3)]
    C, D, H, W = tensor.size()
    patches = []
    for d in range(0, D - kernel_size[0] + 1, stride[0]):
        for h in range(0, H - kernel_size[1] + 1, stride[1]):
            for w in range(0, W - kernel_size[2] + 1, stride[2]):
                patch = tensor[:, d:d+kernel_size[0], h:h+kernel_size[1], w:w+kernel_size[2]]
                patches.append(patch)
    return torch.cat(patches, dim=0)  


class LiverCLSTrainDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, cls_modal_dropout, transform_kwargs,
                 modalities, multisets=False, set_length=100):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        self.cls_modal_dropout = cls_modal_dropout
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)

        self.modalities = modalities
        
        self.train_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B2"]["Train"]
        
        self.train_transforms = get_cls_train_transforms(**transform_kwargs)
        self.patch_size = transform_kwargs["patch_size"]
        
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
        patient_dir = os.path.join(self.data_dir, vendor, patient)
        label = int(patient.split("-")[2][-1]) - 1

        images = []
        modals = []
        
        exists_modalities = [f for f in self.modalities.values() if os.path.exists(os.path.join(patient_dir, f + ".nii.gz"))]
        want_modality = random.choice(exists_modalities)

        for idx, modality in self.modalities.items():
            image_path = os.path.join(patient_dir, modality + ".nii.gz")
            if os.path.exists(image_path):
                if modality == want_modality or random.random() > self.cls_modal_dropout:
                    image = self.train_transforms({"image": image_path})["image"]
                    images.append(sliding_window(image, self.patch_size))
                    modals.extend([idx] * images[-1].shape[0])
        images = torch.cat(images, dim=0)
        modals = torch.tensor(modals)

        if images.shape[0] > 12:
            indices = torch.randperm(images.shape[0])[:12]
            images = images[indices]
            modals = modals[indices]

        return {"image": images, "modal": modals, "label": label}



class LiverCLSValDataset(Dataset):
    def __init__(self, data_dir, process_data_dir, transform_kwargs, modalities):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)

        self.modalities = modalities
        
        self.val_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Val"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Val"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Val"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_A"]["Val"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B1"]["Val"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B2"]["Val"]
        
        self.val_transforms = get_cls_val_transforms(**transform_kwargs)
        self.patch_size = transform_kwargs["patch_size"]
        
    def __len__(self):
        return len(self.val_files)

    def __getitem__(self, index):

        patient = self.val_files[index]
        vendor = "Vendor_" + patient.split("-")[1] 
        patient_dir = os.path.join(self.data_dir, vendor, patient)
        label = int(patient.split("-")[2][-1]) - 1

        images = []
        modals = []
        for idx, modality in self.modalities.items():
            image_path = os.path.join(patient_dir, modality + ".nii.gz")
            if os.path.exists(image_path):
                image = self.val_transforms({"image": image_path})["image"]
                images.append(sliding_window(image, self.patch_size))
                modals.extend([idx] * images[-1].shape[0])
        images = torch.cat(images, dim=0)
        modals = torch.tensor(modals)

        return {"image": images, "modal": modals, "label": label}
    