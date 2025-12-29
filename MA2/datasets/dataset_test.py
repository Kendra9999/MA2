import os
import torch
from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    EnsureTyped,
    Invertd,
    AsDiscreted,
    SaveImaged,
)

from .transforms.transforms import get_test_transforms
from .transforms.transforms_cls import get_cls_val_transforms
from .dataset_cls import sliding_window

class LiverTestDataset(Dataset):
    def __init__(self, test_data_dir, test_modalities, transform_kwargs):
        self.test_data_dir = test_data_dir

        data_A_dir = os.path.join(test_data_dir, 'Vendor_A')
        data_B1_dir = os.path.join(test_data_dir, 'Vendor_B1')
        data_B2_dir = os.path.join(test_data_dir, 'Vendor_B2')
        data_C_dir = os.path.join(test_data_dir, 'Vendor_C')

        patients_A_dir = sorted(os.listdir(data_A_dir)) if os.path.exists(data_A_dir) else []
        patients_B1_dir = sorted(os.listdir(data_B1_dir)) if os.path.exists(data_B1_dir) else []
        patients_B2_dir = sorted(os.listdir(data_B2_dir)) if os.path.exists(data_B2_dir) else []
        patients_C_dir = sorted(os.listdir(data_C_dir)) if os.path.exists(data_C_dir) else []

        self.test_files = patients_A_dir + patients_B1_dir + patients_B2_dir + patients_C_dir

        
        self.target_modalities = test_modalities
    
        self.test_files = [(f, m) for f in self.test_files for m in self.target_modalities]
        self.test_files = [(f, m) for f, m in self.test_files \
                            if os.path.exists(os.path.join(self.test_data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.test_transforms = get_test_transforms(**transform_kwargs)

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        patient, modality = self.test_files[index]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.test_data_dir, vendor, patient, modality + ".nii.gz")
        
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
                                          separate_folder=False, output_postfix="pred",
                                          resample=False),
                               ])
        
        [post_transforms(i) for i in decollate_batch(data)] 


class LiverCLSTestDataset(Dataset):
    def __init__(self, test_data_dir, transform_kwargs, modalities):
        self.test_data_dir = test_data_dir

        data_A_dir = os.path.join(test_data_dir, 'Vendor_A')
        data_B1_dir = os.path.join(test_data_dir, 'Vendor_B1')
        data_B2_dir = os.path.join(test_data_dir, 'Vendor_B2')
        data_C_dir = os.path.join(test_data_dir, 'Vendor_C')

        patients_A_dir = sorted(os.listdir(data_A_dir)) if os.path.exists(data_A_dir) else []
        patients_B1_dir = sorted(os.listdir(data_B1_dir)) if os.path.exists(data_B1_dir) else []
        patients_B2_dir = sorted(os.listdir(data_B2_dir)) if os.path.exists(data_B2_dir) else []
        patients_C_dir = sorted(os.listdir(data_C_dir)) if os.path.exists(data_C_dir) else []

        self.test_files = patients_A_dir + patients_B1_dir + patients_B2_dir + patients_C_dir

        self.modalities = modalities

        self.test_files = [f for f in self.test_files \
                            if any([os.path.exists(os.path.join(self.test_data_dir, 
                                    "Vendor_" + f.split("-")[1], f, m + ".nii.gz")) for m in self.modalities.values()])]
        
        self.test_transforms = get_cls_val_transforms(**transform_kwargs)
        self.patch_size = transform_kwargs["patch_size"]
        
    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):

        patient = self.test_files[index]
        vendor = "Vendor_" + patient.split("-")[1] 
        patient_dir = os.path.join(self.test_data_dir, vendor, patient)
        
        images = []
        modals = []
        for idx, modality in self.modalities.items():
            image_path = os.path.join(patient_dir, modality + ".nii.gz")
            if os.path.exists(image_path):
                image = self.test_transforms({"image": image_path})["image"]
                images.append(sliding_window(image, self.patch_size))
                modals.extend([idx] * images[-1].shape[0])
        images = torch.cat(images, dim=0)
        modals = torch.tensor(modals)

        return {"image": images, "modal": modals, "patient": patient}