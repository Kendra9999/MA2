import os
import copy
import json
import numpy as np
from torch.utils.data import Dataset

from .transforms.transforms_mae import get_mae_train_transforms


def random_mask_image(image, mask_size = (16, 16, 16), mask_ratio_range = (0.5, 0.8)):
    mask_image = copy.deepcopy(image)

    data_shape = image.shape[1:]
    num_total = (data_shape[0] // mask_size[0]) * (data_shape[1] // mask_size[1]) * (data_shape[2] // mask_size[2])
    total_ids = np.arange(num_total)
    np.random.shuffle(total_ids)

    mask_ratio = np.random.uniform(mask_ratio_range[0], mask_ratio_range[1])
    num_mask = int(num_total * mask_ratio)
    mask_ids = total_ids[:num_mask]

    start_x = (mask_ids // (data_shape[2] // mask_size[2]) // (data_shape[1] // mask_size[1])) * mask_size[0]
    start_y = (mask_ids // (data_shape[2] // mask_size[2]) % (data_shape[1] // mask_size[1])) * mask_size[1]
    start_z = (mask_ids % (data_shape[2] // mask_size[2])) * mask_size[2]
    
    for m in range(num_mask):
        mask_image[:, start_x[m]:start_x[m]+mask_size[0],
                      start_y[m]:start_y[m]+mask_size[1],
                      start_z[m]:start_z[m]+mask_size[2]] = 0

    return mask_image


class LiverMAEDataset(Dataset):
    def __init__(self, data_dir, process_data_dir,
                  transform_kwargs, multisets=False, set_length=100):
        self.data_dir = data_dir
        self.process_data_dir = process_data_dir
        
        with open(os.path.join(process_data_dir, 'info.json'), 'r') as f:
            self.info_json = json.load(f)
        
        self.target_modalities = ["DWI_800", "GED1", "GED2", "GED3", "GED4", "T1", "T2"]
        
        self.train_files = self.info_json["Patient_w_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_w_mask"]["Vendor_B2"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_A"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B1"]["Train"] + \
                            self.info_json["Patient_wo_mask"]["Vendor_B2"]["Train"]
        
        self.train_files = [(f, m) for f in self.train_files for m in self.target_modalities]
        self.train_files = [(f, m) for f, m in self.train_files \
                            if os.path.exists(os.path.join(self.data_dir, "Vendor_" + f.split("-")[1], f, m + ".nii.gz"))]
        
        self.train_transforms = get_mae_train_transforms(**transform_kwargs)
        
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

        patient, modality = self.train_files[loc]
        vendor = "Vendor_" + patient.split("-")[1] 
        image_path = os.path.join(self.data_dir, vendor, patient, modality + ".nii.gz")
        
        data = {"image": image_path}
        data = self.train_transforms(data)

        data["mask_image"] = random_mask_image(data["image"])

        return data