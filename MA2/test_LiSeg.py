import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

import torch
from torch import autocast
from torch.utils.data import DataLoader 

from monai.data import list_data_collate
from monai.inferers import sliding_window_inference

from datasets.dataset_test import LiverTestDataset
from utils.utils import dummy_context
from models.segcls_model import SegCLSModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--test_modalities', type=str, required=True, nargs='+', help='Test modalities')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    return args

def main(args):
    checkpoint_path = args.checkpoint_path

    work_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(work_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # dataset
    test_dataset = LiverTestDataset(test_data_dir=args.test_data_dir,
                                    test_modalities=args.test_modalities,
                                    transform_kwargs=config['transform_kwargs'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segcls_model = SegCLSModel(
        'scratch',
        config["n_classes"],
        config["transform_kwargs"]["patch_size"],
        config["num_modals"],
        config["n_cls_classes"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    segcls_model.load_state_dict(checkpoint)
    segcls_model.to(device)
    segcls_model.eval()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        test_loop = tqdm(test_loader, total=len(test_loader))
        test_loop.set_description("test")
        for test_data in test_loop:
            test_images = test_data["image"].to(device) 
            roi_size = config["transform_kwargs"]["patch_size"]
            sw_batch_size = 4
            with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                test_outputs = sliding_window_inference(
                    test_images, roi_size, sw_batch_size,
                    segcls_model, overlap=0.75,
                )

            test_data["pred"] = test_outputs.argmax(1).unsqueeze(1).cpu()

            save_folder = os.path.join(output_dir, 'LiSeg_pred', test_data["patient"][0])
            os.makedirs(save_folder, exist_ok=True)
            test_dataset.save_image(test_data, save_folder)

    print(f"Segmentation results saved to {output_dir}")

    print(f"Do max connected area:")
    for patient in os.listdir(os.path.join(output_dir, 'LiSeg_pred')):
        for file in os.listdir(os.path.join(output_dir, 'LiSeg_pred', patient)):
            itk_image = sitk.ReadImage(os.path.join(output_dir, 'LiSeg_pred', patient, file), sitk.sitkUInt8)
            itk_image = maxConnectArea(itk_image)
            sitk.WriteImage(itk_image, os.path.join(output_dir, 'LiSeg_pred', patient, file))
    print(f"Max connected area done!")



def maxConnectArea(itk_image_):
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_connected = cc_filter.Execute(itk_image_)
    
    output_connected_array = sitk.GetArrayFromImage(output_connected)
    num_connected_label = cc_filter.GetObjectCount()

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_connected)

    max_area = 0
    max_label_idx = 0
    for i in range(1,num_connected_label+1):
        cur_area = lss_filter.GetNumberOfPixels(i)
        if cur_area > max_area: 
            max_area = cur_area
            max_label_idx = i

    re_mask = np.zeros_like(output_connected_array, dtype='uint8')
    re_mask[output_connected_array==max_label_idx] = 1
    
    re_image = sitk.GetImageFromArray(re_mask)
    re_image.SetDirection(itk_image_.GetDirection())
    re_image.SetSpacing(itk_image_.GetSpacing())
    re_image.SetOrigin(itk_image_.GetOrigin())
    
    return re_image
    

if __name__ == "__main__":
    args = parse_args()

    main(args)