import os
import yaml
import argparse
from tqdm import tqdm
import csv

import torch
from torch import autocast
from torch.utils.data import DataLoader 

from monai.data import list_data_collate

from datasets.dataset_test import LiverCLSTestDataset
from utils.utils import dummy_context
from models.segcls_model import SegCLSModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--setting', type=str, required=True, help='Contrast or NonContrast')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    return args

def main(args):
    checkpoint_path = args.checkpoint_path

    work_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(work_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # dataset
    if args.setting == "Contrast":
        modalities = {0: "DWI_800", 1: "GED1", 2: "GED2", 3: "GED3", 4: "GED4", 5: "T1", 6: "T2"}
    elif args.setting == "NonContrast":
        modalities = {0: "DWI_800", 5: "T1", 6: "T2"}
    else: 
        raise ValueError(f"Setting {args.setting} not supported")
    
    test_dataset = LiverCLSTestDataset(test_data_dir=args.test_data_dir,
                                    modalities=modalities,
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

    test_results = []
    if not os.path.exists(os.path.join(output_dir, "LiFS_pred.csv")):
        test_results.append(["Case", "Setting", "Subtask1_prob_S4", "Subtask2_prob_S1"])

    with torch.no_grad():
        test_loop = tqdm(test_loader, total=len(test_loader))
        test_loop.set_description("test")
        for test_data in test_loop:
            test_images = test_data["image"].squeeze(0).unsqueeze(1).to(device) 
            test_modals = test_data["modal"].to(device)

            with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                test_outputs = segcls_model(test_images, mode='cls', modals=test_modals)
                test_probs = [test_output.softmax(dim=1) for test_output in test_outputs]
            
                test_result = [test_data["patient"][0], args.setting,
                               str((test_probs[0][0, 3].item() + test_probs[2][0, 1].item()) / 2), 
                               str((test_probs[0][0, 0].item() + test_probs[1][0, 0].item()) / 2)]
                test_results.append(test_result)

    with open(os.path.join(output_dir, "LiFS_pred.csv"), "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(test_results)

    print(f"LiFS results saved to {output_dir}")    


if __name__ == "__main__":
    args = parse_args()

    main(args)