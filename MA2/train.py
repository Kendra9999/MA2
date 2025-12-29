import os
import time
import yaml
import argparse
import wandb
from tqdm import tqdm
import glob

import torch
import torch.nn.functional as F
from torch import autocast
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler

import monai
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference

from datasets.dataset_train import LiverTrainDataset, OtherTrainDataset, LiverSemiSupDataset
from datasets.dataset_cls import LiverCLSTrainDataset, LiverCLSValDataset
from datasets.dataset_val import LiverValDataset, LiverTestforSemiSupDataset
from datasets.dataset_mae import LiverMAEDataset
from utils.utils import dummy_context, get_logger, setup_seed, cosine_scheduler
from utils.validation import validate_cls
from models.segcls_model import SegCLSModel
from models.losses.cls_loss import CLS_loss

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--entity', type=str, required=True, help='WANDB entity')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_seed(config["seed"])

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    
    if args.auto_resume:
        pre_work_dir = 'work_dirs/works/{}/'.format(config["exp_name"])
        if os.path.exists(pre_work_dir):
            pre_work_dirs = sorted([os.path.join(pre_work_dir, d) for d in os.listdir(pre_work_dir)])
            if len(pre_work_dirs) > 0:
                work_dir = pre_work_dirs[-1]
                print('Auto resume from previous work dir: {}'.format(work_dir))
    else:
        work_dir = os.path.join('work_dirs/works/{}/'.format(config["exp_name"]), time_str)
        os.makedirs(work_dir, exist_ok=True)
    
    with open(os.path.join(work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    logger = get_logger(os.path.join(work_dir, 'run.log'))
    logger.info('work_dir: {}'.format(work_dir))
    logger.info('config: {}'.format(config))


    # dataset
    train_dataset = LiverTrainDataset(config["data_dir"], config["process_data_dir"],
                                      config["transform_kwargs"], config["multisets"][0], config["set_length"][0])
    if config["other_data_dirs"]:
        other_train_dataset = OtherTrainDataset(config["other_data_dirs"], config["transform_kwargs"], 
                                                config["multisets"][1], config["set_length"][1])
        train_dataset = ConcatDataset([train_dataset, other_train_dataset])
    
    val_dataset = LiverValDataset(config["data_dir"], config["process_data_dir"],
                                  config["transform_kwargs"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        # pin_memory=True,
        collate_fn=list_data_collate,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True,
        collate_fn=list_data_collate,
    )

    # dataset for cls
    cls_train_dataset = LiverCLSTrainDataset(config["data_dir"], config["process_data_dir"], 
                                             config["cls_modal_dropout"], config["transform_kwargs"],
                                             modalities={0: "DWI_800", 1: "GED1", 2: "GED2", 3: "GED3", 4: "GED4", 5: "T1", 6: "T2"})
    cls_train_loader = DataLoader(
        cls_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config["num_workers"],
        # pin_memory=True,
        collate_fn=list_data_collate,
    )
    cls_train_loader_iter = iter(cls_train_loader)

    # dataset for cls validation
    cls_val_dataset = LiverCLSValDataset(config["data_dir"], config["process_data_dir"], 
                                         config["transform_kwargs"],
                                         modalities={0: "DWI_800", 1: "GED1", 2: "GED2", 3: "GED3", 4: "GED4", 5: "T1", 6: "T2"})
    cls_val_loader = DataLoader(
        cls_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True,
        collate_fn=list_data_collate,
    )
    cls_val_noncontrast_dataset = LiverCLSValDataset(config["data_dir"], config["process_data_dir"], 
                                         config["transform_kwargs"],
                                         modalities={0: "DWI_800", 5: "T1", 6: "T2"})
    cls_val_noncontrast_loader = DataLoader(
        cls_val_noncontrast_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True,
        collate_fn=list_data_collate,
    )

    # dataset for MAE
    mae_train_dataset = LiverMAEDataset(config["data_dir"], config["process_data_dir"],
                                        config["transform_kwargs"])
    mae_train_loader = DataLoader(
        mae_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        # pin_memory=True,
        collate_fn=list_data_collate,
    )
    mae_train_loader_iter = iter(mae_train_loader)

    # Create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segcls_model = SegCLSModel(
        config["pretrained_ckpt"],
        config["n_classes"],
        config["transform_kwargs"]["patch_size"],
        config["num_modals"],
        config["n_cls_classes"],
    ).to(device)

    # Create Dice + CE loss function
    loss_function = monai.losses.DiceCELoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )
    # Track Dice loss for validation
    valloss_function = monai.losses.DiceLoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )

    # Loss for cls
    cls_loss_function = CLS_loss(
        config["n_cls_classes"], config["cls_main_loss_weights"], 
        config["cls_aux1_loss_weights"], config["cls_aux2_loss_weights"],
        config["cls_aux_loss_weights"],
    ).to(device)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        segcls_model.parameters(), float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"]
    )
    optimizer_cls = torch.optim.AdamW(
        segcls_model.parameters(), float(config["cls_lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_cls, T_max=config["n_epochs"]
    )
    consistency_weight_scheduler = cosine_scheduler(
        config["consistency_weight"], 1, config["n_epochs"], len(train_loader),
    )

    grad_scaler = GradScaler() if device.type == 'cuda' else None

    # start a typical PyTorch training
    val_interval = config["val_interval"]
    best_val_loss = 10000000000
    best_cls_score = 0
    start_epoch = 0

    semisup = False

    if args.auto_resume:
        old_ckpts = sorted(glob.glob(os.path.join(work_dir, "epoch*.pth")))
        if old_ckpts:
            old_ckpt = old_ckpts[-1]
            logger.info('Auto resume from previous epoch checkpoint: {}'.format(old_ckpt))
            checkpoint = torch.load(old_ckpt)
            segcls_model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.step(checkpoint["scheduler"]["last_epoch"])
            optimizer_cls.load_state_dict(checkpoint["optimizer_cls"])
            scheduler_cls.load_state_dict(checkpoint["scheduler_cls"])
            scheduler_cls.step(checkpoint["scheduler_cls"]["last_epoch"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = torch.Tensor([checkpoint["best_val_loss"]]).to(device)
            best_cls_score = checkpoint["best_cls_score"]
            best_loss_epoch = int(sorted(glob.glob(os.path.join(work_dir, "best_dict_epoch*.pth")))[-1].split("/")[-1][15:19])
            print ("best_loss_epoch", best_loss_epoch)
            semisup = checkpoint["semisup"]
            if semisup:
                
                semisup_test_dataset = LiverTestforSemiSupDataset(
                    config["data_dir"],
                    config["process_data_dir"],
                    config["transform_kwargs"],
                )
                semisup_test_loader = DataLoader(
                    semisup_test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    # pin_memory=True,
                    collate_fn=list_data_collate,
                )

                # semi supervised train dataloader
                semisup_predict_dir = os.path.join(work_dir, "semisup_predict")
                
                semisup_train_dataset = LiverSemiSupDataset(
                    config["data_dir"],
                    config["process_data_dir"],
                    semisup_predict_dir,
                    config["transform_kwargs"],
                )
                semisup_train_loader = DataLoader(
                    semisup_train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=True,
                    num_workers=config["num_workers"],
                    # pin_memory=True,
                    collate_fn=list_data_collate,
                )
                semisup_train_loader_iter = iter(semisup_train_loader)

    wandb.init(project = "CARE_Liver",
               entity = args.entity,
               dir = work_dir,
               name = config["exp_name"])
    
    # Training loop
    for epoch in range(start_epoch, config["n_epochs"]):
        logger.info("-" * 10)
        logger.info("epoch {:04d}/{:04d}".format(epoch + 1, config["n_epochs"]))

        segcls_model.train()

        epoch_loss = 0
        step = 0

        train_loop = tqdm(train_loader, total=len(train_loader))
        train_loop.set_description("epoch {:04d}/{:04d}".format(epoch + 1, config["n_epochs"]))
        for batch_data in train_loop:
            step += 1
            inputs = batch_data["image"].to(device) 
            labels = batch_data["label"].to(device)

            if semisup:
                try:
                    semisup_data = next(semisup_train_loader_iter)
                except StopIteration:
                    semisup_train_loader_iter = iter(semisup_train_loader)
                    semisup_data = next(semisup_train_loader_iter)
                
                semisup_inputs = semisup_data["image"].to(device) 
                semisup_labels = semisup_data["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                outputs = segcls_model(inputs)
                loss = loss_function(outputs, labels)

                if semisup:
                    semisup_outputs = segcls_model(semisup_inputs)
                    loss += consistency_weight_scheduler[step-1+epoch*len(train_loader)] * \
                          loss_function(semisup_outputs, semisup_labels)

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                optimizer.step()

            # torch.cuda.empty_cache()
            
            # train classification
            cls_loss_all = 0
            for b in range(config["cls_batch_size"]):
                try:
                    cls_data = next(cls_train_loader_iter)
                except StopIteration:
                    cls_train_loader_iter = iter(cls_train_loader)
                    cls_data = next(cls_train_loader_iter)
                
                cls_inputs = cls_data["image"].squeeze(0).unsqueeze(1).to(device) 
                cls_modals = cls_data["modal"].to(device)
                cls_labels = cls_data["label"].unsqueeze(0).to(device)

                optimizer_cls.zero_grad(set_to_none=True)

                with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                    cls_outputs = segcls_model(cls_inputs, mode='cls', modals=cls_modals)
                    cls_loss = cls_loss_function(cls_outputs, cls_labels)
                cls_loss_all += cls_loss
            cls_loss_avg = cls_loss_all / config["cls_batch_size"]

            if grad_scaler is not None:
                grad_scaler.scale(cls_loss_avg).backward()
                grad_scaler.unscale_(optimizer_cls)
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                grad_scaler.step(optimizer_cls)
                grad_scaler.update()
            else:
                cls_loss_avg.backward()
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                optimizer_cls.step()

            # torch.cuda.empty_cache()

            # train MAE
            try:
                mae_data = next(mae_train_loader_iter)
            except StopIteration:
                mae_train_loader_iter = iter(mae_train_loader)
                mae_data = next(mae_train_loader_iter)
            
            mae_inputs = mae_data["mask_image"].to(device)
            mae_targets = mae_data["image"].to(device) 

            optimizer_cls.zero_grad(set_to_none=True)

            with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                mae_outputs = segcls_model(mae_inputs, mode='mae')
                mae_loss = F.l1_loss(mae_outputs, mae_targets)

            if grad_scaler is not None:
                grad_scaler.scale(mae_loss).backward()
                grad_scaler.unscale_(optimizer_cls)
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                grad_scaler.step(optimizer_cls)
                grad_scaler.update()
            else:
                mae_loss.backward()
                torch.nn.utils.clip_grad_norm_(segcls_model.parameters(), 12)
                optimizer_cls.step()
            
            
            epoch_loss += loss.item() + cls_loss_avg.item() + mae_loss.item()

            train_loop.set_postfix({
                'loss': loss.item(),
                'cls_loss': cls_loss_avg.item(),
                'mae_loss': mae_loss.item(),
                'lr': optimizer.param_groups[0]["lr"],
                'consis_weight': consistency_weight_scheduler[step-1+epoch*len(train_loader)],
            })

            wandb.log({
                "train_loss": loss.item(), "train_cls_loss": cls_loss_avg.item(),
                "train_mae_loss": mae_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "cls_lr": optimizer_cls.param_groups[0]["lr"],
                "consis_weight": consistency_weight_scheduler[step-1+epoch*len(train_loader)],
            })

            # torch.cuda.empty_cache()

        epoch_loss /= step

        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"cls learning rate: {optimizer_cls.param_groups[0]['lr']}")
        logger.info(f"consistency weight: {consistency_weight_scheduler[step-1+epoch*len(train_loader)]}")

        scheduler.step()
        scheduler_cls.step()

        # Validation and checkpointing loop:
        if (epoch + 1) % val_interval == 0:
            
            segcls_model.eval()
            with torch.no_grad():
                val_loss = 0.0
                valstep = 0

                val_loop = tqdm(val_loader, total=len(val_loader))
                val_loop.set_description("epoch {:04d}/{:04d}".format(epoch + 1, config["n_epochs"]))
                for val_data in val_loop:
                    val_images = val_data["image"].to(device) 
                    val_labels = val_data["label"].to(device)
                    roi_size = config["transform_kwargs"]["patch_size"]
                    sw_batch_size = 4
                    with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size,
                            segcls_model, overlap=0.75,
                        )
                        val_loss += valloss_function(val_outputs, val_labels)
                    valstep += 1
                val_loss = val_loss / valstep

                
                # classification validation
                cls_val_main1_acc, cls_val_main2_acc, cls_val_aux1_acc, cls_val_aux2_acc, \
                cls_val_main1_auc, cls_val_main2_auc, cls_val_aux1_auc, cls_val_aux2_auc, \
                cls_val_avg1_acc, cls_val_avg2_acc, cls_val_avg1_auc, cls_val_avg2_auc = validate_cls(segcls_model, cls_val_loader, config, epoch, device)

                cls_val_noncontrast_main1_acc, cls_val_noncontrast_main2_acc, cls_val_noncontrast_aux1_acc, cls_val_noncontrast_aux2_acc, \
                cls_val_noncontrast_main1_auc, cls_val_noncontrast_main2_auc, cls_val_noncontrast_aux1_auc, cls_val_noncontrast_aux2_auc, \
                cls_val_noncontrast_avg1_acc, cls_val_noncontrast_avg2_acc, cls_val_noncontrast_avg1_auc, cls_val_noncontrast_avg2_auc = validate_cls(segcls_model, cls_val_noncontrast_loader, config, epoch, device)
                
                wandb.log({
                    "val_mean_dice": 1 - val_loss.item(),
                    "cls_val_main1_acc": cls_val_main1_acc,
                    "cls_val_main2_acc": cls_val_main2_acc,
                    "cls_val_aux1_acc": cls_val_aux1_acc,
                    "cls_val_aux2_acc": cls_val_aux2_acc,
                    "cls_val_main1_auc": cls_val_main1_auc,
                    "cls_val_main2_auc": cls_val_main2_auc,
                    "cls_val_aux1_auc": cls_val_aux1_auc,
                    "cls_val_aux2_auc": cls_val_aux2_auc,
                    "cls_val_noncontrast_main1_acc": cls_val_noncontrast_main1_acc,
                    "cls_val_noncontrast_main2_acc": cls_val_noncontrast_main2_acc,
                    "cls_val_noncontrast_aux1_acc": cls_val_noncontrast_aux1_acc,
                    "cls_val_noncontrast_aux2_acc": cls_val_noncontrast_aux2_acc,
                    "cls_val_noncontrast_main1_auc": cls_val_noncontrast_main1_auc,
                    "cls_val_noncontrast_main2_auc": cls_val_noncontrast_main2_auc,
                    "cls_val_noncontrast_aux1_auc": cls_val_noncontrast_aux1_auc,
                    "cls_val_noncontrast_aux2_auc": cls_val_noncontrast_aux2_auc,
                    "cls_val_avg1_acc": cls_val_avg1_acc,
                    "cls_val_avg2_acc": cls_val_avg2_acc,
                    "cls_val_avg1_auc": cls_val_avg1_auc,
                    "cls_val_avg2_auc": cls_val_avg2_auc,
                    "cls_val_noncontrast_avg1_acc": cls_val_noncontrast_avg1_acc,
                    "cls_val_noncontrast_avg2_acc": cls_val_noncontrast_avg2_acc,
                    "cls_val_noncontrast_avg1_auc": cls_val_noncontrast_avg1_auc,
                    "cls_val_noncontrast_avg2_auc": cls_val_noncontrast_avg2_auc,
                })

                logger.info(f"cls_val_main1_acc: {cls_val_main1_acc:.4f}")
                logger.info(f"cls_val_main2_acc: {cls_val_main2_acc:.4f}")
                logger.info(f"cls_val_aux1_acc: {cls_val_aux1_acc:.4f}")
                logger.info(f"cls_val_aux2_acc: {cls_val_aux2_acc:.4f}")
                logger.info(f"cls_val_main1_auc: {cls_val_main1_auc:.4f}")
                logger.info(f"cls_val_main2_auc: {cls_val_main2_auc:.4f}")
                logger.info(f"cls_val_aux1_auc: {cls_val_aux1_auc:.4f}")
                logger.info(f"cls_val_aux2_auc: {cls_val_aux2_auc:.4f}")
                logger.info(f"cls_val_noncontrast_main1_acc: {cls_val_noncontrast_main1_acc:.4f}")
                logger.info(f"cls_val_noncontrast_main2_acc: {cls_val_noncontrast_main2_acc:.4f}")
                logger.info(f"cls_val_noncontrast_aux1_acc: {cls_val_noncontrast_aux1_acc:.4f}")
                logger.info(f"cls_val_noncontrast_aux2_acc: {cls_val_noncontrast_aux2_acc:.4f}")
                logger.info(f"cls_val_noncontrast_main1_auc: {cls_val_noncontrast_main1_auc:.4f}")
                logger.info(f"cls_val_noncontrast_main2_auc: {cls_val_noncontrast_main2_auc:.4f}")
                logger.info(f"cls_val_noncontrast_aux1_auc: {cls_val_noncontrast_aux1_auc:.4f}")
                logger.info(f"cls_val_noncontrast_aux2_auc: {cls_val_noncontrast_aux2_auc:.4f}")
                logger.info(f"cls_val_avg1_acc: {cls_val_avg1_acc:.4f}")
                logger.info(f"cls_val_avg2_acc: {cls_val_avg2_acc:.4f}")
                logger.info(f"cls_val_avg1_auc: {cls_val_avg1_auc:.4f}")
                logger.info(f"cls_val_avg2_auc: {cls_val_avg2_auc:.4f}")
                logger.info(f"cls_val_noncontrast_avg1_acc: {cls_val_noncontrast_avg1_acc:.4f}")
                logger.info(f"cls_val_noncontrast_avg2_acc: {cls_val_noncontrast_avg2_acc:.4f}")
                logger.info(f"cls_val_noncontrast_avg1_auc: {cls_val_noncontrast_avg1_auc:.4f}")
                logger.info(f"cls_val_noncontrast_avg2_auc: {cls_val_noncontrast_avg2_auc:.4f}")
                
                
                cls_score = (cls_val_main1_acc + cls_val_main2_acc + cls_val_aux1_acc + cls_val_aux2_acc + \
                             cls_val_main1_auc + cls_val_main2_auc + cls_val_aux1_auc + cls_val_aux2_auc + \
                            cls_val_noncontrast_main1_acc + cls_val_noncontrast_main2_acc + \
                            cls_val_noncontrast_aux1_acc + cls_val_noncontrast_aux2_acc + \
                            cls_val_noncontrast_main1_auc + cls_val_noncontrast_main2_auc + \
                            cls_val_noncontrast_aux1_auc + cls_val_noncontrast_aux2_auc) / 16
                if cls_score > best_cls_score:
                    best_cls_score = cls_score
                    torch.save(
                        segcls_model.state_dict(),
                        os.path.join(work_dir,
                        "best_cls_dict_epoch{:04d}_score{:.4f}.pth".format(epoch + 1, best_cls_score)),
                    )
                    logger.info("saved new best cls loss model")

                    # Remove old checkpoints
                    old_ckpts = sorted(glob.glob(os.path.join(work_dir, "best_cls_dict_epoch*.pth")))[:-config["max_best_ckpt"]]
                    if old_ckpts:
                        for old_ckpt in old_ckpts:
                            os.remove(old_ckpt)


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_epoch = epoch + 1
                    torch.save(
                        segcls_model.state_dict(),
                        os.path.join(work_dir,
                        "best_dict_epoch{:04d}_dice{:.4f}.pth".format(epoch + 1, 1 - val_loss.item())),
                    )
                    logger.info("saved new best loss model")

                    # Remove old checkpoints
                    old_ckpts = sorted(glob.glob(os.path.join(work_dir, "best_dict_epoch*.pth")))[:-config["max_best_ckpt"]]
                    if old_ckpts:
                        for old_ckpt in old_ckpts:
                            os.remove(old_ckpt)

                    # semi supervision
                    if not semisup and (1 - val_loss.item() > config["semisup_dice"]):
                        semisup = True
                        logger.info(f"start semi-supervision training")

                        semisup_predict_dir = os.path.join(work_dir, "semisup_predict")

                        semisup_test_dataset = LiverTestforSemiSupDataset(
                            config["data_dir"],
                            config["process_data_dir"],
                            config["transform_kwargs"],
                        )
                        semisup_test_loader = DataLoader(
                            semisup_test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            # pin_memory=True,
                            collate_fn=list_data_collate,
                        )
                    
                    if semisup:
                    
                        logger.info(f"predict segmentation for semi-supervision training")
                        segcls_model.eval()
                        with torch.no_grad():
                            test_loop = tqdm(semisup_test_loader, total=len(semisup_test_loader))
                            test_loop.set_description("epoch {:04d}/{:04d}".format(epoch + 1, config["n_epochs"]))
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

                                save_folder = os.path.join(semisup_predict_dir, test_data["vendor"][0], test_data["patient"][0])
                                os.makedirs(save_folder, exist_ok=True)
                                semisup_test_dataset.save_image(test_data, save_folder)
                            
                        # semi supervised train dataloader
                        semisup_train_dataset = LiverSemiSupDataset(
                            config["data_dir"],
                            config["process_data_dir"],
                            semisup_predict_dir,
                            config["transform_kwargs"],
                        )
                        semisup_train_loader = DataLoader(
                            semisup_train_dataset,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            num_workers=config["num_workers"],
                            # pin_memory=True,
                            collate_fn=list_data_collate,
                        )
                        semisup_train_loader_iter = iter(semisup_train_loader)

                
                logger.info(
                    "current epoch: {} current mean dice: {:.4f}"
                    " best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, 1 - val_loss.item(),
                        1 - best_val_loss.item(), best_loss_epoch,
                    )
                )

            checkpoint = {
                "state_dict": segcls_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer_cls": optimizer_cls.state_dict(),
                "scheduler_cls": scheduler_cls.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss.item(),
                "best_cls_score": best_cls_score,
                "semisup": semisup,
            }
            torch.save(
                checkpoint,
                os.path.join(work_dir, 'epoch{:04d}.pth'.format(epoch + 1)),
            )

            # Remove old checkpoints
            old_ckpts = sorted(glob.glob(os.path.join(work_dir, "epoch*.pth")))[:-config["max_ckpt"]]
            if old_ckpts:
                for old_ckpt in old_ckpts:
                    os.remove(old_ckpt)
            
            # torch.cuda.empty_cache()

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()

    main(args)