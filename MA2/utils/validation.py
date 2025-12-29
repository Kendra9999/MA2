from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from torch import autocast

from utils.utils import dummy_context

def validate_cls(segcls_model, cls_val_loader, config, epoch, device):

    cls_main1_targets, cls_main2_targets, cls_aux1_targets, cls_aux2_targets = [], [], [], []
    cls_main1_preds, cls_main2_preds, cls_aux1_preds, cls_aux2_preds = [], [], [], []
    cls_main1_probs, cls_main2_probs, cls_aux1_probs, cls_aux2_probs = [], [], [], []

    cls_avg1_preds, cls_avg2_preds, cls_avg1_probs, cls_avg2_probs = [], [], [], []
    
    cls_val_loop = tqdm(cls_val_loader, total=len(cls_val_loader))
    cls_val_loop.set_description("epoch {:04d}/{:04d}".format(epoch + 1, config["n_epochs"]))
    for cls_val_data in cls_val_loop:
        cls_val_images = cls_val_data["image"].squeeze(0).unsqueeze(1).to(device) 
        cls_val_modals = cls_val_data["modal"].to(device)
        cls_val_labels = cls_val_data["label"]

        with autocast(device.type, enabled=True) if device.type == 'cuda' else dummy_context():
            cls_val_outputs = segcls_model(cls_val_images, mode='cls', modals=cls_val_modals)
            cls_val_probs = [cls_val_output.softmax(dim=1) for cls_val_output in cls_val_outputs]
            cls_val_preds = [cls_val_prob.argmax(dim=1) for cls_val_prob in cls_val_probs]
            
            cls_main1_probs.append(1 - cls_val_probs[0][:, 0].item())
            cls_main2_probs.append(cls_val_probs[0][:, 3].item())
            cls_aux1_probs.append(cls_val_probs[1][:, 1].item())
            cls_aux2_probs.append(cls_val_probs[2][:, 1].item())
            
            cls_main1_targets.append(cls_val_labels.item() != 0)
            cls_main2_targets.append(cls_val_labels.item() == 3)
            cls_aux1_targets.append(cls_val_labels.item() != 0)
            cls_aux2_targets.append(cls_val_labels.item() == 3)

            cls_main1_preds.append(cls_val_preds[0].item() != 0)
            cls_main2_preds.append(cls_val_preds[0].item() == 3)
            cls_aux1_preds.append(cls_val_preds[1].item())
            cls_aux2_preds.append(cls_val_preds[2].item())

            cls_avg1_probs.append(((1 - cls_val_probs[0][:, 0].item()) + cls_val_probs[1][:, 1].item()) / 2)
            cls_avg2_probs.append((cls_val_probs[0][:, 3].item() + cls_val_probs[2][:, 1].item()) / 2)
            cls_avg1_preds.append(cls_avg1_probs[-1] >= 0.5)
            cls_avg2_preds.append(cls_avg2_probs[-1] >= 0.5)

    cls_val_main1_acc = accuracy_score(cls_main1_targets, cls_main1_preds)
    cls_val_main2_acc = accuracy_score(cls_main2_targets, cls_main2_preds)
    cls_val_aux1_acc = accuracy_score(cls_aux1_targets, cls_aux1_preds)
    cls_val_aux2_acc = accuracy_score(cls_aux2_targets, cls_aux2_preds)

    cls_val_main1_auc = roc_auc_score(cls_main1_targets, cls_main1_probs)
    cls_val_main2_auc = roc_auc_score(cls_main2_targets, cls_main2_probs)
    cls_val_aux1_auc = roc_auc_score(cls_aux1_targets, cls_aux1_probs)
    cls_val_aux2_auc = roc_auc_score(cls_aux2_targets, cls_aux2_probs)

    cls_val_avg1_acc = accuracy_score(cls_main1_targets, cls_avg1_preds)
    cls_val_avg2_acc = accuracy_score(cls_main2_targets, cls_avg2_preds)
    cls_val_avg1_auc = roc_auc_score(cls_main1_targets, cls_avg1_probs)
    cls_val_avg2_auc = roc_auc_score(cls_main2_targets, cls_avg2_probs)
    
    return cls_val_main1_acc, cls_val_main2_acc, cls_val_aux1_acc, cls_val_aux2_acc, \
            cls_val_main1_auc, cls_val_main2_auc, cls_val_aux1_auc, cls_val_aux2_auc, \
            cls_val_avg1_acc, cls_val_avg2_acc, cls_val_avg1_auc, cls_val_avg2_auc