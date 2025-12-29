import torch
from torch import nn

class CLS_loss(nn.Module):
    def __init__(self, n_cls_classes, cls_main_loss_weights, cls_aux1_loss_weights, cls_aux2_loss_weights,
                 cls_aux_loss_weights):
        super(CLS_loss, self).__init__()
        self.n_cls_classes = n_cls_classes
        self.cls_aux_loss_weights = torch.tensor(cls_aux_loss_weights)

        self.main_cls_loss = nn.CrossEntropyLoss(
                         weight=torch.tensor(cls_main_loss_weights))
        self.aux1_cls_loss = nn.CrossEntropyLoss(
                         weight=torch.tensor(cls_aux1_loss_weights))
        self.aux2_cls_loss = nn.CrossEntropyLoss(
                         weight=torch.tensor(cls_aux2_loss_weights))
        
    def forward(self, cls_outputs, cls_labels):
        main_outputs = cls_outputs[0]
        main_labels = cls_labels[:, 0].long()
        main_loss = self.main_cls_loss(main_outputs, main_labels)

        aux1_outputs = cls_outputs[1]
        aux1_labels = (cls_labels > 0)[:, 0].long()
        aux1_loss = self.aux1_cls_loss(aux1_outputs, aux1_labels)

        aux2_outputs = cls_outputs[2]
        aux2_labels = (cls_labels > 2)[:, 0].long()
        aux2_loss = self.aux2_cls_loss(aux2_outputs, aux2_labels)

        return main_loss + self.cls_aux_loss_weights[0] * aux1_loss + self.cls_aux_loss_weights[1] * aux2_loss
