import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetOutBlock

from .unet import Unet
from .vit import ClassifyViT

class SegCLSModel(nn.Module):
    def __init__(
        self, pretrained_ckpt, n_classes, patch_size, num_modals, n_cls_classes,
        dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16,
    ):
        super(SegCLSModel, self).__init__()
        
        # Initialize base U-Net model
        self.backbone = Unet(dimension, input_nc, output_nc, num_downs, ngf)
        
        if pretrained_ckpt == 'scratch':
            print("Training from random initialization.")
            pass
        else:
            print("Transferring from pretrained network.")
            self.backbone.load_state_dict(torch.load(pretrained_ckpt))
            
        # Add final classification layer
        self.fin_layer = UnetOutBlock(dimension, output_nc, n_classes + 1, False)

        # Classification model
        self.cls_vit = ClassifyViT(feature_size=[p//(2**num_downs) for p in patch_size],
                                   patch_size=[2, 2, 2],
                                   num_modals=num_modals, num_classes_list=[n_cls_classes, 2, 2],
                                   dim_in=ngf*(2**num_downs), dim = 1024, depth = 6,
                                   heads = 8, mlp_dim = 2048)
        
        # mae head
        self.mae_head = nn.Conv3d(
            in_channels=sum([ngf*(2**(i+1)) for i in range(len(self.backbone.encoder_idx))]),
            out_channels=1, kernel_size=1)
        

    def forward(self, x, mode='seg', modals=None):
        if mode == 'seg':
            y = self.backbone(x)
            y = self.fin_layer(y)
            return y
        
        elif mode == 'cls':
            y = self.backbone(x, out_layer_ids=[self.backbone.decoder_idx[0]-3])[0]
            y = self.cls_vit(y, modals.view(-1))
            return y
        
        elif mode == 'mae':
            ys = self.backbone(x, 
                out_layer_ids=[self.backbone.encoder_idx[i]+5 for i in range(len(self.backbone.encoder_idx))])
            ys = [F.interpolate(y, size=x.shape[2:], mode='trilinear', align_corners=False) for y in ys]
            ys = torch.cat(ys, dim=1)
            ys = self.mae_head(ys)
            return ys

        else:
            raise ValueError(f"Unknown mode: {mode}")