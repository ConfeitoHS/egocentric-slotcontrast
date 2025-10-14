# import argparse
# import pathlib
from slotcontrast import models, configuration, metrics

config_overrides = None

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--continue",
#     dest="continue_from",
#     type=pathlib.Path,
#     help="Continue training from this log folder or checkpoint path",
# )
# parser.add_argument("--config_overrides_file", help="Configuration to override")
# parser.add_argument("config", help="Configuration to run")                                         # positional argument
# parser.add_argument("config_overrides", nargs="*", help="Additional arguments")                    # positional argument


def load_model(args, config_overrides = None):
    if config_overrides is None:
        config_overrides = args['config_overrides']
        
    config = configuration.load_config(args['config'], config_overrides)   # load the configuration yml files from args.config and args.config_overrides(optional)
    if args['config_overrides_file'] is not None:
        config = configuration.override_config(
            config,
            override_config_path=args['config_overrides_file'],
            additional_overrides=config_overrides,
        )
    
    if config.train_metrics is not None:
        train_metrics = {
            name: metrics.build(config) for name, config in config.train_metrics.items()
        }
    else:
        train_metrics = None
    
    if config.val_metrics is not None:
        val_metrics = {name: metrics.build(config) for name, config in config.val_metrics.items()}
    else:
        val_metrics = None

    if 'continue_from' in args.keys():
        config.model['load_weights'] = args['continue_from']
    else:
        config.model['load_weights'] = None

    # print(config.model['load_weights'])
    print("/n      <===Load SlotContrast Model from {}===>      \n\n".format(config.model['load_weights']))
    import time; time.sleep(3)

    model = models.build(
        config.model, 
        config.optimizer, 
        train_metrics, 
        val_metrics) 
    # import pdb; pdb.set_trace()
    return model

import torch, torch.nn as nn
import math
import torch.nn.functional as F
class SCModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = load_model(args)
        self.fc = nn.Identity()
        self.pooling = args['pooling']

        # self.slot_pooling = args.slot_pooling
    
    def forward(self, x): # input:[64,3,224,224], output:[64,22]
        output = self.backbone(x)  
        output = output['processor']['corrector']['slots']  # slots: [64, 1, 11, 64]
        if self.pooling == "mean":
            output = output.mean(dim=2).squeeze(1)  # [64, 64]
        elif self.pooling == "cat":
            output = output.squeeze(1).reshape(output.shape[0],-1)


        # B, K, D = slots.shape
        # if self.slot_pooling == 'mean':
        #     slots = slots.mean(dim=1)  # [64,15,192] -> [64,192]
        # elif self.slot_pooling == 'cat':
        #     slots = slots.contiguous().view(B, K * D)  # [64,15,192] -> [64,15*192]
        # else:
        #     raise ValueError(f"Unknown slot pooling method: {self.slot_pooling}")
        
        return self.fc(output)
    
    def get_mask(self, x):
        video = x.unsqueeze(1)
        inputs = {
            'video': video,
            'batch_padding_mask': torch.tensor(False).repeat(x.shape[0]).to(x.device)
        }
        output = self.backbone(inputs)  
        # import pdb; pdb.set_trace()
        slots = output['processor']['corrector']['slots']  # slots: [64, 1, 11, 64]
        masks = output['processor']['corrector']['masks']  # masks: [64, 1, 11, 256]
        
        # [1, 1, 11, 256]  [24, 3, 4096, 15]
        # import pdb; pdb.set_trace()
        attns = masks.transpose(-1,-2)
        B,T,P,K= attns.size()  
        H_enc = int(math.isqrt(P))
        # import pdb; pdb.set_trace()
        attns = attns\
            .transpose(-1,-2)\
            .reshape(B, T, K, 1, H_enc, H_enc)
            # .repeat_interleave(224 // H_enc, dim=-2)\
            # .repeat_interleave(224 // H_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = F.interpolate(
            attns.flatten(0, 2),          # [B*T*K, 1, H_enc, H_enc]
            size=(224, 224),
            mode="bilinear",              #  "bicubic"
            align_corners=False
        ).view(B, T, K, 1, 224, 224)
        
        # import pdb; pdb.set_trace()
        attns = video.unsqueeze(2) * attns  + (1. - attns)
        
        # attns = attns\
        #     .transpose(-1,-2)\
        #     .reshape(B, T, K, 1, H_enc, H_enc)

        #     .repeat_interleave(224 // H_enc, dim=-2)\
        #     .repeat_interleave(224 // H_enc, dim=-1)          # B, T, num_slots, 1, H, W
        # attns = video.unsqueeze(2) * attns + (1. - attns)

        return attns
        
    
class SCModel_CVCL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = load_model(args)
        self.fc = nn.Identity()
        self.pooling = args['pooling']

        # self.slot_pooling = args.slot_pooling
    
    def forward(self, x): # input:[B,3,224,224] output:[B, 512]
        inputs = {
            'video': x.unsqueeze(1),
            'batch_padding_mask': torch.tensor(False).repeat(x.shape[0]).to(x.device)
        }
        output = self.backbone(inputs)  
        # import pdb; pdb.set_trace()
        output = output['processor']['corrector']['slots']  # slots: [B, 1, 11, 64]
        if self.pooling == "mean":
            output = output.mean(dim=2).squeeze(1)  # [B, 64]
        elif self.pooling == "cat":
            output = output.squeeze(1).reshape(output.shape[0],-1)

        # import pdb; pdb.set_trace()
        return self.fc(output), None

class SCModel_CVCL_MASKED(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = load_model(args)
        self.fc = nn.Identity()
        self.pooling = args['pooling']

        # self.slot_pooling = args.slot_pooling

    def forward(self, x): # input:[B,3,224,224] output:[B, 512]
        inputs = {
            'video': x.unsqueeze(1),
            'batch_padding_mask': torch.tensor(False).repeat(x.shape[0]).to(x.device)
        }
        output = self.backbone(inputs)
        # import pdb; pdb.set_trace()
        slots = output['processor']['corrector']['slots']  # slots: [B, 1, 11, 64]
        masks = output['processor']['corrector']['masks']   # [B,1,11,256]

        attns = masks.transpose(-1,-2)
        B,T,P,K= attns.size()
        H_enc = int(math.isqrt(P))
        attns = attns\
            .transpose(-1,-2)\
            .reshape(B, T, K, 1, H_enc, H_enc)
        attns = F.interpolate(
            attns.flatten(0, 2),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).view(B, T, K, 1, 224, 224)
            
        scores = attns.sum(dim=(3,4,5)) # (B,T,K)
        _,top2_idx = scores.topk(2,dim=-1) # (B,T,2)
        idx_expanded = top2_idx.unsqueeze(-1).expand(-1, -1, -1, slots.size(-1))
        selected_slots = torch.gather(slots, 2, idx_expanded)

        if self.pooling == "mean":
            output = selected_slots.mean(dim=2).squeeze(1)  # [B, 64]
        elif self.pooling == "cat":
            output = selected_slots.squeeze(1).reshape(output.shape[0],-1)

        return self.fc(output), None

class SCModel_CVCL_VISION_ENCODER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = load_model(args)
        self.fc = nn.Identity()
        self.pooling = args['pooling']
        self.linear = False
        if "linear" in args:
            self.linear = args['linear']

        # self.slot_pooling = args.slot_pooling
    
    def forward(self, x): # input:[B,3,224,224]  output: [B,512]
        inputs = {
            'video': x.unsqueeze(1),
            'batch_padding_mask': torch.tensor(False).repeat(x.shape[0]).to(x.device)
        }
        
        output = self.backbone(inputs)  
        output = output['encoder']['backbone_features']  # slots: [B, T, 256, 768]
        if self.pooling == "mean":
            output = output.mean(dim=2).squeeze(1)  # [B,768]
        elif self.pooling == "cat":
            output = output.squeeze(1).reshape(output.shape[0],-1)

        if self.linear:
            return self.fc(output)
        
        return self.fc(output), None


if __name__ == "__main__":
    args = {
        "config": "/home/wz3008/slotcontrast/configs/saycam.yml",
        "continue_from": "/home/wz3008/slotcontrast/logs/saycam/2025-08-25-19-07-43_train_saycam_video/checkpoints/step=90615.ckpt",
        "config_overrides_file": None,
        "config_overrides": None,
        "pooling": "mean",
    }
    # load_model(args)
    model = SCModel_CVCL_MASKED(args)

