import argparse
from tqdm import tqdm
import os
import json
import yaml
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from model import ILModel

### PL Class
class PLModel(pl.LightningModule):
    def __init__(self, cfg):
        super(PLModel, self).__init__()
        self.cfg = cfg
        in_c = len(cfg["dataset"]["using_states"])+len(cfg["dataset"]["using_pre_states"])
        global_feature_ch = len(cfg["dataset"]["using_meta_states"])
        self.model = ILModel(in_c=in_c, global_feature_ch=global_feature_ch)
    
    def forward(self, state_maps, global_features):
        return self.model(state_maps, global_features)
    

### Save Weight
def save_weight():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()
    
    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ### Load Model
    model = PLModel.load_from_checkpoint(args.ckpt_path, cfg=cfg, strict=False, map_location="cpu")
    model.eval()
    save_path = os.path.join(os.path.dirname(args.ckpt_path), "weight.pth")
    torch.save(model.model.state_dict(), save_path)

if __name__ == '__main__':
    save_weight()