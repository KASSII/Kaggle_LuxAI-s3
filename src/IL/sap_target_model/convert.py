import argparse
from tqdm import tqdm
import os
import json
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ds import ILDataset
from model import ILModel
from utils import seed_everything, split_into_folds

### PL Class
class PLModel(pl.LightningModule):
    def __init__(self, cfg):
        super(PLModel, self).__init__()
        self.cfg = cfg
        in_c = len(cfg["dataset"]["using_states"])+2
        global_feature_ch = len(cfg["dataset"]["using_meta_states"])
        self.model = ILModel(in_c=in_c, global_feature_ch=global_feature_ch)
    
    def forward(self, state_maps, global_features):
        return self.model(state_maps, global_features)
    

### Convert
def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()
    
    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ### Load Model
    model = PLModel.load_from_checkpoint(args.ckpt_path, cfg=cfg, strict=False, map_location="cpu")
    model.eval()

    ### DataLoader
    data_df = pd.read_csv(os.path.join(cfg["dataset"]["dataset_path"], "sap_data_list.csv"))
    data_df = split_into_folds(data_df, cfg["dataset"]["n_splits"])
    data_df = data_df.query("fold==0").reset_index(drop=True)
    data_df = data_df[0:5].reset_index(drop=True)
    print(f"data_num => {len(data_df)}")

    test_dataset = ILDataset(data_df, cfg, phase="valid")
    test_dl = DataLoader(test_dataset, **cfg["dataloader"]["valid"])

    ### Convert
    print("### Convert ###")
    jit_model = torch.jit.script(model.model)
    save_path = os.path.join(os.path.dirname(args.ckpt_path), "jit_model.pt")
    jit_model.save(save_path)
    jit_model = torch.jit.load(save_path, map_location="cpu")

    ### Check
    print("### Inference ###")
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Inference", leave=True):
            state_maps, global_features, labels = batch
            state_maps = state_maps.to(model.device)
            global_features = global_features.to(model.device)
            outputs = model(state_maps, global_features)
            jit_outputs = jit_model(state_maps, global_features)

            for i in range(len(outputs)):
                print(f"==== {i} ====")
                output = outputs[i][0].cpu()
                jit_output = jit_outputs[i][0].cpu()
                gt = labels[i][0].cpu().numpy()
                mask = labels[i][1].cpu().numpy()
                ys, xs = np.where(mask>0)

                gt = gt[ys.min():ys.max()+1, xs.min():xs.max()+1]
                pred = output[ys.min():ys.max()+1, xs.min():xs.max()+1]
                _h, _w = pred.shape
                prob = F.softmax(pred.reshape(-1), dim=0).reshape(_h, _w)

                jit_pred = jit_output[ys.min():ys.max()+1, xs.min():xs.max()+1]
                _h, _w = jit_pred.shape
                jit_prob = F.softmax(jit_pred.reshape(-1), dim=0).reshape(_h, _w)

                print("gt")
                print(gt.astype(np.int32))
                print("-----")
                print("pred")
                print((prob.numpy()*100))
                print("-----")
                print("pred")
                print((jit_prob.numpy()*100))

if __name__ == '__main__':
    convert()