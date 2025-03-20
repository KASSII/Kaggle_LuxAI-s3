import argparse
import os
import datetime
import yaml
import shutil
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from ds import ILDataset
from model import ILModel
from utils import seed_everything, split_into_folds

### PL Class
class PLDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_df, valid_df):
        super(PLDataModule, self).__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df
    
    def setup(self, stage=None):
        cfg = self.cfg
        self.train_dataset = ILDataset(self.train_df, cfg, phase="train")
        self.valid_dataset = ILDataset(self.valid_df, cfg, phase="valid")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg["dataloader"]["train"])
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg["dataloader"]["valid"])

class PLModel(pl.LightningModule):
    def __init__(self, cfg):
        super(PLModel, self).__init__()
        self.cfg = cfg
        in_c = len(cfg["dataset"]["using_states"])+2
        global_feature_ch = len(cfg["dataset"]["using_meta_states"])
        self.model = ILModel(in_c=in_c, global_feature_ch=global_feature_ch)
        self.train_losses = []
    
    def forward(self, state_maps, global_features):
        return self.model(state_maps, global_features)
    
    def masked_cross_entropy_loss(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes cross-entropy loss with masking.

        Args:
            pred (torch.Tensor): Predicted values (B, 1, 24, 24)
            gt (torch.Tensor): Ground truth labels (B, 1, 24, 24), where only one coordinate is 1 and the rest are 0
            mask (torch.Tensor): Mask indicating regions to include in loss calculation (B, 1, 24, 24), where 1 means included, 0 means ignored

        Returns:
            torch.Tensor: The average loss over the masked region in the batch.
        """
        B = pred.shape[0]

        pred = pred.view(B, 24, 24)
        gt = gt.view(B, 24, 24)
        mask = mask.view(B, 24, 24).float()

        pred_probs = F.softmax(pred.view(B, -1), dim=-1)
        gt_indices = gt.view(B, -1).argmax(dim=-1)
        gt_probs = pred_probs[torch.arange(B), gt_indices]
        loss = -torch.log(gt_probs + 1e-9)


        mask_pixels = mask.view(B, -1).sum(dim=-1)
        mask_loss = (loss * mask.view(B, -1).max(dim=-1).values)
        mask_loss = torch.where(mask_pixels > 0, mask_loss, torch.zeros_like(mask_loss))
        return mask_loss.sum() / (mask_pixels.sum() + 1e-9)

    def training_step(self, batch, batch_idx):
        state_maps, global_features, labels = batch
        pred = self(state_maps, global_features)
        gt = labels[:, 0:1, :, :]
        mask = labels[:, 1:2, :, :]
        loss = self.masked_cross_entropy_loss(pred, gt, mask)

        self.train_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        state_maps, global_features, labels = batch
        pred = self(state_maps, global_features)
        gt = labels[:, 0:1, :, :]
        mask = labels[:, 1:2, :, :]
        loss = self.masked_cross_entropy_loss(pred, gt, mask)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_train_loss = np.array(self.train_losses).mean()
        self.log('avg_train_loss', avg_train_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.train_losses.clear()
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = optim.AdamW(self.parameters(), **self.cfg["optimizer"])
        
        # Scheduler
        train_loader = self.trainer.datamodule.train_dataloader()
        num_training_steps = len(train_loader) * self.trainer.max_epochs
        num_warmup_steps = int(self.cfg["scheduler"]["warmup"] * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    

### Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    ### Fix Random Seed
    seed_everything(cfg["seed"])

    ### Output dir
    if cfg["add_unique_str"]:
        dt_now = datetime.datetime.now()
        run_name = dt_now.strftime("%Y%m%d_%H%M%S") + f"_{cfg['run_name']}"
    else:
        run_name = cfg['run_name']
    _output_dir = os.path.join(cfg["output_dir"], cfg["exp_name"], run_name)
    os.makedirs(os.path.join(cfg["output_dir"], cfg["exp_name"]), exist_ok=True)
    os.makedirs(_output_dir, exist_ok=True)

    ### Save Cfg
    shutil.copy(args.cfg_path, os.path.join(_output_dir, "config.yaml"))

    ### Load CSV
    data_df = pd.read_csv(os.path.join(cfg["dataset"]["dataset_path"], "sap_data_list.csv"))
    if "use_submit_ids" in cfg["dataset"]:
        use_submit_ids = cfg["dataset"]["use_submit_ids"]
        data_df = data_df.query("sub_id in @use_submit_ids").reset_index(drop=True)
    data_df = split_into_folds(data_df, cfg["dataset"]["n_splits"])

    ### Fold Loop
    for fold in cfg["train_folds"]:
        if fold > -1:
            print(f"### fold{fold} ###")
            output_dir = os.path.join(_output_dir, f"fold{fold}")
        else:
            print(f"### full data train ###")
            output_dir = os.path.join(_output_dir, f"fulldata")
        os.makedirs(output_dir, exist_ok=True)
        
        ### Create CSV
        if fold > -1:
            train_df = data_df.query("fold!=@fold")
            valid_df = data_df.query("fold==@fold")
        else:
            train_df = data_df.copy()
            valid_df = data_df.query("fold==0")

        ### DataModule
        data_module = PLDataModule(cfg, train_df, valid_df)

        ### Model
        model = PLModel(cfg)
        if "pretrined_weight" in cfg:
            model.model.load_state_dict(torch.load(cfg["pretrined_weight"]))
            print(f"Load Weight: {cfg['pretrined_weight']}")

        ### Train
        # callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=output_dir,
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=2,
            verbose=True,
            mode='min'
        )
        call_backs = [checkpoint_callback]
        if cfg["do_early_stopping"]:
            call_backs.append(early_stop_callback)

        # loggers
        csv_logger = CSVLogger(output_dir, name="csv_log")
        logger = [csv_logger]
        if cfg["wandb_log"]:
            wandb_cfg = {
                "project": f"LuxS3_IL",
                "group": cfg['task_type'],
                "job_type": cfg["exp_name"],
                "name": cfg["run_name"]
            }
            wandb_logger = WandbLogger(**wandb_cfg)
            wandb_logger.experiment.config.update(cfg)
            logger.append(wandb_logger)

        # exec
        trainer = pl.Trainer(
            **cfg["trainer"], 
            callbacks=call_backs, 
            logger=logger
        )
        trainer.fit(model, data_module)

        # PostProcess
        if cfg["wandb_log"]:
            wandb.save(args.cfg_path)
            wandb.finish()
        del trainer
        del model
        del data_module
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()