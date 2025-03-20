import os
import lmdb
import blosc
import pickle
import yaml
import random
import numpy as np
from torch.utils.data import Dataset

def preprocess(target_state_data, preprocess_cfg, state_name, state_data_spec, state_data):
    if state_name in preprocess_cfg:
        # mask
        if "mask" in preprocess_cfg[state_name]:
            mask_name, cond, replace_num = preprocess_cfg[state_name]["mask"]
            mask_idx = state_data_spec.index(mask_name)
            mask = (state_data[mask_idx]==cond)
            target_state_data[mask] = replace_num

        # clip
        if "clip" in preprocess_cfg[state_name]:
            min_val, max_val = preprocess_cfg[state_name]["clip"]
            target_state_data = np.clip(target_state_data, min_val, max_val)
        # normalize
        if "normalize" in preprocess_cfg[state_name]:
            min_val, max_val = preprocess_cfg[state_name]["normalize"]
            target_state_data = (target_state_data-min_val)/(max_val-min_val)
    return target_state_data

def rotate_map(state, gt, k):
    for _ in range(k):
        state = np.rot90(state, -1, axes=(1,2))
        gt = np.rot90(gt, -1, axes=(1,2))
    return state, gt

def flip_map(state, gt):
    state = state[:, ::-1, :]
    gt = gt[:, ::-1, :]
    return state, gt


class ILDataset(Dataset):
    def __init__(self, df, cfg, phase="train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.phase = phase

        self.env = lmdb.open(os.path.join(cfg["dataset"]["dataset_path"], "data.lmdb"), readonly=True, lock=False, readahead=False, meminit=False)
        with open(os.path.join(cfg["dataset"]["dataset_path"], "cfg.yaml"), "r") as f:
            self.dataset_cfg = yaml.safe_load(f)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Load Data
        target_row = self.df.loc[index]
        data_idx = target_row["idx"].item()
        with self.env.begin() as txn:
            state_data = txn.get(f"state:{data_idx}".encode())
            state_data = pickle.loads(blosc.decompress(state_data))
            action = txn.get(f"action:{data_idx}".encode())
            action = pickle.loads(blosc.decompress(action))

        state_data_spec = self.dataset_cfg["target_states"]
        using_states = self.cfg["dataset"]["using_states"]
        using_meta_states = self.cfg["dataset"]["using_meta_states"]
        preprocess_cfg = self.cfg["dataset"]["preprocess_cfg"]

        # create state map
        state_map = []
        for state_name in using_states:
            state_idx = state_data_spec.index(state_name)
            target_state_data = state_data[state_idx]

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, state_data_spec, state_data)

            state_map.append(target_state_data)
        state_map = np.array(state_map).astype(np.float32)

        # create meta state
        global_feature = []
        for state_name in using_meta_states:
            state_idx = state_data_spec.index(state_name)
            target_state_data = state_data[state_idx]

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, state_data_spec, state_data)

            global_feature.append(target_state_data[0][0].item())
        global_feature = np.array(global_feature).astype(np.float32)

        # SAP area
        target_x = int(target_row["target_x"].item())
        target_y = int(target_row["target_y"].item())
        unit_sap_range = int(state_data[state_data_spec.index("meta.unit_sap_range")][0][0].item())
        self_unit_pos = state_data[state_data_spec.index("unit.self_unit_pos")]
        # target_sap_area
        target_sap_area = np.zeros((24, 24))
        for dy in range(-1*unit_sap_range, unit_sap_range+1):
            for dx in range(-1*unit_sap_range, unit_sap_range+1):
                ny = target_y + dy
                nx = target_x + dx
                if ny<0 or ny>=24 or nx<0 or nx>=24:
                    continue
                target_sap_area[ny][nx] = 1.0
        target_sap_area = target_sap_area[np.newaxis, :, :]
        
        # other_sap_area
        other_sap_area = np.zeros((24, 24))
        ys, xs = np.where(action[5]==1)
        for y, x in zip(ys, xs):
            if y==target_y and x==target_x:
                continue
            for dy in range(-1*unit_sap_range, unit_sap_range+1):
                for dx in range(-1*unit_sap_range, unit_sap_range+1):
                    ny = y + dy
                    nx = x + dx
                    if ny<0 or ny>=24 or nx<0 or nx>=24:
                        continue
                    other_sap_area[ny][nx] = (self_unit_pos[y][x]*action[5][y][x]).item()
        other_sap_area = other_sap_area[np.newaxis, :, :]
        state_map = np.concatenate([state_map, target_sap_area, other_sap_area], axis=0).astype(np.float32)
        

        # GT label
        dx = int(action[6][target_y, target_x].item())
        dy = int(action[7][target_y, target_x].item())
        sap_label = np.zeros((24, 24))
        sap_label[target_y+dy][target_x+dx] = 1
        sap_label_mask = target_sap_area.copy()
        sap_label_mask[:, target_y+dy, target_x+dx] = sap_label_mask.sum().item()

        sap_label = sap_label[np.newaxis, :, :]
        gt = np.concatenate([sap_label, sap_label_mask], axis=0).astype(np.float32)

        # Augmentation
        if self.phase=="train":
            # rotate
            if self.cfg["dataset"]["augmentation"]["rotate"]:
                rot_k = random.randint(0, 3)
                state_map, gt = rotate_map(state_map, gt, rot_k)
                state_map = state_map.copy()
                gt = gt.copy()
            # flip
            if self.cfg["dataset"]["augmentation"]["flip"]:
                if random.randint(0, 1) == 1:
                    state_map, gt = flip_map(state_map, gt)
                    state_map = state_map.copy()
                    gt = gt.copy()

        return state_map, global_feature, gt