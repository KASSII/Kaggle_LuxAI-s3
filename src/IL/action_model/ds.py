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

def rotate_map(state, action, k):
    for _ in range(k):
        state = np.rot90(state, -1, axes=(1,2))
        action = np.rot90(action, -1, axes=(1,2))

        new_action = action.copy()
        new_action[2, :, :] = action[1, :, :]    # up -> right
        new_action[3, :, :] = action[2, :, :]    # right -> down
        new_action[4, :, :] = action[3, :, :]    # down -> left
        new_action[1, :, :] = action[4, :, :]    # left -> up

        new_action[6, :, :] = -1*action[7, :, :]    # sap_x -> -1*sap_y
        new_action[7, :, :] = action[6, :, :]    # sap_y -> sap_x
        action = new_action
    return state, action

def flip_map(state, action):
    state = state[:, ::-1, :]
    action = action[:, ::-1, :]

    new_action = action.copy()
    new_action[3, :, :] = action[1, :, :]    # up -> down
    new_action[1, :, :] = action[3, :, :]    # down -> up

    new_action[7, :, :] = -1*action[7, :, :]    # sap_y -> -1*sap_y
    action = new_action
    return state, action


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
        
        # Pre frame data
        steps = target_row["steps"]
        if (steps-1)%101 == 0:
            pre_data_idx = data_idx
        else:
            pre_data_idx = data_idx-1
        with self.env.begin() as txn:
            pre_state_data = txn.get(f"state:{pre_data_idx}".encode())
            pre_state_data = pickle.loads(blosc.decompress(pre_state_data))

        state_data_spec = self.dataset_cfg["target_states"]
        using_states = self.cfg["dataset"]["using_states"]
        using_pre_states = self.cfg["dataset"]["using_pre_states"]
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
        
        for state_name in using_pre_states:
            state_idx = state_data_spec.index(state_name)
            target_state_data = pre_state_data[state_idx]

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, state_data_spec, pre_state_data)

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

        # Augmentation
        if self.phase=="train":
            # rotate
            if self.cfg["dataset"]["augmentation"]["rotate"]:
                rot_k = random.randint(0, 3)
                state_map, action = rotate_map(state_map, action, rot_k)
                state_map = state_map.copy()
                action = action.copy()
            # flip
            if self.cfg["dataset"]["augmentation"]["flip"]:
                if random.randint(0, 1) == 1:
                    state_map, action = flip_map(state_map, action)
                    state_map = state_map.copy()
                    action = action.copy()

        # Action label binarization
        if "label_binarization" in self.cfg["dataset"] and self.cfg["dataset"]["label_binarization"]:
            action = (action>0).astype(np.float16)

        return state_map, global_feature, action