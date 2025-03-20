import argparse
import os
import numpy as np
import pandas as pd
import yaml
import lmdb
import blosc
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True)
args = parser.parse_args()

data_df = pd.read_csv(os.path.join(args.dataset_path, "data_list.csv"))
with open(os.path.join(args.dataset_path, "cfg.yaml"), "r") as f:
    cfg = yaml.safe_load(f)
env = lmdb.open(os.path.join(args.dataset_path, "data.lmdb"), readonly=True, lock=False, readahead=False, meminit=False)

sap_data_info = {
    "idx": [],
    "sub_id": [],
    "episode_id": [],
    "steps": [],
    "match_round": [],
    "match_steps": [],
    "target_x": [],
    "target_y": []
}
for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
    data_idx = row["idx"].item()
    with env.begin() as txn:
        action = txn.get(f"action:{data_idx}".encode())
        action = pickle.loads(blosc.decompress(action))
    if action[5].sum() == 0:
        continue

    ys, xs = np.where(action[5]==1)
    for y, x in zip(ys, xs):
        sap_data_info["idx"].append(row["idx"])
        sap_data_info["sub_id"].append(row["sub_id"])
        sap_data_info["episode_id"].append(row["episode_id"])
        sap_data_info["steps"].append(row["steps"])
        sap_data_info["match_round"].append(row["match_round"])
        sap_data_info["match_steps"].append(row["match_steps"])
        sap_data_info["target_x"].append(x)
        sap_data_info["target_y"].append(y)
sap_data_df = pd.DataFrame.from_dict(sap_data_info)
sap_data_df.to_csv(os.path.join(args.dataset_path, "sap_data_list.csv"), index=False)
print(sap_data_df)