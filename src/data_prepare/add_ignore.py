import argparse
import json
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="datas/scraping")
parser.add_argument("--sub_id", required=True)
parser.add_argument("--lb_csv", default="datas/publicLB_250211.csv")
parser.add_argument("--rank_thresh", type=int, default=50)
args = parser.parse_args()

root_dir = args.root_dir
target_sub_id = args.sub_id

df = pd.read_csv(os.path.join(root_dir, target_sub_id, "data_list.csv"))
lb_df = pd.read_csv(args.lb_csv)

rank_info = {}
opp_team_list = df["opp_team_name"].unique().tolist()
for opp_team in opp_team_list:
    try:
        rank = lb_df.query("TeamName==@opp_team")["Rank"].values[0].item()
        rank_info[opp_team] = rank
    except:
        print(f"Not Found: {opp_team}")
        rank_info[opp_team] = 10000


ignores = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    opp_team_name = row["opp_team_name"]
    rank = rank_info[opp_team_name]
    if rank > args.rank_thresh:
        ignores.append(True)
    else:
        ignores.append(False)
df["ignore"] = ignores
df.to_csv(os.path.join(root_dir, target_sub_id, "data_list.csv"), index=False)

print(df)
print(f"data num: {len(df)}")
print(f"win data num: {len(df.query('reward>=3'))}")
print("-----")
_df = df.query("not ignore")
print(f"data num: {len(_df)}")
print(f"win data num: {len(_df.query('reward>=3'))}")