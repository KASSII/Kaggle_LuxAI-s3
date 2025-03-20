import argparse
import json
import pandas as pd
from glob import glob
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="datas/scraping")
parser.add_argument("--sub_id", required=True)
parser.add_argument("--add_opp_team_name", action="store_true")
args = parser.parse_args()

root_dir = args.root_dir
target_sub_id = args.sub_id

data_list = glob(os.path.join(root_dir, target_sub_id, "json/*.json"))
data_list.sort()

# Get target_team_name
all_team_names = []
for i, data_path in enumerate(data_list):
    with open(data_path) as f:
        data = json.load(f)
    all_team_names += data["info"]["TeamNames"]

    if i>30:
        break
target_team_name = max(all_team_names, key=all_team_names.count)
print(f"target_team_name: {target_team_name}")


# Make data_list
data_dict = {
    "episode_id": [],
    "reward": [],
    "self_id": [],
    "self_status": [],
    "opp_status": []
}
if args.add_opp_team_name:
    data_dict["opp_team_name"] = []

for data_path in tqdm(data_list):
    ep_id = os.path.basename(data_path).split(".")[0]
    with open(data_path) as f:
        data = json.load(f)
    
    try:
        team_names = data["info"]["TeamNames"]
        self_id = team_names.index(target_team_name)
    except:
        print(f"skip: {ep_id}")
        continue

    opp_id = 0 if self_id==1 else 1
    
    reward = data["rewards"][self_id]
    self_status = data["statuses"][self_id]
    opp_status = data["statuses"][opp_id]

    data_dict["episode_id"].append(ep_id)
    data_dict["reward"].append(reward)
    data_dict["self_id"].append(self_id)
    data_dict["self_status"].append(self_status)
    data_dict["opp_status"].append(opp_status)
    if args.add_opp_team_name:
        data_dict["opp_team_name"].append(team_names[opp_id])

df = pd.DataFrame.from_dict(data_dict)
df.to_csv(os.path.join(root_dir, target_sub_id, "data_list.csv"), index=False)
print(df)
print(f"data num: {len(df)}")
print(f"win data num: {len(df.query('reward>=3'))}")
