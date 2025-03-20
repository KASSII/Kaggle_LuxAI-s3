import argparse
import pandas as pd
import yaml
from glob import glob
import os
from tqdm import tqdm
import json
import numpy as np
import lmdb
import pickle
import blosc
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_path", required=True)
args = parser.parse_args()
cfg_path = args.cfg_path

nebula_speed_value_list = [0.025, 0.05, 0.1, 0.15]  # 40, 20, 10, 6~7 turn
def generate_sequence(limit):
    sequence = [15]  # 初項
    increments = [6, 7, 7]  # 以降の増分パターン
    index = 0

    while sequence[-1] + increments[index % 3] <= limit:
        sequence.append(sequence[-1] + increments[index % 3])
        index += 1

    return [8] + sequence
seq_015 = generate_sequence(505)


def vote_speed(step):
    vote = np.array([0, 0, 0, 0])
    if (step-1)%40 == 0:
        vote[0] = 4
    else:
        vote[0] = -1

    if (step-1)%20 == 0:
        vote[1] = 2
    else:
        vote[1] = -1

    if (step-1)%10 == 0:
        vote[2] = 1
    else:
        vote[2] = -1

    if step in seq_015:
        vote[3] = 1
    else:
        vote[3] = -1

    return vote

def check_drift_direction(tyle_type_memory, tyle_type, visible_mask):
    direction = 0
    fixed_value = -1

    ### Right Up
    shifted = np.full(tyle_type_memory.shape, fixed_value, dtype=tyle_type_memory.dtype)
    shifted[0:23, 1:24] = tyle_type_memory[1:24, 0:23]
    both_visible_area = (shifted[visible_mask]>=0) & (tyle_type[visible_mask]>=0)
    if np.array_equal(shifted[visible_mask][both_visible_area], tyle_type[visible_mask][both_visible_area]):
        direction += 1
    
    ### Left Down
    shifted = np.full(tyle_type_memory.shape, fixed_value, dtype=tyle_type_memory.dtype)
    shifted[1:24, 0:23] = tyle_type_memory[0:23, 1:24]
    both_visible_area = (shifted[visible_mask]>=0) & (tyle_type[visible_mask]>=0)
    if np.array_equal(shifted[visible_mask][both_visible_area], tyle_type[visible_mask][both_visible_area]):
        direction -= 1
    
    return direction

def shift_map(direction, tyle_type_memory):
    fixed_value = -1

    if direction == 0:
        shifted = tyle_type_memory
    elif direction == 1:
        shifted = np.full(tyle_type_memory.shape, fixed_value, dtype=tyle_type_memory.dtype)
        shifted[0:23, 1:24] = tyle_type_memory[1:24, 0:23]
    elif direction == -1:
        shifted = np.full(tyle_type_memory.shape, fixed_value, dtype=tyle_type_memory.dtype)
        shifted[1:24, 0:23] = tyle_type_memory[0:23, 1:24]
    return shifted

def update_type_map(tyle_type_memory, direction, speed, step):
    do_shift = False
    if speed==0.15:
        do_shift = (step in seq_015)
    else:
        n = int(1/speed)
        do_shift = (step-1)%n==0
    
    if do_shift and step>1:
        tyle_type_memory = shift_map(direction, tyle_type_memory)
    return tyle_type_memory


def convert_action_command(action_command):
    move_mapping = {
        0: 0,    # center -> center
        1: 3,    # up -> down
        2 :4,    # right -> left
        3: 1,    # down -> up
        4: 2,    # left -> right
        5: 5,    # sap -> sap
    }
    return move_mapping[action_command]

def apply_map_symmetry(state_map, visible_mask):
    sym_mask = np.rot90(np.flip(state_map, axis=1), k=-1)
    return np.where(visible_mask, state_map, sym_mask)

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

output_dir = cfg["output_dir"]
os.makedirs(output_dir, exist_ok=True)
shutil.copy(cfg_path, os.path.join(output_dir, "cfg.yaml"))

data_info = {
    "idx": [],
    "sub_id": [],
    "episode_id": [],
    "steps": [],
    "match_round": [],
    "match_steps": [],
}
data_idx = 0

env = lmdb.open(os.path.join(output_dir, "data.lmdb"), map_size=8192**3)
with env.begin(write=True) as txn:
    for sub_id in cfg["target_sub_ids"]:
        print(f"===== {sub_id} =====")
        df = pd.read_csv(os.path.join(cfg["input_root_dir"], sub_id, "data_list.csv"))
        if cfg["use_only_win"]:
            df = df.query("reward>=3").reset_index(drop=True)
        if cfg["use_only_done_status"]:
            df = df.query("self_status=='DONE' and opp_status=='DONE'").reset_index(drop=True)
        if ("exclude_ignore" in cfg) and cfg["exclude_ignore"]:
            if "ignore" in df:
                df = df.query("not ignore").reset_index(drop=True)
        print(f"Episode Num = {len(df)}")
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            self_id = row["self_id"]
            opp_id = 1 if self_id==0 else 0
            episode_id = row["episode_id"]
            with open(os.path.join(cfg["input_root_dir"], f"{sub_id}/json/{episode_id}.json")) as f:
                data = json.load(f)
            
            ### GT nebura drift speed
            nebula_tile_drift_speed = data["steps"][0][0]["info"]["replay"]["params"]["nebula_tile_drift_speed"]
            nebula_tile_drift_speed = round(nebula_tile_drift_speed, 4)

            ### GT unit_sap_dropoff_factor
            unit_sap_dropoff_factor = data["steps"][0][0]["info"]["replay"]["params"]["unit_sap_dropoff_factor"]
            
            ### env_cfg
            env_cfg = data["configuration"]["env_cfg"]
            map_height = env_cfg["map_height"]
            map_width = env_cfg["map_width"]
            unit_move_cost = env_cfg["unit_move_cost"]
            unit_sap_cost = env_cfg["unit_sap_cost"]
            unit_sap_range = env_cfg["unit_sap_range"]

            ### Initialize
            relic_node_map = np.zeros((map_height, map_width)).astype(np.int16)
            relic_node_surround_map = np.zeros((map_height, map_width)).astype(np.int16)
            find_all_relic = False
            point_prob_map = np.zeros((map_height, map_width))
            point_confirm_map = np.zeros((map_height, map_width))
            pre_self_point = 0
            pre_opp_point = 0
            nebula_speed_vote = np.array([0, 0, 0, 0])     # [0.025, 0.05, 0.1, 0.15]
            # nebula_direction = 0
            # nebula_speed = 0
            ### Use GT when create DB ###
            nebula_speed = abs(nebula_tile_drift_speed)
            nebula_direction = 1 if nebula_tile_drift_speed>0 else -1

            tyle_type_memory = None

            ### steps
            for step, step_data in enumerate(data["steps"]):
                # last step per match
                if step % 101 == 0:
                    # nebula confirm check
                    if nebula_direction != 0:
                        ### commentout when use GT ###
                        # nebula_direction = 1 if nebula_direction>0 else -1
                        # nebula_speed = nebula_speed_value_list[nebula_speed_vote.argmax()]
                        tyle_type_memory = update_type_map(tyle_type_memory, nebula_direction, nebula_speed, step)

                    continue

                if (step-1)%101 == 0:
                    # Reset
                    if (step-1)//101 < 3:
                        point_prob_map[point_prob_map<0] = 0.0
                    find_all_relic = False
                    pre_self_point = 0
                    pre_opp_point = 0

                obs_info = {}
                self_data = step_data[self_id]
                obs = self_data["observation"]["obs"]
                obs = obs.replace("true", "1")
                obs = obs.replace("false", "0")
                obs = eval(obs)
                
                ### Unit
                self_unit_position = obs["units"]["position"][self_id]
                opp_unit_position = obs["units"]["position"][opp_id]
                self_energy = obs["units"]["energy"][self_id]
                opp_energy = obs["units"]["energy"][opp_id]
                self_unit_mask = obs["units_mask"][self_id]
                opp_unit_mask = obs["units_mask"][opp_id]

                obs_info["unit"] = {}
                obs_info["unit"]["self_unit_pos"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["self_energy"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["self_enable_move"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["self_enable_sap"] = np.zeros((map_height, map_width)).astype(np.int16)
                for p, e, m in zip(self_unit_position, self_energy, self_unit_mask):
                    if m==1:
                        x, y = p
                        obs_info["unit"]["self_unit_pos"][y][x] += 1
                        obs_info["unit"]["self_energy"][y][x] += e
                        obs_info["unit"]["self_enable_move"][y][x] = 1 if e >= unit_move_cost else 0
                        obs_info["unit"]["self_enable_sap"][y][x] = 1 if e >= unit_sap_cost else 0

                obs_info["unit"]["opp_unit_pos"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["opp_energy"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["opp_enable_move"] = np.zeros((map_height, map_width)).astype(np.int16)
                obs_info["unit"]["opp_enable_sap"] = np.zeros((map_height, map_width)).astype(np.int16)
                for p, e, m in zip(opp_unit_position, opp_energy, opp_unit_mask):
                    if m==1:
                        x, y = p
                        obs_info["unit"]["opp_unit_pos"][y][x] += 1
                        obs_info["unit"]["opp_energy"][y][x] += e
                        obs_info["unit"]["opp_enable_move"][y][x] = 1 if e >= unit_move_cost else 0
                        obs_info["unit"]["opp_enable_sap"][y][x] = 1 if e >= unit_sap_cost else 0
                
                ### Map Feature
                tyle_type = np.array(obs["map_features"]["tile_type"])
                tyle_type = tyle_type.transpose((1, 0))  # [x][y] -> [y][x]
                energy = np.array(obs["map_features"]["energy"])
                energy = energy.transpose((1, 0))  # [x][y] -> [y][x]
                visible_mask = (tyle_type!=-1)
                tyle_type = apply_map_symmetry(tyle_type, visible_mask)
                energy = apply_map_symmetry(energy, visible_mask)
                relic_nodes = obs["relic_nodes"]
                relic_nodes_mask = obs["relic_nodes_mask"]

                ## nebula speed check
                if tyle_type_memory is None:
                    tyle_type_memory = tyle_type

                # not fixed nebula speed
                if nebula_speed == 0:
                    both_visible_area = (tyle_type_memory[visible_mask]>=0) & (tyle_type[visible_mask]>=0)
                    if not np.array_equal(tyle_type_memory[visible_mask][both_visible_area], tyle_type[visible_mask][both_visible_area]):
                        vote = vote_speed(step)
                        direction = check_drift_direction(tyle_type_memory, tyle_type, visible_mask)

                        tyle_type_memory = shift_map(direction, tyle_type_memory)
                        nebula_speed_vote += vote
                        nebula_direction += direction
                # fixed nebula speed
                else:
                    tyle_type_memory = update_type_map(tyle_type_memory, nebula_direction, nebula_speed, step)
                tyle_type_memory[(tyle_type!=-1)] = tyle_type[(tyle_type!=-1)]


                # register dict
                obs_info["map"] = {}
                obs_info["map"]["tyle_type"] = tyle_type_memory
                obs_info["map"]["energy"] = energy
                obs_info["map"]["visible_mask"] = visible_mask
                for node, mask in zip(relic_nodes, relic_nodes_mask):
                    if mask==1:
                        x, y = node
                        relic_node_map[y][x] = 1

                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = x+dx, y+dy
                                if 0<=nx<map_width and 0<=ny<map_height:
                                    relic_node_surround_map[ny][nx] = 1
                        
                        # symmetory
                        sym_x = map_height-1-y
                        sym_y = map_width-1-x
                        relic_node_map[sym_y][sym_x] = 1

                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = sym_x+dx, sym_y+dy
                                if 0<=nx<map_width and 0<=ny<map_height:
                                    relic_node_surround_map[ny][nx] = 1

                obs_info["map"]["relic_nodes"] = relic_node_map
                obs_info["map"]["relic_node_surround"] = relic_node_surround_map
                
                ### Meta Data
                match_steps = obs["match_steps"]
                match_round = (step-1)//101
                self_team_point = obs["team_points"][self_id]
                opp_team_point = obs["team_points"][opp_id]
                self_team_win = obs["team_wins"][self_id]
                opp_team_win = obs["team_wins"][opp_id]
                self_reward = self_team_point - pre_self_point
                opp_reward = opp_team_point - pre_opp_point

                obs_info["meta"] = {}
                obs_info["meta"]["self_reward"] = (np.ones((map_height, map_width)).astype(np.int16) * self_reward)
                obs_info["meta"]["opp_reward"] = (np.ones((map_height, map_width)).astype(np.int16) * opp_reward)
                obs_info["meta"]["match_steps"] = (np.ones((map_height, map_width)).astype(np.int16) * match_steps)
                obs_info["meta"]["match_round"] = (np.ones((map_height, map_width)).astype(np.int16) * match_round)
                obs_info["meta"]["self_team_point"] = (np.ones((map_height, map_width)).astype(np.int16) * self_team_point)
                obs_info["meta"]["opp_team_point"] = (np.ones((map_height, map_width)).astype(np.int16) * opp_team_point)
                obs_info["meta"]["self_team_win"] = (np.ones((map_height, map_width)).astype(np.int16) * self_team_win)
                obs_info["meta"]["opp_team_win"] = (np.ones((map_height, map_width)).astype(np.int16) * opp_team_win)
                obs_info["meta"]["unit_move_cost"] = (np.ones((map_height, map_width)).astype(np.int16) * unit_move_cost)
                obs_info["meta"]["unit_sap_cost"] = (np.ones((map_height, map_width)).astype(np.int16) * unit_sap_cost)
                obs_info["meta"]["unit_sap_range"] = (np.ones((map_height, map_width)).astype(np.int16) * unit_sap_range)
                obs_info["meta"]["nebula_tile_drift_speed"] = (np.ones((map_height, map_width)).astype(np.int16) * nebula_speed * nebula_direction)
                obs_info["meta"]["unit_sap_dropoff_factor"] = (np.ones((map_height, map_width)).astype(np.int16) * unit_sap_dropoff_factor)

                pre_self_point = self_team_point
                pre_opp_point = opp_team_point

                ### Point Prob Map
                # 1. Update Probability according to reward
                find_all_relic = obs_info["map"]["relic_nodes"].sum() >= min(5, (match_round*2)+1)
                confirm_reward = point_confirm_map[obs_info["unit"]["self_unit_pos"]>0].sum().item()
                reward = self_reward - confirm_reward
                if reward > 0:
                    const = cfg["point_prob_configs"]["add_const"]
                else:
                    if match_steps>=50 or match_round > 2 or find_all_relic:
                        const = -np.inf
                    else:
                        const = cfg["point_prob_configs"]["sub_const"] / (50-match_steps)

                _memo_map = np.zeros((map_height, map_width)).astype(np.int8)
                for p, m in zip(self_unit_position, self_unit_mask):
                    if m==1:
                        x, y = p
                        if _memo_map[y][x] == 0 and point_confirm_map[y][x]==0:
                            point_prob_map[y][x] += const
                            _memo_map[y][x] += 1

                            # symmetory
                            sym_x = map_height-1-y
                            sym_y = map_width-1-x
                            point_prob_map[sym_y][sym_x] += const

                # 2. check confirm
                _mask = (obs_info["map"]["relic_node_surround"] > 0) & (point_confirm_map==0) & (point_prob_map!=-np.inf)
                if reward > 0 and _memo_map[_mask].sum() == reward:
                    point_confirm_map[(_memo_map==1) & _mask] = 1
                    sym_point_confirm_map = np.rot90(np.fliplr(point_confirm_map), -1)
                    point_confirm_map += sym_point_confirm_map
                    point_confirm_map = np.clip(point_confirm_map, 0, 1)
                point_prob_map[point_confirm_map==1] = np.inf

                # 3. check find all relic
                if find_all_relic:
                    point_prob_map[obs_info["map"]["relic_node_surround"]==0] = -np.inf
                
                obs_info["point"] = {}
                obs_info["point"]["prob_map"] = point_prob_map

                ### Packing Status Data
                states = []
                for target_state in cfg["target_states"]:
                    category, name = target_state.split(".")
                    states.append(obs_info[category][name])
                states = np.array(states)
                states = states.astype(np.float16)


                ###############################################################################
                ### Next Action
                next_step_data = data["steps"][step+1]
                next_self_data = next_step_data[self_id]
                next_action = next_self_data["action"]

                next_action_map = np.zeros((6, map_height, map_width))
                next_action_mask = np.zeros((1, map_height, map_width))
                next_sap_pos = np.zeros((2, map_height, map_width))    # (x, y)
                _memo_map = np.zeros((6, map_height, map_width)).astype(np.int8)
                for p, a, m in zip(self_unit_position, next_action, self_unit_mask):
                    if m==1:
                        x, y = p
                        action_idx = a[0]  if self_id==0 else convert_action_command(a[0])
                        next_action_map[action_idx, y, x] += 1
                        next_action_mask[0, y, x] = 1
                        _memo_map[:, y, x] += 1

                        # sap action
                        if action_idx == 5:
                            coef = 1.0 if self_id==0 else -1.0
                            next_sap_pos[0, y, x] = a[1] * coef
                            next_sap_pos[1, y, x] = a[2] * coef
                next_action_map = next_action_map / (_memo_map+1e-9)

                ### Packing Action Data
                actions = [
                    next_action_map,
                    next_sap_pos,
                    next_action_mask,
                ]
                actions = np.concatenate(actions)
                actions = actions.astype(np.float16)

                ###############################################################################
                ### Convert view point
                # if player1, convert to player0 view point
                if self_id == 1:
                    states = np.rot90(states, 2, axes=(1,2))
                    actions = np.rot90(actions, 2, axes=(1,2))
                
                ###############################################################################
                ### Write data list
                data_info["idx"].append(data_idx)
                data_info["sub_id"].append(sub_id)
                data_info["episode_id"].append(episode_id)
                data_info["steps"].append(step)
                data_info["match_round"].append(match_round)
                data_info["match_steps"].append(match_steps)

                ###############################################################################
                ### Register lmdb
                key_state = f"state:{data_idx}".encode("utf-8")
                key_action = f"action:{data_idx}".encode("utf-8")
                txn.put(key_state, blosc.compress(pickle.dumps(states), cname="lz4"))
                txn.put(key_action, blosc.compress(pickle.dumps(actions), cname="lz4"))
                data_idx += 1

env.close()
data_df = pd.DataFrame.from_dict(data_info)
print(data_df)
print(f"data_num: {len(data_df)}")
data_df.to_csv(os.path.join(output_dir, "data_list.csv"), index=False)