import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

nebula_speed_value_list = [0.025, 0.05, 0.1, 0.15]  # 40, 20, 10, 6~7 turn
def generate_sequence(limit):
    sequence = [15]
    increments = [6, 7, 7]
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


def preprocess(target_state_data, preprocess_cfg, state_name, obs_info):
    if state_name in preprocess_cfg:
        # mask
        if "mask" in preprocess_cfg[state_name]:
            mask_state_name, cond, replace_num = preprocess_cfg[state_name]["mask"]
            mask_category, mask_name = mask_state_name.split(".")
            mask = (obs_info[mask_category][mask_name]==cond)
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

def revert_rot_action(action, k):
    for _ in range(k):
        action = np.rot90(action, 1, axes=(1,2))
        new_action = action.copy()
        new_action[1, :, :] = action[2, :, :]    # up -> right
        new_action[2, :, :] = action[3, :, :]    # right -> down
        new_action[3, :, :] = action[4, :, :]    # down -> left
        new_action[4, :, :] = action[1, :, :]    # left -> up
        action = new_action
    return action

def apply_map_symmetry(state_map, visible_mask):
    sym_mask = np.rot90(np.flip(state_map, axis=1), k=-1)
    return np.where(visible_mask, state_map, sym_mask)

def sample_with_cumulative_threshold(probs, thresh):
    if probs.sum()==0:
        return 0

    # 確率の降順でインデックスを取得
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # 累積確率を計算
    cumulative_probs = np.cumsum(sorted_probs)

    # 累積確率が閾値を超えた時点までのインデックスを取得
    valid_count = np.searchsorted(cumulative_probs, thresh, side='right') + 1
    valid_indices = sorted_indices[:valid_count]  # 元のインデックスで取得
    valid_probs = sorted_probs[:valid_count]  # 対応する確率

    # 確率を正規化
    valid_probs /= valid_probs.sum()

    # 確率に基づいてサンプリング
    return int(np.random.choice(valid_indices, p=valid_probs))


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        working_folder = Path(__file__).parent

        with open(os.path.join(working_folder, "config.yaml"), "r") as f:
            self.cfg = yaml.safe_load(f)
        
        with open(os.path.join(working_folder, "dataset_config.yaml"), "r") as f:
            self.dataset_cfg = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(os.path.join(working_folder, "jit_model.pt"), map_location=self.device)
        self.sap_model = torch.jit.load(os.path.join(working_folder, "jit_sap_model.pt"), map_location=self.device)
        self.sap_dropoff_model_path = os.path.join(working_folder, "jit_sap_dropoff_model.pt")
        self.sap_dropoff_action_model_path = os.path.join(working_folder, "jit_sap_dropoff_action_model.pt")

        self.do_tta = True
        self.do_sap_tta = True

        ### prob thresh
        self.action_prob_thresh = 0.7
        self.sap_prob_thresh = 0.6
        
        ### env_cfg
        map_height = self.env_cfg["map_height"]
        map_width = self.env_cfg["map_width"]

        ### Initialize
        self.relic_node_map = np.zeros((map_height, map_width)).astype(np.int16)
        self.relic_node_surround_map = np.zeros((map_height, map_width)).astype(np.int16)
        self.find_all_relic = False
        self.point_prob_map = np.zeros((map_height, map_width))
        self.point_confirm_map = np.zeros((map_height, map_width))
        self.pre_self_point = 0
        self.pre_opp_point = 0
        self.pre_obs_info = None

        self.nebula_speed_vote = np.array([0, 0, 0, 0])     # [0.025, 0.05, 0.1, 0.15]
        self.nebula_direction = 0
        self.nebula_speed = 0
        self.tyle_type_memory = None

        self.pre_sap_targets = []
        self.pre_raw_obs = None
        self.sap_dropoff = -1
        self.confirmed_sap_dropoff = False


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        ########## Update & Gather Information ##########
        if step == 0:
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        if (step-1)%101 == 0:
            # Reset
            if (step-1)//101 < 3:
                self.point_prob_map[self.point_prob_map<0] = 0.0
            self.find_all_relic = False
            self.pre_self_point = 0
            self.pre_opp_point = 0
            self.pre_obs_info = None

            # nebula confirm check
            if self.nebula_direction != 0:
                self.nebula_direction = 1 if self.nebula_direction>0 else -1
                self.nebula_speed = nebula_speed_value_list[self.nebula_speed_vote.argmax()]
        
        obs_info = {}
        ### env_cfg
        map_height = self.env_cfg["map_height"]
        map_width = self.env_cfg["map_width"]
        unit_move_cost = self.env_cfg["unit_move_cost"]
        unit_sap_cost = self.env_cfg["unit_sap_cost"]
        unit_sap_range = self.env_cfg["unit_sap_range"]

        ### Unit
        self_unit_position = obs["units"]["position"][self.team_id]
        opp_unit_position = obs["units"]["position"][self.opp_team_id]
        self_energy = obs["units"]["energy"][self.team_id]
        opp_energy = obs["units"]["energy"][self.opp_team_id]
        self_unit_mask = obs["units_mask"][self.team_id]
        opp_unit_mask = obs["units_mask"][self.opp_team_id]

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
        if self.tyle_type_memory is None:
            self.tyle_type_memory = tyle_type
        
        # not fixed nebula speed
        if self.nebula_speed == 0:
            both_visible_area = (self.tyle_type_memory[visible_mask]>=0) & (tyle_type[visible_mask]>=0)
            if not np.array_equal(self.tyle_type_memory[visible_mask][both_visible_area], tyle_type[visible_mask][both_visible_area]):
                vote = vote_speed(step)
                direction = check_drift_direction(self.tyle_type_memory, tyle_type, visible_mask)

                self.tyle_type_memory = shift_map(direction, self.tyle_type_memory)
                self.nebula_speed_vote += vote
                self.nebula_direction += direction
        # fixed nebula speed
        else:
            self.tyle_type_memory = update_type_map(self.tyle_type_memory, self.nebula_direction, self.nebula_speed, step)
        self.tyle_type_memory[(tyle_type!=-1)] = tyle_type[(tyle_type!=-1)]


        obs_info["map"] = {}
        obs_info["map"]["tyle_type"] = self.tyle_type_memory
        obs_info["map"]["energy"] = energy
        obs_info["map"]["visible_mask"] = visible_mask
        for node, mask in zip(relic_nodes, relic_nodes_mask):
            if mask==1:
                x, y = node
                self.relic_node_map[y][x] = 1

                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x+dx, y+dy
                        if 0<=nx<map_width and 0<=ny<map_height:
                            self.relic_node_surround_map[ny][nx] = 1
                
                # symmetory
                sym_x = map_height-1-y
                sym_y = map_width-1-x
                self.relic_node_map[sym_y][sym_x] = 1

                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = sym_x+dx, sym_y+dy
                        if 0<=nx<map_width and 0<=ny<map_height:
                            self.relic_node_surround_map[ny][nx] = 1

        obs_info["map"]["relic_nodes"] = self.relic_node_map
        obs_info["map"]["relic_node_surround"] = self.relic_node_surround_map
        
        ### Meta Data
        match_steps = obs["match_steps"]
        match_round = (step-1)//101
        self_team_point = obs["team_points"][self.team_id]
        opp_team_point = obs["team_points"][self.opp_team_id]
        self_team_win = obs["team_wins"][self.team_id]
        opp_team_win = obs["team_wins"][self.opp_team_id]
        self_reward = self_team_point - self.pre_self_point
        opp_reward = opp_team_point - self.pre_opp_point

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

        self.pre_self_point = self_team_point
        self.pre_opp_point = opp_team_point


        ### Point Prob Map
        # 1. Update Probability according to reward
        self.find_all_relic = obs_info["map"]["relic_nodes"].sum() >= min(5, (match_round*2)+1)
        confirm_reward = self.point_confirm_map[obs_info["unit"]["self_unit_pos"]>0].sum().item()
        reward = self_reward - confirm_reward
        if reward > 0:
            const = self.dataset_cfg["point_prob_configs"]["add_const"]
        else:
            if match_steps>=50 or match_round > 2 or self.find_all_relic:
                const = -np.inf
            else:
                const = self.dataset_cfg["point_prob_configs"]["sub_const"] / (50-match_steps)

        _memo_map = np.zeros((map_height, map_width)).astype(np.int8)
        for p, m in zip(self_unit_position, self_unit_mask):
            if m==1:
                x, y = p
                if _memo_map[y][x] == 0 and self.point_confirm_map[y][x]==0:
                    self.point_prob_map[y][x] += const
                    _memo_map[y][x] += 1

                    # symmetory
                    sym_x = map_height-1-y
                    sym_y = map_width-1-x
                    self.point_prob_map[sym_y][sym_x] += const

        # 2. check confirm
        _mask = (obs_info["map"]["relic_node_surround"] > 0) & (self.point_confirm_map==0) & (self.point_prob_map!=-np.inf)
        if reward > 0 and _memo_map[_mask].sum() == reward:
            self.point_confirm_map[(_memo_map==1) & _mask] = 1
            sym_point_confirm_map = np.rot90(np.fliplr(self.point_confirm_map), -1)
            self.point_confirm_map += sym_point_confirm_map
            self.point_confirm_map = np.clip(self.point_confirm_map, 0, 1)
        self.point_prob_map[self.point_confirm_map==1] = np.inf

        # 3. check find all relic
        if self.find_all_relic:
            self.point_prob_map[obs_info["map"]["relic_node_surround"]==0] = -np.inf
        
        obs_info["point"] = {}
        obs_info["point"]["prob_map"] = self.point_prob_map


        ### Chcek sap_data_list_with_sap_dropoff
        if len(self.pre_sap_targets) > 0 and self.pre_raw_obs is not None and self.sap_dropoff<0:
            for pre_sap_target in self.pre_sap_targets:
                x, y = pre_sap_target
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx==0 and dy==0:
                            continue
                        if 0<=x+dx<map_width and 0<=y+dy<map_height:
                            if obs_info["unit"]["opp_unit_pos"][y+dy][x+dx] > 0:
                                target_opp_x = x+dx
                                target_opp_y = y+dy

                                adjacent_self_unit_flag = False
                                for _dx, _dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    if 0<=target_opp_x+_dx<map_width and 0<=target_opp_y+_dy<map_height and obs_info["unit"]["self_unit_pos"][target_opp_y+_dy][target_opp_x+_dx]>0:
                                        adjacent_self_unit_flag = True
                                if adjacent_self_unit_flag:
                                    continue

                                target_idx = -1
                                for _i, pos in enumerate(opp_unit_position):
                                    if pos[0]==target_opp_x and pos[1]==target_opp_y:
                                        target_idx = _i
                                        break
                                
                                pre_opp_unit_mask = self.pre_raw_obs["units_mask"][self.opp_team_id]
                                if pre_opp_unit_mask[target_idx]:
                                    pre_target_opp_pos = self.pre_raw_obs["units"]["position"][self.opp_team_id][target_idx]
                                    cur_target_opp_energy = obs["units"]["energy"][self.opp_team_id][target_idx]
                                    cur_tyle_energy = obs_info["map"]["energy"][target_opp_y][target_opp_x]
                                    pre_target_opp_energy = self.pre_raw_obs["units"]["energy"][self.opp_team_id][target_idx]

                                    if pre_target_opp_pos[0]==target_opp_x and pre_target_opp_pos[1]==target_opp_y:
                                        move_cost = 0.0
                                    else:
                                        move_cost = unit_move_cost
                                    
                                    energy_diff = (pre_target_opp_energy-move_cost+cur_tyle_energy)-cur_target_opp_energy
                                    if int(np.floor(unit_sap_cost*1.0)) == energy_diff:
                                        self.sap_dropoff = 1.0
                                    elif int(np.floor(unit_sap_cost*0.5)) == energy_diff:
                                        self.sap_dropoff = 0.5
                                    elif int(np.floor(unit_sap_cost*0.25)) == energy_diff:
                                        self.sap_dropoff = 0.25
        
        if not self.confirmed_sap_dropoff and self.sap_dropoff > 0:
            # print(f"(step:{step}) sap_dropoff: {self.sap_dropoff}")
            self.confirmed_sap_dropoff = True
            self.model = torch.jit.load(self.sap_dropoff_action_model_path, map_location=self.device)
            self.sap_model = torch.jit.load(self.sap_dropoff_model_path, map_location=self.device)
            # print("change model")
        
        ### Update pre raw obs
        self.pre_raw_obs = obs


        ########## Create Model Input ##########
        using_states = self.cfg["dataset"]["using_states"]
        using_pre_states = self.cfg["dataset"]["using_pre_states"]
        using_meta_states = self.cfg["dataset"]["using_meta_states"]
        preprocess_cfg = self.cfg["dataset"]["preprocess_cfg"]

        ### create state map
        if self.pre_obs_info is None:
            self.pre_obs_info = obs_info

        state_map = []
        for state_name in using_states:
            category, name = state_name.split(".")
            target_state_data = obs_info[category][name].astype(np.float16)

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, obs_info)

            state_map.append(target_state_data)
        state_map = np.array(state_map).astype(np.float32)

        pre_frame_states = []
        for state_name in using_pre_states:
            category, name = state_name.split(".")
            target_state_data = self.pre_obs_info[category][name].astype(np.float16)

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, self.pre_obs_info)

            pre_frame_states.append(target_state_data)
        state_map_with_pre_frame = np.concatenate([state_map, np.array(pre_frame_states)], axis=0).astype(np.float32)

        # create global feature
        global_feature = []
        for state_name in using_meta_states:
            category, name = state_name.split(".")
            target_state_data = obs_info[category][name].astype(np.float16)

            # preprocess
            target_state_data = preprocess(target_state_data, preprocess_cfg, state_name, obs_info)

            global_feature.append(target_state_data[0][0].item())
        
        if self.confirmed_sap_dropoff:
            global_feature.append(self.sap_dropoff)

        global_feature = np.array(global_feature).astype(np.float32)
        

        ########## Predict Action ##########
        if self.do_tta:
            state_maps = torch.tensor([
                state_map_with_pre_frame,
                np.rot90(state_map_with_pre_frame, -1, axes=(1,2)),
                np.rot90(state_map_with_pre_frame, -2, axes=(1,2)),
                np.rot90(state_map_with_pre_frame, -3, axes=(1,2)),
            ])
            global_features = torch.tensor([
                global_feature,
                global_feature,
                global_feature,
                global_feature
            ])
        else:
            state_maps = torch.tensor(state_map_with_pre_frame).unsqueeze(0)
            global_features = torch.tensor(global_feature).unsqueeze(0)
        state_maps = state_maps.to(self.device)
        global_features = global_features.to(self.device)

        with torch.no_grad():
            outputs = self.model(state_maps, global_features)
        
        if self.do_tta:
            with torch.no_grad():
                _flip_outputs = self.model(torch.flip(state_maps, dims=[2]), global_features)
            _flip_outputs = torch.flip(_flip_outputs, dims=[2])
            flip_outputs = _flip_outputs.clone()
            flip_outputs[:, 3, :, :] = _flip_outputs[:, 1, :, :]
            flip_outputs[:, 1, :, :] = _flip_outputs[:, 3, :, :]
            outputs = (outputs+flip_outputs)/2

            outputs = outputs.cpu().numpy()
            ave_outputs = outputs[0]
            for k in range(1, 4):
                revert_action = revert_rot_action(outputs[k], k)
                ave_outputs += revert_action
            ave_outputs = ave_outputs / 4
            prob_map = F.softmax(torch.tensor(ave_outputs), dim=0).cpu().numpy()
        else:
            prob_map = F.softmax(outputs[0], dim=0).cpu().numpy()
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self_unit_position = obs["units"]["position"][self.team_id]
        self_unit_mask = obs["units_mask"][self.team_id]
        self_unit_energy = obs["units"]["energy"][self.team_id]
        min_energy_order = self_unit_energy.argsort()
        sap_unit_ids = []
        target_sap_areas = []
        # for i, (pos, mask, energy) in enumerate(zip(self_unit_position, self_unit_mask, self_unit_energy)):
        for i in min_energy_order:
            pos = self_unit_position[i]
            mask = self_unit_mask[i]
            energy = self_unit_energy[i]
            if mask:
                x, y = pos
                probs = prob_map[:, y, x].copy()

                if energy-unit_move_cost < 0:
                    probs[1:5] = 0.0
                if energy-unit_sap_cost < 0:
                    probs[5] = 0.0
                
                probs = probs / (probs+1e-9).sum()
                action = sample_with_cumulative_threshold(probs, self.action_prob_thresh)
                dx, dy = 0, 0

                if obs_info["unit"]["self_unit_pos"][y][x] > 1:
                    unit_num = obs_info["unit"]["self_unit_pos"][y][x]
                    updated_action_prob = max(0.0, probs[action]-1.0/unit_num)
                    prob_map[:, y, x][action] = updated_action_prob

                # sap action
                if action==5:
                    sap_unit_ids.append(i)

                    target_sap_area = np.zeros((24, 24))
                    for dy in range(-1*unit_sap_range, unit_sap_range+1):
                        for dx in range(-1*unit_sap_range, unit_sap_range+1):
                            ny = y + dy
                            nx = x + dx
                            if ny<0 or ny>=24 or nx<0 or nx>=24:
                                continue
                            target_sap_area[ny][nx] = 1.0
                    target_sap_areas.append(target_sap_area)
                actions[i] = [action, 0, 0]
        
        # Estimate Sap Target
        self.pre_sap_targets = []
        if len(sap_unit_ids) > 0:
            sap_state_maps = []
            sap_global_features = []
            for sap_unit_id, target_sap_area in zip(sap_unit_ids, target_sap_areas):
                other_sap_area = np.zeros((24, 24))
                for _sap_unit_id, _target_sap_area in zip(sap_unit_ids, target_sap_areas):
                    if sap_unit_id == _sap_unit_id:
                        continue
                    other_sap_area += _target_sap_area
                
                target_sap_area = target_sap_area[np.newaxis, :, :]
                other_sap_area = other_sap_area[np.newaxis, :, :]
                sap_state_map = np.concatenate([state_map, target_sap_area, other_sap_area], axis=0).astype(np.float32)
                sap_state_maps.append(sap_state_map)
                sap_global_features.append(global_feature)
            
            sap_state_maps = torch.tensor(sap_state_maps).to(self.device)
            sap_global_features = torch.tensor(sap_global_features).to(self.device)
            with torch.no_grad():
                sap_outputs = self.sap_model(sap_state_maps, sap_global_features)
            
            if self.do_sap_tta:
                with torch.no_grad():
                    flip_sap_outputs = self.sap_model(torch.flip(sap_state_maps, dims=[2]), sap_global_features)
                flip_sap_outputs = torch.flip(flip_sap_outputs, dims=[2])
                sap_outputs = (sap_outputs+flip_sap_outputs)/2

                for k in range(1, 4):
                    rot_sap_state_maps = torch.rot90(sap_state_maps, k=-1*k, dims=(2, 3))
                    with torch.no_grad():
                        rot_sap_outputs = self.sap_model(rot_sap_state_maps, sap_global_features)
                    sap_outputs += torch.rot90(rot_sap_outputs, k=k, dims=(2, 3))
                sap_outputs = sap_outputs/4
            
            for i in range(len(sap_outputs)):
                sap_output = sap_outputs[i][0].cpu()
                sap_unit_id = sap_unit_ids[i]
                mask = target_sap_areas[i]
                ys, xs = np.where(mask>0)
                sap_pred = sap_output[ys.min():ys.max()+1, xs.min():xs.max()+1]
                _h, _w = sap_pred.shape
                sap_prob = F.softmax(sap_pred.reshape(-1), dim=0).reshape(_h, _w).numpy()
                
                flat_probs = sap_prob.ravel()
                sampled_index = sample_with_cumulative_threshold(flat_probs, self.sap_prob_thresh)
                # sampled_index = np.random.choice(len(flat_probs), p=flat_probs)
                _x, _y = np.unravel_index(sampled_index, sap_prob.shape)[::-1]
                sap_target_x = _x + xs.min()
                sap_target_y = _y + ys.min()
                x, y = self_unit_position[sap_unit_id]
                dx = sap_target_x - x
                dy = sap_target_y - y
                actions[sap_unit_id][1] = dx
                actions[sap_unit_id][2] = dy

                self.pre_sap_targets.append([sap_target_x, sap_target_y])

        ### Update pre obs info
        self.pre_obs_info = obs_info

        return actions
