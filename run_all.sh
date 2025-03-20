# 1. Data Prepare
# 1-1. make json list
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 42704976 --add_opp_team_name
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 42705163 --add_opp_team_name
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 43152191
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 43155694
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 43212163
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 43212846
python3 -m uv run src/data_prepare/make_json_list.py --sub_id 43276830

# 1-2. check out the episode with a low LB opponent
python3 -m uv run src/data_prepare/add_ignore.py --sub_id 42704976
python3 -m uv run src/data_prepare/add_ignore.py --sub_id 42705163

# 1-3. create DB for IL training
python3 -m uv run src/data_prepare/create_db.py --cfg_path src/data_prepare/db_cfg.yaml


# 2. IL
# 2-1. train action model (w/o sap_dropoff)
python3 -m uv run src/IL/action_model/train.py --cfg_path src/IL/action_model/config/config.yaml
python3 -m uv run src/IL/action_model/save_weight.py --cfg_path src/IL/action_model/config/config.yaml --ckpt_path datas/output/IL/action_model/basemodel_wo_sapdropoff/fold0/best-checkpoint.ckpt
python3 -m uv run src/IL/action_model/train.py --cfg_path src/IL/action_model/config/config_ft.yaml
python3 -m uv run src/IL/action_model/convert.py --cfg_path src/IL/action_model/config/config_ft.yaml --ckpt_path datas/output/IL/action_model/ft_wo_sapdropoff/fulldata/best-checkpoint.ckpt

# 2-2. train action model (with sap_dropoff)
python3 -m uv run src/IL/action_model/train.py --cfg_path src/IL/action_model/config/config_sapdropoff.yaml
python3 -m uv run src/IL/action_model/save_weight.py --cfg_path src/IL/action_model/config_sapdropoff/config.yaml --ckpt_path datas/output/IL/action_model/basemodel_with_sapdropoff/fold0/best-checkpoint.ckpt
python3 -m uv run src/IL/action_model/train.py --cfg_path src/IL/action_model/config/config_ft_sapdropoff.yaml
python3 -m uv run src/IL/action_model/convert.py --cfg_path src/IL/action_model/config/config_ft_sapdropoff.yaml --ckpt_path datas/output/IL/action_model/ft_with_sapdropoff/fulldata/best-checkpoint.ckpt

# 2-3. train sap target model (w/o sap_dropoff)
python3 -m uv run src/IL/sap_target_model/train.py --cfg_path src/IL/sap_target_model/config/config.yaml
python3 -m uv run src/IL/sap_target_model/save_weight.py --cfg_path src/IL/sap_target_model/config/config.yaml --ckpt_path datas/output/IL/sap_target_model/basemodel_wo_sapdropoff/fold0/best-checkpoint.ckpt
python3 -m uv run src/IL/sap_target_model/train.py --cfg_path src/IL/sap_target_model/config/config_ft.yaml
python3 -m uv run src/IL/sap_target_model/convert.py --cfg_path src/IL/sap_target_model/config/config_ft.yaml --ckpt_path datas/output/IL/sap_target_model/ft_wo_sapdropoff/fulldata/best-checkpoint.ckpt

# 2-4. train sap target model (with sap_dropoff)
python3 -m uv run src/IL/sap_target_model/train.py --cfg_path src/IL/sap_target_model/config/config_sapdropoff.yaml
python3 -m uv run src/IL/sap_target_model/save_weight.py --cfg_path src/IL/sap_target_model/config_sapdropoff/config.yaml --ckpt_path datas/output/IL/sap_target_model/basemodel_with_sapdropoff/fold0/best-checkpoint.ckpt
python3 -m uv run src/IL/sap_target_model/train.py --cfg_path src/IL/sap_target_model/config/config_ft_sapdropoff.yaml
python3 -m uv run src/IL/sap_target_model/convert.py --cfg_path src/IL/sap_target_model/config/config_ft_sapdropoff.yaml --ckpt_path datas/output/IL/sap_target_model/ft_with_sapdropoff/fulldata/best-checkpoint.ckpt


# 3. Create Submit Agent
python3 -m uv run src/create_sub.py
