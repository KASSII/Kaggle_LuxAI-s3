import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--lux_path", default="Lux-Design-S3/kits/python/lux")
parser.add_argument("--agent", default="src/agent.py")
parser.add_argument("--dataset_cfg_path", default="datas/db/luxs3_db/cfg.yaml")
parser.add_argument("--action_model_path", default="datas/output/IL/action_model/ft_wo_sapdropoff/fulldata/jit_model.pt")
parser.add_argument("--sap_target_model_path", default="datas/output/IL/sap_target_model/ft_wo_sapdropoff/fulldata/jit_model.pt")
parser.add_argument("--action_model_sapdropoff_path", default="datas/output/IL/action_model/ft_with_sapdropoff/fulldata/jit_model.pt")
parser.add_argument("--sap_target_model_sapdropoff_path", default="datas/output/IL/sap_target_model/ft_with_sapdropoff/fulldata/jit_model.pt")
parser.add_argument("--output_dir", default="datas/submit")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
shutil.copytree(args.lux_path, os.path.join(args.output_dir, "lux"), dirs_exist_ok=True)
shutil.copyfile(args.agent, os.path.join(args.output_dir, "agent.py"))
shutil.copyfile(args.dataset_cfg_path, os.path.join(args.output_dir, "dataset_config.yaml"))
shutil.copyfile(args.action_model_path, os.path.join(args.output_dir, "jit_model.pt"))
shutil.copyfile(args.sap_target_model_path, os.path.join(args.output_dir, "jit_sap_model.pt"))
shutil.copyfile(args.action_model_sapdropoff_path, os.path.join(args.output_dir, "jit_sap_dropoff_action_model.pt"))
shutil.copyfile(args.sap_target_model_sapdropoff_path, os.path.join(args.output_dir, "jit_sap_dropoff_model.pt"))
