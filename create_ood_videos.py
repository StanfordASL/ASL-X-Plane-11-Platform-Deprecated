import yaml
import pandas as pd
from aslxplane.utils.analysis_utils import get_episode_dict, animate_episode_with_ood
import aslxplane.perception.models as xplanemodels
import torch.nn as nn
import torchvision.transforms as tform
import torch
from aslxplane.perception.estimators import TaxiNet
from aslxplane.perception.monitors import AutoEncoderMonitor


data_dir = "../Xplane-data-dir/xplane-asl-test/snowy_video/"
save_dir = "videos/"
label_file = "labels.csv"
ood_detector_file = "data/models/taxinet-autoencoder03-21_14-21.pt"
real_time_x = 5

with open(data_dir + "params/simulator_params.yaml") as file:
	simulator_params = yaml.load(file, Loader=yaml.FullLoader)
with open(data_dir + "params/experiment_params.yaml") as file:
    experiment_params = yaml.load(file, Loader=yaml.FullLoader)

df = pd.read_csv(data_dir + label_file)
episode_dict = get_episode_dict(df)

monitor = AutoEncoderMonitor(
    experiment_params["runtime_monitor"]["model_file"],
    experiment_params["runtime_monitor"]["threshold"]
)

perception_model = TaxiNet(
    experiment_params["state_estimation"]["model_file"],
    experiment_params["logging"]["normalization"]
)

triggers_fallback = experiment_params["runtime_monitor"]["monitor"] != "None"

for episode_num in episode_dict.keys():
    animate_episode_with_ood(
        data_dir,
        save_dir,
        df,
        episode_dict,
        episode_num,
        simulator_params,
        experiment_params,
        perception_model,
        monitor,
        real_time_x=real_time_x,
        triggers_fallback=triggers_fallback
    )
