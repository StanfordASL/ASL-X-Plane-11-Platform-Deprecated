import yaml
import pandas as pd
from aslxplane.utils.analysis_utils import get_episode_dict, animate_episode_with_traj

data_dir = "../Xplane-data-dir/xplane-asl-test/nominal_video/"
save_dir = "videos/"
label_file = "labels.csv"
real_time_x = 5

with open("params/simulator_params.yaml") as file:
	simulator_params = yaml.load(file, Loader=yaml.FullLoader)

df = pd.read_csv(data_dir + label_file)
episode_dict = get_episode_dict(df)

for episode_num in episode_dict.keys():
    animate_episode_with_traj(
        data_dir,
        save_dir,
        df,
        episode_dict,
        episode_num,
        simulator_params,
        real_time_x=real_time_x
    )
