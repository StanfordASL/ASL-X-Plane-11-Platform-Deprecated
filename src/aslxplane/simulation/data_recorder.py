import os
import pandas as pd
import numpy as np
import wandb
from copy import copy
from PIL import Image
import matplotlib.pyplot as plt


class DataRecorder:

    def __init__(self, recorder_params):
        self.params = recorder_params

        if not os.path.exists(self.params["output_dir"]):
            os.makedirs(self.params["output_dir"])
        
        self.output_dir = self.params["output_dir"]
        self.label_file = self.output_dir + "/labels.csv"

        self.curr_episode = -1
        self.curr_time_step = 0
        self.episode_start_time = None
        self.wandb_data = None

    def init_wandb_logging(self, experiment_params_file, simulator_params_file):
        wandb.init(
            # set the wandb project where this run will be logged
            project="asl-closed-loop-monitor",
            config={"yaml":experiment_params_file}
        )
        self.wandb_data = {
            "time_of_day" : [],
            "weather" : [],
            "corruption": [],
            "initial_cte": [],
            "initial_he": [],
            "failures": []
            }

    def log_wandb(self, episode_params, is_failure, observation):

        self.wandb_data["time_of_day"].append(episode_params["start_time"])
        self.wandb_data["weather"].append(episode_params["weather"])
        self.wandb_data["corruption"].append(episode_params["ood"]["corruption"])
        self.wandb_data["initial_cte"].append(episode_params["initial_position"]["cte"])
        self.wandb_data["initial_he"].append(episode_params["initial_position"]["he"])
        self.wandb_data["failures"].append(is_failure)
        
        log_data = {
            "time_of_day": wandb_histogram_bug_fixer(self.wandb_data, "time_of_day"),
            "weather": wandb_histogram_bug_fixer(self.wandb_data, "weather"),
            "initial_cte": wandb_histogram_bug_fixer(self.wandb_data, "initial_cte"),
            "initial_he": wandb_histogram_bug_fixer(self.wandb_data, "initial_he"),
            "failures": wandb_histogram_bug_fixer(self.wandb_data, "failures"),
            "corruptions": wandb_histogram_bug_fixer(self.wandb_data, "corruption")
        }
        log_data["observation"] = wandb.Image(observation)
        wandb.log(log_data)

    def finish_wandb(self):
        wandb.finish()

    def reset(self):
        """Resets datarecorder after an episode"""
        self.curr_episode += 1
        self.curr_time_step = 0
        self.episode_start_time = None

    def record(self, xplane, true_state, observation):
        """
        Adds a row of data to labels.csv and saves the current observation
        corresponding to the current timestep
        """
        absolute_time = xplane.get_zulu_time()
        if self.curr_time_step == 0:
            self.episode_start_time = absolute_time

        relative_time_seconds = xplane.get_zulu_time() - self.episode_start_time

        time_of_day, time_of_day_label = xplane.get_time_of_day()
        cloud_cover, cloud_cover_label = xplane.get_weather()

        cte, he, dtp, speed = true_state

        image_corruption, _ = xplane.get_corruption_status()

        img_filename = 'MWH_Runway04_' \
            + time_of_day_label + '_' \
            + cloud_cover_label + '_' \
            + str(self.curr_episode) + '_' \
            + str(self.curr_time_step) + '.png'

        row_data = {
            "image_filename": img_filename,
            "absolute_time_GMT_seconds": absolute_time,
            "relative_time_seconds": relative_time_seconds,
            "distance_to_centerline_meters": cte,
            "distance_to_centerline_NORMALIZED": cte / self.params["normalization"]["cte_constant"],
            "downtrack_position_meters": dtp,
            "downtrack_position_NORMALIZED": xplane.get_perc_down_runway(),
            "heading_error_degrees": he,
            "heading_error_NORMALIZED": he / self.params["normalization"]["he_constant"],
            "speed_meters_per_second": speed,
            "period_of_day": time_of_day,
            "cloud_type": cloud_cover,
            "image_corruption":image_corruption
        }

        if self.curr_episode == 0 and self.curr_time_step == 0:
            append_to_csv(self.label_file, row_data, header=True)
        else:
            append_to_csv(self.label_file, row_data, header=False)

        observation.save(self.output_dir + "/" + img_filename)
        
        self.curr_time_step += 1

def append_to_csv(output_file, row_dict, header=False):
    """
    appends a row of data to a csv file without loading the csv file into program memory
    Args:
        output_file (str): path to csv file
        row_dict (dict): dictionary with column:row_value items
        header (bool): boolean indicating whether the column names should be stored as well
    """
    row_df = pd.DataFrame([row_dict], columns=list(row_dict.keys()))
    row_df.to_csv(output_file, mode="a", header=header, index=False)

def wandb_histogram_bug_fixer(data, title):
    table = wandb.Table(
        data= [[pt] for pt in data[title]],
        columns=[title]
    )
    hist = wandb.plot.histogram(table, title, title=title)
    return hist


