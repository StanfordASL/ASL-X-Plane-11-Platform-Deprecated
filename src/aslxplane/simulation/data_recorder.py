import os
import pandas as pd


class DataRecorder:

    def __init__(self, recorder_params):
        self.params = recorder_params

        if not os.path.exists(self.recorder_params["output_dir"]):
            os.makedirs(self.recorder_params["output_dir"])

        self.curr_episode = 0
        self.curr_time_step = 0


    def reset(self, episode_params):
        """Resets datarecorder after an episode"""
        self.curr_episode += 1
        self.curr_time_step = 0

    def record(self):
        """
        Adds a row of data to labels.csv and saves the current observation
        corresponding to the current timestep
        """