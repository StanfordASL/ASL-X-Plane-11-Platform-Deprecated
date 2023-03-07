import xpc3
import time
import cv2
import mss
import pandas as pd
import numpy as np
import aslxplane.simulation.corruptions as corruptions
import xpc3.xpc3_helper as xpc3_helper
from PIL import Image

# bridge needs to do what?
# needs to
# 1. retrieve observations from the simulator
# 2. send given control actions to the simulator
# 3. pause execution until a timestep has passed in simulator time
# 4. send out ground truth state measurements
# 5. XXX assess whether a trajector is completed
# 6. reset the simulator with a given set of initial conditions
# 7. TODO: figure out a way to change simulator settings within an episode

class XPlaneBridge:

    def __init__(self, client, params):
        self.client = client
        self.params = params

        # screenshot camera setup
        self.screenshotter = mss.mss()
        self.monitor = self.screenshotter.monitors[self.params["screenshot_camera"]["monitor_id"]]
        padding = self.params["screenshot_camera"]["padding"]
        self.monitor["left"] += padding
        self.monitor["top"] += padding
        self.monitor["width"] -= padding
        self.monitor["height"] -= padding

        self.observation_corruption = None
        self.start_time = None

    def reset(self, episode_params):
        """ reset simulator to given initial conditions and wait for a few seconds"""
        xpc3_helper.reset(self.client)
        # set time of day
        self.client.sendDREF("sim/time/zulu_time_sec", episode_params["start_time"] * 3600 + 8 * 3600)
        # set weather conditions
        self.client.sendDREF("sim/weather/cloud_type[0]", episode_params["weather"])
        # set simspeed
        self.client.sendDREF("sim/time/sim_speed", self.params["simulator"]["sim_speed"])
        # reset to starting position on the runway
        xpc3_helper.reset(self.client, dtpInit=self.params["simulator"]["starting_position_pct"] / 100 * self.params["simulator"]["runway_length"])
        # other required reset steps
        xpc3_helper.sendBrake(self.client, 0)

        # Reset the image corruptions:
        if episode_params["ood"]["corruption"] == "None":
            self.observation_corruption = None
        elif episode_params["ood"]["corruption"] == "Noise":
            self.observation_corruption = corruptions.Noise()
        elif episode_params["ood"]["corruption"] == "Rain":
            self.observation_corruption = corruptions.Rain()
        elif episode_params["ood"]["corruption"] == "Motion Blur":
            self.observation_corruption = corruptions.Motionblur()
        elif episode_params["ood"]["corruption"] == "Rain and Motion Blur":
            self.observation_corruption = corruptions.RainyBlur()
        elif episode_params["ood"]["corruption"] == "Snow":
            self.observation_corruption = corruptions.Snow()

        time.sleep(self.params["simulator"]["episode_pause_time"])
        self.client.pauseSim(False)
        self.start_time = self.get_zulu_time()


    def send_control(self, rudder, throttle):
        """ reformats and sends the given control action into hte simulator"""
        self.client.sendCTRL([0, rudder, rudder, throttle])

    def get_observation(self):
        """ retrieves the current observation from the simulator, and applies a specified transformation """

        # Screenshot image, convert to RGB, then resize to specified observation size
        screenshot = self.screenshotter.grab(self.monitor)
        img = cv2.cvtColor(np.array(screenshot),
                       cv2.COLOR_BGRA2RGB)[self.params["screenshot_camera"]["toolbar_crop_pixels"]:, :, :]
        img = cv2.resize(img, (self.params["screenshot_camera"]["width"], self.params["screenshot_camera"]["height"]))

        # Apply the transformation to the image, convert to PIL Image
        if self.observation_corruption is not None:
            img = self.observation_corruption.add_corruption(img)
        observation = Image.fromarray(img)
        return observation

    def sleep(self):
        """pauses execution until next timestep in simulator time"""
        start_time = self.get_zulu_time()
        end_time = start_time
        while end_time - start_time < self.params["simulator"]["time_step"]:
            time.sleep(0.001)
            end_time = self.get_zulu_time()

    def get_zulu_time(self):
        return self.client.getDREF("sim/time/zulu_time_sec")[0]
    
    def get_local_time(self):
        # zulu_time = self.get_zulu_time()
        # time = pd.to_datetime(zulu_time - 8 * 3600, unit="s")
        # return time.hour + time.minute / 60
        local_time = self.client.getDREF("sim/time/local_time_sec")[0] / 3600
        return local_time
    
    def get_ground_truth_state(self):
        """retrieves current ground truth state from the simulator
        returns: 
            cte (float): current cross track error in meters
            he (float): current heading error angle in degrees
            dtp (float): current down-track position in meters
            speed (float): current velocity of the plane in meters/second
        """
        speed = xpc3_helper.getSpeed(self.client)
        cte, dtp, he = xpc3_helper.getHomeState(self.client)
        return cte, he, dtp, speed
    
    def episode_complete(self):
        perc_down_runway = xpc3_helper.getPercDownRunway(self.client)
        curr_time = self.get_zulu_time()
        return (perc_down_runway > self.params["simulator"]["ending_position_pct"] or
                 curr_time  - self.start_time > self.params["simulator"]["max_episode_time"])
