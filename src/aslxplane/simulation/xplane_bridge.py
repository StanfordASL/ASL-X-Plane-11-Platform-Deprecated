import sys
import os
import pandas as pd
import numpy as np
import xpc3
import xpc3_helper
import time

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

    def reset(self, params):
        """ reset simulator to given initial conditions and wait for a few seconds"""
        return

    def send_control(self, input):
        """ reformats and sends the given control action into hte simulator"""

    def get_observation(self):
        """ retrieves the current observation from the simulator, and applies a specified transformation """

    def sleep(self):
        """pauses execution until next timestep"""

    def get_ground_truth_state(self):
        """retrieves current ground truth state"""

    