import sys
import os
import pandas as pd

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import numpy as np
import xpc3
import xpc3_helper
import time

import mss
import cv2

import settings_conformal as settings

def main():
    with xpc3.XPlaneConnect() as client:
        # Set weather and time of day
        #client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
        
        # client.sendDREF("sim/weather/rain_percent", 0)
        # Run the trajectories
        out_dir = settings.OUT_DIR
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(settings.NUM_TRAJECTORIES):
            runTrainingCase(client, settings)
            time.sleep(1)
    

def runTaxiNet(client, simSpeed = 1.0):
    """ Runs a sinusoidal trajectory down the runway

        Args:
            client: XPlane Client
            headingLimit: max degrees aircraft heading direction will deviate from 
                          runway direction (might go slightly past this)
            turn: dictates rudder/nosewheel strength (gain on the rudder/nosewheel command)
                  Larger value means tighter turns
            centerCTE: center of sinusoidal trajectory (in meters from centerline)
            -------------------
            simSpeed: increase beyond 1 to speed up simulation
            endPerc: percentage down runway to end trajectory and reset
    """

    # Reset to beginning of runway
    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client, dtpInit=settings.START_PERC / 100 * settings.RUNWAY_LENGTH)
    xpc3_helper.sendBrake(client, 0)

    getState = settings.GET_STATE
    getControl = settings.GET_CONTROL

    time.sleep(5) # 5 seconds to get terminal window out of the way
    client.pauseSim(False)

    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    endTime = startTime
    timeoutStart = startTime
    while xpc3_helper.getPercDownRunway(client) < settings.END_PERC and startTime - timeoutStart < 60.0:
        # getSinusoidalControl(client, headingLimit, turn, centerCTE)
        # time.sleep(0.03)
        speed = xpc3_helper.getSpeed(client)
        throttle = 0.5
        if speed > 20:
            throttle = 0.0
        elif speed < 6:
            throttle = 0.7
        # print(startTime)

        cte, he = getState(client)
        cte_true, _, he = xpc3_helper.getHomeState(client)
        print("true:", cte_true, "pred:", cte, "speed", speed)
        rudder = getControl(client, cte, he)
        client.sendCTRL([0, rudder, rudder, throttle])

        # Wait for next timestep
        while endTime - startTime < 1:
            endTime = client.getDREF("sim/time/zulu_time_sec")[0]
            time.sleep(0.001)

        # Set things for next round
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime
        _, dtp, _ = xpc3_helper.getHomeState(client)
        time.sleep(0.001)

    xpc3_helper.reset(client)


def runTrainingCase(client, settings):
    """ Main function to run the training cases. Samples simulator parameters and runs one training trajectory.

        Args:
            client: XPlane Client
            settings: simulation sampling settings
    """

    # Set time of day
    time_of_day = np.random.uniform(settings.TIME_OF_DAY_START, settings.TIME_OF_DAY_END)
    client.sendDREF("sim/time/zulu_time_sec", time_of_day * 3600 + 8 * 3600)
    # Set weather type
    weather = np.random.choice(list(settings.CLOUD_TYPES.keys()))
    client.sendDREF("sim/weather/cloud_type[0]", weather)

    runTaxiNet(client)


    

if __name__ == "__main__":
    main()
