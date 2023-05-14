import yaml
import numpy as np
import xpc3.xpc3 as xpc3
import tqdm
import wandb
import matplotlib.pyplot as plt
from aslxplane.simulation.xplane_bridge import XPlaneBridge
from aslxplane.control.xplanecontroller import XPlaneController, SinusoidController
from aslxplane.perception.estimators import TaxiNet, GroundTruthEstimator
from aslxplane.perception.monitors import AutoEncoderMonitor
from aslxplane.simulation.data_recorder import DataRecorder

import os
import sys

if len(sys.argv) < 2:
	experiment_dir = ""
else:
	experiment_dir = sys.argv[1]

experiment_params_file = os.path.join(experiment_dir, "params/experiment_params.yaml")
simulator_params_file = os.path.join(experiment_dir, "params/simulator_params.yaml")

with open(simulator_params_file) as file:
	simulator_params = yaml.load(file, Loader=yaml.FullLoader)
with open(experiment_params_file) as file:
	experiment_params = yaml.load(file, Loader=yaml.FullLoader)
        
def sample_episode_params(experiment_params):
    episode_params = {}
    episode_params["start_time"] = np.random.uniform(*experiment_params["ood"]["experiment_time_range"])
    episode_params["weather"] = np.random.choice(experiment_params["weather_types"])
    episode_params["ood"] = {
        "corruption": np.random.choice(experiment_params["ood"]["corruption"]), 
        "transient_range": [
            np.random.uniform(*experiment_params["ood"]["transient_time_range"]["start_time_range"]),
            np.random.uniform(*experiment_params["ood"]["transient_time_range"]["end_time_range"])
        ]
    }
    episode_params["initial_position"] = {
        "he":np.random.uniform(*experiment_params["initial_position"]["he_range"]), 
        "cte":np.random.uniform(*experiment_params["initial_position"]["cte_range"])
    }
    return episode_params

def run_trajectory(xplane, controller, estimator, episode_params, monitor=None, data_recorder=None, debug=False):
    xplane.reset(episode_params)
    controller.reset()

    if data_recorder is not None:
        data_recorder.reset()
    
    while not xplane.episode_complete():
        observation = xplane.get_observation()
        ground_truth_state = xplane.get_ground_truth_state()

        if debug:
            plt.imshow(observation)
            plt.draw()
            plt.pause(0.001)

        if data_recorder is not None:
            data_recorder.record(xplane, ground_truth_state, observation)

        cte, he = estimator.get_estimate(observation)
        speed = ground_truth_state[-1]

        if monitor is not None:
            anomaly_score, estop = monitor.monitor(observation, [cte, he, speed])
        else:
            estop = False

        rudder, throttle = controller.get_input((cte, he, speed), estop=estop)

        xplane.send_control(rudder, throttle, estop=estop)
        xplane.sleep()
    
    observation = xplane.get_observation()
    ground_truth_state = xplane.get_ground_truth_state()
    if data_recorder is not None:
            data_recorder.record(xplane, ground_truth_state, observation)

    return xplane.is_failure(), observation

with xpc3.XPlaneConnect() as client:
    xplane = XPlaneBridge(client, simulator_params)

    if experiment_params["controller"]["type"] == "xplane_controller":
        controller = XPlaneController(
            experiment_params["controller"]["steering"],
            experiment_params["controller"]["speed"],
            simulator_params["simulator"]["time_step"]
        )
    elif experiment_params["controller"]["type"] == "sinusoid_controller":
        controller = SinusoidController(
            experiment_params["controller"]["steering"],
            experiment_params["controller"]["speed"],
            simulator_params["simulator"]["time_step"]
        )
    else: 
        raise(NotImplementedError)

    if experiment_params["state_estimation"]["estimator"] == "taxinet":
        estimator = TaxiNet(
            experiment_params["state_estimation"]["model_file"], 
            experiment_params["logging"]["normalization"]
        )
    else:
        estimator = GroundTruthEstimator(xplane)

    if experiment_params["runtime_monitor"]["monitor"] == "autoencoder":
        monitor = AutoEncoderMonitor(
            experiment_params["runtime_monitor"]["model_file"],
            experiment_params["runtime_monitor"]["threshold"]
        )
    else:
        monitor = None

    if experiment_params["logging"]["log_data"]:
        data_recorder = DataRecorder(experiment_params["logging"], experiment_dir)
    else:
        data_recorder = None

    if experiment_params["logging"]["use_wandb"]:
        data_recorder.init_wandb_logging(experiment_params_file, simulator_params_file)

    if experiment_params["debug"]["perception"]:
        plt.ion()
        plt.show()

    episode_iterator = tqdm.tqdm(range(experiment_params["num_episodes"]))
    for i in episode_iterator:
        episode_params = sample_episode_params(experiment_params)
        episode_iterator.set_description("Episode {}, Corruption Type = ".format(i) + episode_params["ood"]["corruption"])
        is_failure, last_observation = run_trajectory(
            xplane,
            controller,
            estimator,
            episode_params,
            monitor=monitor,
            data_recorder=data_recorder,
            debug=experiment_params["debug"]["perception"]
        )
        if experiment_params["logging"]["use_wandb"]:
            data_recorder.log_wandb(episode_params, is_failure, last_observation)

    xplane.reset(episode_params)

    if experiment_params["logging"]["use_wandb"]:
        data_recorder.finish_wandb()

