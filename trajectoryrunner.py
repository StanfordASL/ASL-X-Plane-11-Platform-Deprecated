import yaml
import numpy as np
import xpc3.xpc3 as xpc3
import tqdm
from aslxplane.simulation.xplane_bridge import XPlaneBridge
from aslxplane.control.xplanecontroller import XPlaneController
from aslxplane.perception.estimators import TaxiNet, GroundTruthEstimator
from aslxplane.simulation.data_recorder import DataRecorder

with open("params/simulator_params.yaml") as file:
	simulator_params = yaml.load(file, Loader=yaml.FullLoader)
with open("params/experiment_params.yaml") as file:
	experiment_params = yaml.load(file, Loader=yaml.FullLoader)
        
def sample_episode_params(experiment_params):
      episode_params = {}
      episode_params["start_time"] = np.random.uniform(*experiment_params["ood"]["experiment_time_range"])
      episode_params["weather"] = np.random.choice(experiment_params["weather_types"])
      episode_params["ood"] = {"corruption": np.random.choice(experiment_params["ood"]["corruption"])}
      return episode_params

def run_trajectory(xplane, controller, estimator, episode_params, data_recorder=None):
      print("resetting simulator")
      xplane.reset(episode_params)
      controller.reset()

      if data_recorder is not None:
            data_recorder.reset()

      while not xplane.episode_complete():
            observation = xplane.get_observation()
            ground_truth_state = xplane.get_ground_truth_state()

            if data_recorder is not None:
                  data_recorder.record(xplane, ground_truth_state, observation)

            cte, he = estimator.get_estimate(observation)
            speed = ground_truth_state[-1]

            rudder, throttle = controller.get_input((cte, he, speed))

            xplane.send_control(rudder, throttle)
            xplane.sleep()

with xpc3.XPlaneConnect() as client:
      xplane = XPlaneBridge(client, simulator_params)
      controller = XPlaneController(
            experiment_params["controller"]["steering"],
            experiment_params["controller"]["speed"],
            simulator_params["simulator"]["time_step"]
      )

      if experiment_params["state_estimation"]["estimator"] == "taxinet":
            estimator = TaxiNet(
                  experiment_params["state_estimation"]["model_file"], 
                  experiment_params["logging"]["normalization"]
            )
      else:
            estimator = GroundTruthEstimator(xplane)

      if experiment_params["logging"]["log_data"]:
            data_recorder = DataRecorder(experiment_params["logging"])
      else:
            data_recorder = None

      for i in range(experiment_params["num_episodes"]):
            episode_params = sample_episode_params(experiment_params)
            run_trajectory(
                  xplane,
                  controller,
                  estimator,
                  episode_params,
                  data_recorder=data_recorder
            )

