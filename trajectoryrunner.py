import yaml
import numpy as np
import xpc3.xpc3 as xpc3
from aslxplane.simulation.xplane_bridge import XPlaneBridge
from aslxplane.control.xplanecontroller import XPlaneController
from aslxplane.perception.estimators import TaxiNet

with open("params/simulator_params.yaml") as file:
	simulator_params = yaml.load(file, Loader=yaml.FullLoader)
with open("params/experiment_params.yaml") as file:
	experiment_params = yaml.load(file, Loader=yaml.FullLoader)
        
def sample_episode_params(experiment_params):
      episode_params = {}
      episode_params["start_time"] = np.random.uniform(*experiment_params["ood"]["experiment_time_range"])
      episode_params["weather"] = np.random.choice(experiment_params["weather_types"])
      episode_params["ood"] = {"corruption": experiment_params["ood"]["corruption"]}
      return episode_params

def run_trajectory(xplane, controller, estimator, episode_params):
    print("resetting simulator")
    xplane.reset(episode_params)
    controller.reset()

    while not xplane.episode_complete():
        observation = xplane.get_observation()
        cte, he = estimator.get_estimate(observation)
        ground_truth_state = xplane.get_ground_truth_state()
        speed = ground_truth_state[-1]
        he = ground_truth_state[1]
        rudder, throttle = controller.get_input((cte, he, speed))
        xplane.send_control(rudder, throttle)
        xplane.sleep()
        print("cte", cte, "he",he, "speed",speed)

with xpc3.XPlaneConnect() as client:
      xplane = XPlaneBridge(client, simulator_params)
      controller = XPlaneController(
            experiment_params["controller"]["steering"],
            experiment_params["controller"]["speed"],
            simulator_params["simulator"]["time_step"]
      )
      estimator = TaxiNet(experiment_params["state_estimation"]["model_file"])
      episode_params = sample_episode_params(experiment_params)

      run_trajectory(
            xplane,
            controller,
            estimator,
            episode_params
      )

