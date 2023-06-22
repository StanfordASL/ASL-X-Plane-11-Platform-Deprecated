import sys
from pathlib import Path
import random
from argparse import ArgumentParser
import json

import numpy as np
from tqdm import tqdm
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.flaml import BlendSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import RunConfig

import xpc

try:
    from . import utils
    from .utils import XPlaneConnectWrapper, FlightState, VideoRecorder
    from .utils import deg2rad, rad2deg, reset_flight
    from .controller import FlightController, DEFAULT_COST_CONFIG
except ImportError:
    root_path = Path(__file__).parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    from aslxplane.flight_control import utils
    from aslxplane.flight_control.utils import XPlaneConnectWrapper, FlightState, VideoRecorder
    from aslxplane.flight_control.utils import deg2rad, rad2deg, reset_flight
    from aslxplane.flight_control.controller import FlightController, DEFAULT_COST_CONFIG


def main():
    # offsets_to_test = [(0.0, 0.0), (-2e3, 0.0), (2e3, 0.0)]
    offsets_to_test = [(0.0, 0.0)]

    ################################################################################################

    def cost_fn(x_hist, target, approach_ang):
        x_hist, target = np.array(x_hist), np.array(target)[:2]
        dx = target[:2] - x_hist[:, :2]
        min_approach_idx = np.argmin(np.linalg.norm(dx, axis=-1))
        approach_alt = x_hist[min_approach_idx, 2]
        if approach_alt > 25.0:
            return 1e9
        v_norm = np.array([np.cos(approach_ang), np.sin(approach_ang)])
        v_par = np.sum(dx * v_norm[None, :], -1)[:, None] * v_norm[None, :]
        v_perp = dx - v_par
        return np.linalg.norm(v_perp, axis=-1)[min_approach_idx]

    def run_trial(cost_config, sim_speed=3.0, record_video=False, view=None, abort_at=1e9):
        reset_flight(XPlaneConnectWrapper(), on_crash_only=False)
        controller = FlightController(config={"sim_speed": sim_speed}, verbose=False, view=view)
        hist_list, cost_list = [], []
        recorder = None
        for offset in tqdm(offsets_to_test):
            controller.cost_config.update(cost_config)
            controller.config.update({"x0_offset": offset[0], "y0_offset": offset[1]})
            if record_video:
                recorder = VideoRecorder(
                    f"recording_{random.randint(0, int(1e6)):05d}",
                    controller.flight_state,
                    2,
                    display=True,
                )
            crashed = controller.loop(abort_at=abort_at)
            controller.done = False
            if crashed:
                if recorder is not None:
                    recorder.close()
                controller.close()
                return dict(objective=1e9)
            hist = {
                "x": [x.tolist() for x in controller.x_hist],
                "u": [z.tolist() for z in controller.u_hist],
                "t": controller.t_hist,
            }
            cost = cost_fn(
                hist["x"], target=controller.target, approach_ang=controller.approach_ang
            )
            cost_list.append(cost)
            print(f"cost = {cost:.4e}")
            hist_list.append(dict(offset=offset, hist=hist))
            if cost > 1e8:
                break
        objective = np.max(cost_list)
        if recorder is not None:
            recorder.close()
        controller.close()
        return dict(
            objective=objective, cost_list=json.dumps(cost_list), hist_list=json.dumps(hist_list)
        )

    parser = ArgumentParser()
    parser.add_argument("--experiment", type=int, default=5)
    args = parser.parse_args()

    # experiment 4 #################################################################################
    if args.experiment == 5:
        cost_config = {}
        run_trial(
            cost_config,
            sim_speed=1.0,
            record_video=True,
            view=xpc.ViewType.FullscreenNoHud,
            abort_at=97,
        )

    elif args.experiment == 4:
        log_scale = lambda x: tune.loguniform(x * 1e-1, x * 1e1)
        configs = dict(
            perp_cost=481.605499,
            perp_quad_cost=log_scale(0.001356),
            par_cost=log_scale(70.124013),
            # perp_quad_cost=0.001356,
            # par_cost=70.124013,
            heading_cost=1e4,
        )
        # run_trial(config)
        algo = ConcurrencyLimiter(BlendSearch(), max_concurrent=1)
        tuner = tune.Tuner(
            run_trial,
            tune_config=tune.TuneConfig(
                metric="objective", mode="min", search_alg=algo, num_samples=500
            ),
            run_config=RunConfig(name="flight_tune_lqr_heading_1e4"),
            param_space=configs,
        )
        tuner.fit()

    # experiment 3 #################################################################################
    elif args.experiment == 3:
        config = dict(
            perp_cost=log_scale(481.605499),
            perp_quad_cost=0.001356,
            par_cost=70.124013,
            par_quad_cost=1e-9,
            heading_cost=1e4,
        )
        run_trial(config)

    # experiment 2 #################################################################################
    elif args.experiment == 2:
        perp_cost0, perp_quad_cost0, par_cost0, par_quad_cost0 = (
            481.605499,
            0.001356,
            70.124013,
            1e-3,
        )
        configs = {
            "perp_cost": perp_cost0,
            "par_cost": par_cost0,
            # "perp_cost": tune.loguniform(perp_cost0 * 1e-1, perp_cost0 * 1e1),
            "perp_quad_cost": tune.loguniform(perp_quad_cost0 * 1e-1, perp_quad_cost0 * 1e1),
            # "par_cost": tune.loguniform(par_cost0 * 1e-1, par_cost0 * 1e1),
            "par_quad_cost": tune.loguniform(par_quad_cost0 * 1e-4, par_quad_cost0 * 1e4),
        }

        algo = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=1)
        tuner = tune.Tuner(
            run_trial,
            tune_config=tune.TuneConfig(
                metric="objective", mode="min", search_alg=algo, num_samples=500
            ),
            run_config=RunConfig(name="flight_tune_lqr_with_par_quad_cost"),
            param_space=configs,
        )
        tuner.fit()

    # experiment 1##################################################################################
    elif args.experiment == 1:
        par_cost0 = DEFAULT_COST_CONFIG["par_cost"]
        perp_cost0 = DEFAULT_COST_CONFIG["perp_cost"]
        perp_quad_cost0 = DEFAULT_COST_CONFIG["perp_quad_cost"]

        configs = {
            "perp_cost": tune.loguniform(perp_cost0 * 1e-1, perp_cost0 * 1e1),
            "perp_quad_cost": tune.loguniform(perp_quad_cost0 * 1e-1, perp_quad_cost0 * 1e1),
            "par_cost": tune.loguniform(par_cost0 * 1e-1, par_cost0 * 1e1),
        }

        # all_configs = [x[1] for x in tune.search.variant_generator.generate_variants(configs)]
        # output = run_trial({"perp_cost": 1e3, "perp_quad_cost": 7e-4})

        algo = ConcurrencyLimiter(HyperOptSearch(), max_concurrent=1)
        tuner = tune.Tuner(
            run_trial,
            tune_config=tune.TuneConfig(
                metric="objective", mode="min", search_alg=algo, num_samples=500
            ),
            run_config=RunConfig(name="flight_tune_lqr2"),
            param_space=configs,
        )
        tuner.fit()


if __name__ == "__main__":
    main()
