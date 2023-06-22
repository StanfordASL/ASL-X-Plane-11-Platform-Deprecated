import sys
import os
import time
from pathlib import Path
import json
import math
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Optional
from multiprocessing import Lock
from threading import Thread

os.environ["JAX_PLATFORM_NAME"] = "CPU"

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import xpc
import dashing as dsh

from pmpc import Problem
from pmpc.remote import solve_problems, RegisteredFunction
from pmpc.experimental.jax.root import all_sensitivity_L

try:
    from . import dynamics, utils
    from .lqr import design_LQR_controller
    from .utils import XPlaneConnectWrapper, deg2rad, rad2deg, FlightState, reset_flight
except ImportError:
    root_path = Path(__file__).parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    from aslxplane.flight_control import dynamics
    from aslxplane.flight_control import utils
    from aslxplane.flight_control.lqr import design_LQR_controller
    from aslxplane.flight_control.utils import XPlaneConnectWrapper, FlightState
    from aslxplane.flight_control.utils import deg2rad, rad2deg, reset_flight

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "CPU"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JFI_COPY_NUMPY"] = "0"

from jfi import jaxm

####################################################################################################


DEFAULT_COST_CONFIG = {
    #"heading_cost": 1e5,
    "heading_cost": 1e4,
    "roll_cost": 3e4,
    "position_cost": 1e0,
    "altitude_cost": 1e2,
    #"par_cost": 3e2,
    "par_cost": 498.863996,
    #"perp_cost": 1e3,
    "perp_cost": 481.605499,
    #"perp_quad_cost": 7e-4,
    "perp_quad_cost": 0.002698,
    "par_quad_cost": 1e-3,
}

DEFAULT_CONFIG = {
    "sim_speed": 1.0,
    "x0_offset": 0.0,
    "y0_offset": 0.0,
}


class FlightController:
    def __init__(
        self,
        config: dict[str, Any] = DEFAULT_CONFIG,
        cost_config: dict[str, float] = DEFAULT_COST_CONFIG,
        verbose: bool = True,
        view: Optional[xpc.ViewType] = None
    ):
        self.verbose = verbose
        self.config, self.cost_config = deepcopy(DEFAULT_CONFIG), deepcopy(DEFAULT_COST_CONFIG)
        self.config.update(deepcopy(config))
        self.cost_config.update(deepcopy(cost_config))

        self.xp = XPlaneConnectWrapper()
        self.flight_state = FlightState()
        time.sleep(0.3)
        self.int_state = np.zeros(6)
        self.state0, self.posi0 = self.get_curr_state(), self.xp.getPOSI()
        self.view = view if view is not None else xpc.ViewType.Chase

        # touchdown
        self.v_ref = 50.0
        self.params = dict()
        self.params["pos_ref"] = np.array([0.0, 0.0, 300.0])
        self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, deg2rad(self.posi0[5])])

        self.brake(0)
        self.t_start = time.time()
        self.u_hist, self.x_hist, self.t_hist = [], [], []
        self.controller = "lqr"
        self.read_dynamics()
        self.target = self.get_curr_state()
        #self.approach_ang = deg2rad(self.posi0[5]) - 0.15
        self.approach_ang = deg2rad(self.posi0[5]) - 0.1
        self.lock = Lock()
        self.ts, self.X, self.U, self.Ls = None, None, None, None
        self.done = False
        self.reset()

        if self.controller == "mpc":
            self.mpc_thread = Thread(target=self.mpc_worker, args=(self.lock,))
            self.mpc_thread.start()
        else:
            self.mpc_thread = None

        # define UI display
        self.ui = dsh.VSplit(
            dsh.HSplit(
                dsh.Text(title="State", text="", border_color=2, color=2),
                dsh.Text(title="Control", text="", border_color=2, color=2),
            ),
            dsh.Log(title="Debug", border_color=2, color=2),
            dsh.HBrailleChart(title="Distance to approach line", border_color=2, color=2),
        )
        self.ui_log = self.ui.items[1]
        self.last_ui_update = time.time()
        self.data = dict()

    def loop(self, abort_at: float = math.inf):
        t_loop_start = time.time()
        self.reset()

        self.data = dict()
        self.it = 0
        self.u_hist, self.x_hist, self.t_hist = [], [], []
        self.t_start = time.time()

        #T = 300.0 / self.config["sim_speed"]
        T = 110.0 / self.config["sim_speed"]
        dt_small = 1.0 / 50.0
        t_prev = 0.0
        while not self.done and time.time() - self.t_start < T:
            if time.time() - t_loop_start > abort_at:
                self.controller = "pid"
            t_prev = time.time()
            self.control()
            # self.advance_state(dt_small)
            sleep_for = max(0, dt_small - (time.time() - t_prev))
            time.sleep(sleep_for)
            self.it += 1
            is_crashed = self.xp.getDREF("sim/flightmodel2/misc/has_crashed")[0] > 0.0
            self.ui_log.append(f"Time = {time.time() - self.t_start:.2f} s")
            if is_crashed:
                reset_flight(self.xp)
                return True
        self.reset()
        return False

    @staticmethod
    def build_control(pitch=0, roll=0, yaw=0, throttle=0, gear=0, flaps=0):
        return [min(max(x, -1), 1) for x in [pitch, roll, yaw]] + [
            min(max(x, 0), 1) for x in [throttle, gear, flaps]
        ]

    def brake(self, brake=1):
        self.xp.sendDREF(utils.BRAKE, brake)

    def reset(self):
        """Reset the simulation to the state about 5km in the air behind the runway."""
        self.xp.sendDREF(utils.SIM_SPEED, self.config["sim_speed"])
        self.xp.sendVIEW(self.view)
        for _ in range(1):
            # arrest speed
            self.xp.sendPOSI(self.posi0)
            self.xp.sendDREFs(list(utils.SPEEDS.values()), [0 for _ in utils.SPEEDS.values()])
            # arrest rotation
            self.xp.sendDREFs(
                list(utils.ROTATION_SPEEDS.values()), [0 for _ in utils.ROTATION_SPEEDS.values()]
            )
            self.xp.sendPOSI(self.posi0)
            self.xp.sendCTRL(self.build_control())
            self.brake()

            posi = list(copy(self.posi0))
            posi[2] = 300
            dist = 6e3
            # posi[0] += dist / DEG_TO_METERS * -math.cos(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            # posi[1] += dist / DEG_TO_METERS * -math.sin(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            posi[0] += (
                dist / utils.DEG_TO_METERS * -math.cos(deg2rad(posi[5]))
                + self.config["x0_offset"] / utils.DEG_TO_METERS
            )
            posi[1] += (
                dist / utils.DEG_TO_METERS * -math.sin(deg2rad(posi[5]))
                + self.config["y0_offset"] / utils.DEG_TO_METERS
            )

            # set the plane at the new reset position, match simulation speed to heading
            self.xp.sendPOSI(posi)
            v = 60.0
            vx, vz = v * math.sin(deg2rad(self.posi0[5])), v * -math.cos(deg2rad(self.posi0[5]))
            self.xp.sendDREFs([utils.SPEEDS["local_vx"], utils.SPEEDS["local_vz"]], [vx, vz])

            time.sleep(0.3)
        self.data = dict()

    def get_time_state(self):
        return tuple(self.flight_state.last_sim_time_and_state)

    def get_curr_time(self):
        return self.flight_state.last_sim_time_and_state[0]

    def get_curr_state(self):
        state = self.flight_state.last_sim_time_and_state[1]
        state = np.concatenate([np.array(state), self.int_state])
        return state

    ################################################################################

    def read_dynamics(self):
        dynamics_path = (
            Path(__file__).absolute().parents[3] / "notebooks" / "data" / "dynamics_new2.json"
        )
        dynamics_state = json.loads(dynamics_path.read_text())
        fn = getattr(dynamics, "int_f_fx_fu_fn2")
        self.params.update({k: jaxm.array(v) for (k, v) in dynamics_state["params"].items()})
        params = copy(self.params)
        self.f_fx_fu_fn = RegisteredFunction(lambda x, u, *args: fn(x, u, params))

    def advance_state(self, dt):
        state = self.get_curr_state()
        pos_int = self.int_state[:3] + dt * (np.array(state[:3]) - self.params["pos_ref"])
        ang_int = self.int_state[3:6] + dt * (np.array(state[5:8]) - self.params["ang_ref"])
        int_state = np.concatenate([np.array(pos_int), np.array(ang_int)])
        self.int_state = 0.99**dt * int_state

    ################################################################################

    def construct_problem(self, state):
        p = Problem(N=10, xdim=11 + 6, udim=4)
        p.f_fx_fu_fn = self.f_fx_fu_fn
        p.x0 = state
        p.reg_x, p.reg_u = 3e-5 * np.ones(2)
        p.solver_settings = dict(p.solver_settings, solver="osqp")
        u_u = np.ones((p.N, p.udim))
        u_l = np.concatenate([-np.ones((p.N, p.udim - 1)), np.zeros((p.N, 1))], axis=-1)
        p.u_l, p.u_u = u_l, u_u
        x_ref = np.copy(p.x0)

        target = self.target[:2] + 400 * np.array([math.cos(self.approach_ang), math.sin(self.approach_ang)])
        dist = np.linalg.norm(target[:2] - p.x0[:2])

        # for lqr #####################################################
        cc = self.cost_config
        q_diag = (
            np.array(
                [cc["position_cost"], cc["position_cost"], cc["altitude_cost"]]
                + [1e3, 1e0]
                + [1e0, cc["roll_cost"], cc["heading_cost"]]
                + [1e-3, 1e-3, 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
            )
            / 1e3
        )

        Q = np.diag(q_diag)
        v_norm = np.array([math.cos(self.approach_ang), math.sin(self.approach_ang)])

        dx = np.array(target[:2]) - np.array(p.x0[:2])
        v_par = np.sum(dx * v_norm) * v_norm
        if np.sum(v_par * v_norm) < -200.0:
            self.done = True
        v_perp = dx - v_par
        self.ui.items[1].append(f"Distance to approach line: {np.linalg.norm(v_perp):.4e} m")
        d_par = math.sqrt(max(5e2**2 - np.linalg.norm(v_perp) ** 2, 0)) / np.linalg.norm(v_par)
        x_ref[:2] = (
            p.x0[:2]
            + max(np.linalg.norm(v_perp), 1e2) * v_perp / np.linalg.norm(v_perp)
            + d_par * v_par
        )
        angle = np.cos(p.x0[7] - self.params["ang_ref"][2])
        self.ui.items[1].append(f"Angle to approach line: {angle:.4e}")
        self.ui.items[2].append(float(np.linalg.norm(v_perp)) / 1e1 + 1)

        if "cost_approx" not in self.data:

            def cost_fn(x0, target, v_norm):
                """Compute a position cost as a scalar."""
                dx = target[:2] - x0[:2]
                v_par = jaxm.sum(dx * v_norm) * v_norm
                v_perp = dx - v_par
                v_perp_norm = jaxm.linalg.norm(v_perp)
                v_perp_norm2 = jaxm.sum(v_perp**2)
                v_par_norm = jaxm.linalg.norm(v_par)
                cc = self.cost_config
                Jv_perp = jaxm.where(
                    v_perp_norm > 1e3, v_perp_norm, cc["perp_quad_cost"] * v_perp_norm2
                )
                Jv_par = v_par_norm
                return cc["perp_cost"] * Jv_perp + cc["par_cost"] * Jv_par

            @jaxm.jit
            def cost_approx(x0, target, v_norm):
                """Develop a quadratic approximation of the cost function based on a scalar cost."""
                g = jaxm.grad(cost_fn, argnums=0)(x0, target, v_norm)
                H = jaxm.hessian(cost_fn, argnums=0)(x0, target, v_norm)
                Q = H + 1e-3 * jaxm.eye(H.shape[-1])
                ref = x0 - jaxm.linalg.solve(Q, g)
                return Q, ref

            self.data["cost_approx"] = cost_approx

        x_ref[2] = min(
            max(self.posi0[2], self.params["pos_ref"][2] * (dist / 5e3) ), 300.0
        )  # altitude
        x_ref[3:5] = self.v_ref, 0.0  # velocities
        x_ref[5:8] = self.params["ang_ref"]
        x_ref[8:11] = 0  # dangles
        x_ref[11:] = 0  # integrated errors

        Qx, refx = self.data["cost_approx"](p.x0[:2], np.array(target)[:2], np.array(v_norm))
        x_ref[:2] = refx[:2]
        Q[:2, :2] = Qx[:2, :2] / 1e3

        p.X_ref = x_ref
        p.Q = Q
        p.R = np.diag(np.array([1e0, 3e-1, 1e2, 1e0])) * 1e-1
        p.U_ref = np.array([0.0, 0.0, 0.0, 0.0])
        p.slew_rate = 1e2

        # interpolate previoous solution for MPC warm start
        if hasattr(self, "t_prev"):
            t_int = self.get_curr_time() + self.dt * np.arange(p.N + 1)
            opts = dict(axis=0, fill_value="extrapolate")
            X_prev = interp1d(self.t_prev, self.X_prev, **opts)(t_int)[1:, :]
            U_prev = interp1d(self.t_prev[:-1], self.U_prev, **opts)(t_int)[:-1, :]
            p.X_prev, p.U_prev = X_prev, U_prev
            p.max_it = 100
        else:
            p.X_prev = p.x0
            p.U_prev = np.zeros((p.N, p.udim))
            p.max_it = 400

        norm = np.linalg.norm(p.Q[0, :, :]) + np.linalg.norm(p.R[0, :, :])
        p.Q, p.R, p.slew_rate = p.Q / norm, p.R / norm, p.slew_rate / norm
        return p

    def mpc_worker(self, lock):
        while not self.done:
            t0, state = self.get_curr_time(), self.get_curr_state()
            p = self.construct_problem(state)
            X, U, _ = solve_problems([dict(**p)])[0]
            if X is not None and U is not None:
                Ls = all_sensitivity_L(X, U, dict(**p))
                with lock:
                    self.X, self.U, self.Ls = X, U, Ls
                    self.ts = t0 + self.dt * np.arange(p.N + 1)
                    self.t_prev, self.X_prev, self.U_prev = self.ts, self.X, self.U
            time.sleep(0.1)

    ################################################################################

    def control(self):
        """Compute and apply the control action."""
        if self.controller == "pid" or (self.controller == "mpc" and self.X is None):
            state = self.get_curr_state()
            pitch, roll, heading = state[5:8]
            pitch_ref, roll_ref, heading_ref = deg2rad(5.0), 0.0, self.state0[7]
            u_pitch = -1.0 * (pitch - pitch_ref)
            u_roll = -1.0 * (roll - roll_ref)
            u_heading = -1.0 * (30.0 / state[3]) * (heading - heading_ref)
            throttle = 0.7
            u = np.array([u_pitch, u_roll, u_heading, throttle])
        elif self.controller == "mpc":
            with self.lock:
                ts, X, U, Ls = self.ts, self.X, self.U, self.Ls
            curr_time, state = self.get_curr_time(), self.get_curr_state()
            opts = dict(axis=0, fill_value="extrapolate")
            x = interp1d(ts, X, **opts)(curr_time)
            u = interp1d(ts[:-1], U, **opts)(curr_time)
            u = u + Ls[0, :, :] @ (state - x)
        elif self.controller == "lqr":
            state = self.get_curr_state()
            p = self.construct_problem(state)
            Q, R, x_ref, u_ref = p.Q[0, :, :], p.R[0, :, :], p.X_ref[0, :], p.U_ref[0, :]
            # print(f"Distance to x_ref: {np.linalg.norm(state[:2] - p.X_ref[0, :2]):.4e}")
            u0 = np.zeros(p.udim)
            f, fx, fu = p.f_fx_fu_fn(state, u0)
            A, B, d = fx, fu, f - fx @ state - fu @ u0
            L, l = design_LQR_controller(A, B, d, Q, R, x_ref, u_ref, T=10)
            u = L @ state + l

        u_pitch, u_roll, u_heading, throttle = np.clip(u, [-1, -1, -1, 0], [1, 1, 1, 1])

        # initiate landing ####################################
        if state[2] < 5.0 or self.data.get("fixed_pitch", None) is not None:
            self.data.setdefault("fixed_pitch", u_pitch - 0.05)
            u_pitch, u_roll, u_heading, throttle = self.data["fixed_pitch"], 0.0, 0.0, 0.0
            self.brake(1)
        # initiate landing ####################################

        self.t_hist.append(self.get_curr_time())
        self.x_hist.append(copy(state))
        self.u_hist.append(copy(u))
        ctrl = self.build_control(pitch=u_pitch, roll=u_roll, yaw=u_heading, throttle=throttle)
        self.xp.sendCTRL(ctrl)

        # update the UI
        if self.verbose and time.time() - self.last_ui_update > 0.33:
            state_names = list(utils.FULL_STATE.keys())[:11]
            self.ui.items[0].items[0].text = "\n".join(
                [f"{name:<10s} {s:+07.4e}" for (name, s) in zip(state_names, state[:11])]
            )
            ctrl_names = ["pitch", "roll", "yaw", "throttle"]
            self.ui.items[0].items[1].text = "\n".join(
                [f"{name:<10s} {c:+07.4e}" for (name, c) in zip(ctrl_names, ctrl)]
            )
            self.last_ui_update = time.time()
            self.ui.display()

    def close(self):
        self.flight_state.close()
        self.done = True
        if self.mpc_thread is not None:
            self.mpc_thread.join()


####################################################################################################