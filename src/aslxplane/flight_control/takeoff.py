import sys
import os
import time
import json
import math
from copy import copy
from pathlib import Path
from multiprocessing import Process, Lock, Array, Event, Value
from threading import Thread

os.environ["JFI_COPY_NUMPY"] = "0"

import redis
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

try:
    from ...xpc3 import xpc3 as xpc
    from . import dynamics
    from .dynamics import fwd_fn, f_fx_fu_fn, nn_fwd_fn, nn_f_fx_fu_fn
except ImportError:
    root_path = Path(__file__).parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    # from xpc3.xpc3 import XPlaneConnect
    import xpc
    import dynamics
    from dynamics import fwd_fn, f_fx_fu_fn, nn_fwd_fn, nn_f_fx_fu_fn
    from lqr import design_LQR_controller

from jfi import jaxm
from pmpc import Problem
from pmpc.remote import solve_problems, RegisteredFunction
from pmpc.experimental.jax.root import all_sensitivity_L


SPEEDS = {
    # "airspeed": "sim/flightmodel/position/true_airspeed",
    # "vertical_speed": "sim/flightmodel/position/vh_ind",
    # "groundspeed": "sim/flightmodel/position/groundspeed",
    "local_vx": "sim/flightmodel/position/local_vx",
    "local_vy": "sim/flightmodel/position/local_vy",
    "local_vz": "sim/flightmodel/position/local_vz",
}
BRAKE = "sim/flightmodel/controls/parkbrake"
ROTATION_SPEEDS = {
    "dpitch": "sim/flightmodel/position/Q",
    "dyaw": "sim/flightmodel/position/R",
    "droll": "sim/flightmodel/position/P",
}
SIM_SPEED = "sim/time/sim_speed"

FULL_STATE = {
    "x": "sim/flightmodel/position/latitude",
    "y": "sim/flightmodel/position/longitude",
    "z": "sim/flightmodel/position/elevation",
    "v": "sim/flightmodel/position/true_airspeed",
    "vh": "sim/flightmodel/position/vh_ind",
    "pitch": "sim/flightmodel/position/theta",
    "roll": "sim/flightmodel/position/phi",
    "yaw": "sim/flightmodel/position/psi",
    "dpitch": "sim/flightmodel/position/Q",
    "droll": "sim/flightmodel/position/P",
    "dyaw": "sim/flightmodel/position/R",
}
STATE_REF = {
    # "lat_ref": "sim/flightmodel/position/lat_ref",
    # "lon_ref": "sim/flightmodel/position/lon_ref",
    "lat_ref": "sim/flightmodel/position/latitude",
    "lon_ref": "sim/flightmodel/position/longitude",
}
SIM_TIME = "sim/time/total_flight_time_sec"

DEG_TO_METERS = 1852 * 60

####################################################################################################


def rad2deg(x):
    return x * 180 / math.pi


def deg2rad(x):
    return x * math.pi / 180


####################################################################################################


class TakeoffController:
    def __init__(self, rconn: redis.Redis):
        self.xp = xpc.XPlaneConnect()
        self.flight_state = FlightState()
        time.sleep(0.3)
        self.int_state = np.zeros(6)
        self.state0, self.posi0 = self.get_curr_state(), self.xp.getPOSI()
        self.rconn = rconn

        # touchdown
        self.v_ref = 50.0
        self.params = dict()
        self.params["pos_ref"] = np.array([0.0, 0.0, 300.0])
        self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, deg2rad(self.posi0[5])])
        #self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, 4.0])

        # cruise
        # self.v_ref = 60.0
        # self.params["pos_ref"] = np.array([0.0, 0.0, 100.0])
        # self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, 4.0])

        self.brake(0)
        self.t_start = time.time()
        T, self.dt = 150.0, 0.5
        N = math.ceil(T / self.dt)
        self.u_random = 3e-1 * np.random.randn(*(N, 4))
        self.u_random[: round(5.0 / self.dt), :] = 0
        self.u_random = 0 * self.u_random
        self.it = 0
        self.u_hist, self.x_hist, self.t_hist = [], [], []
        self.controller = "lqr"
        self.read_dynamics()
        self.target = self.get_curr_state()
        self.lock = Lock()
        self.ts, self.X, self.U, self.Ls = None, None, None, None
        self.done = False

        self.reset()
        if self.controller == "mpc":
            self.mpc_thread = Thread(target=self.mpc_worker, args=(self.lock,))
            self.mpc_thread.start()
        else:
            self.mpc_thread = None

        dt_small = 1.0 / 50.0
        t_prev = 0.0
        try:
            while time.time() - self.t_start < T:
                t_prev = time.time()
                self.control()
                #self.advance_state(dt_small)
                sleep_for = max(0, dt_small - (time.time() - t_prev))
                print(f"Sleeping for {sleep_for:.4e} s")
                time.sleep(sleep_for)
                self.it += 1
        except KeyboardInterrupt:
            pass
        self.reset()

    @staticmethod
    def build_control(pitch=0, roll=0, yaw=0, throttle=0, gear=0, flaps=0):
        return [min(max(x, -1), 1) for x in [pitch, roll, yaw]] + [
            min(max(x, 0), 1) for x in [throttle, gear, flaps]
        ]

    def brake(self, brake=1):
        self.xp.sendDREF(BRAKE, brake)

    def reset(self):
        self.xp.sendDREF(SIM_SPEED, 1.0)
        self.xp.sendVIEW(xpc.ViewType.Chase)
        for _ in range(1):
            self.xp.pauseSim(True)
            # arrest speed
            posi0 = list(copy(self.posi0))
            self.xp.sendPOSI(self.posi0)
            self.xp.sendDREFs(list(SPEEDS.values()), [0 for _ in SPEEDS.values()])
            # arrest rotation
            self.xp.sendDREFs(list(ROTATION_SPEEDS.values()), [0 for _ in ROTATION_SPEEDS.values()])
            self.xp.sendPOSI(self.posi0)
            self.xp.sendCTRL(self.build_control())
            self.brake()

            posi0[2] = 300
            dist = 5e3
            posi0[0] += dist / DEG_TO_METERS * -math.cos(deg2rad(posi0[5]))
            posi0[1] += dist / DEG_TO_METERS * -math.sin(deg2rad(posi0[5])) + 2e3 / DEG_TO_METERS
            self.xp.sendPOSI(posi0)
            v = 60.0
            vx, vz = v * math.sin(deg2rad(posi0[5])), v * -math.cos(deg2rad(posi0[5]))
            self.xp.sendDREFs([SPEEDS["local_vx"], SPEEDS["local_vz"]], [vx, vz])

            self.xp.pauseSim(False)
            time.sleep(0.3)

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
        dynamics_state = json.loads(self.rconn.get("flight/dynamics_new2"))
        # fn = getattr(dynamics, dynamics_state["name"])
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
        # p.reg_x, p.reg_u = 1e-5 * np.ones(2)
        # p.reg_x, p.reg_u = 1e-5 * np.ones(2)
        p.reg_x, p.reg_u = 3e-5 * np.ones(2)
        p.solver_settings = dict(p.solver_settings, solver="osqp")
        u_u = np.ones((p.N, p.udim))
        u_l = np.concatenate([-np.ones((p.N, p.udim - 1)), np.zeros((p.N, 1))], axis=-1)
        p.u_l, p.u_u = u_l, u_u
        x_l, x_u = -1e5 * np.ones(p.xdim), 1e5 * np.ones(p.xdim)
        # x_l[2] = 30.0
        # p.x_l, p.x_u = np.tile(x_l, (p.N, 1)), np.tile(x_u, (p.N, 1))
        x_ref = np.copy(p.x0)

        dist = np.linalg.norm(self.target[:2] - p.x0[:2])

        # for lqr #####################################################
        # x_cost_scale = (
        #    [1e-9, 1e-9, 1e1]  # position
        #    + [1e1, 1e1]  # velocities
        #    + [1e1, 1e3, 1e4]  # angles
        #    + [1e0, 1e0, 1e0]  # angular rate
        #    + [1e-9, 1e-9, 1e-9]  # integrated position
        #    + [1e-9, 3e-9, 1e-9]  # integrated angles
        # )
        #roll_cost = max(1e5 * (1 - (time.time() - self.t_start) / 20.0), 0) + 3e3
        #roll_cost = max(1e6 * (1 - (time.time() - self.t_start) / 20.0), 0) + 1e4
        #roll_cost = max(1e6 * (1 - (time.time() - self.t_start) / 20.0), 0) + 3e4
        #pitch_cost = max(1e7 * (1 - (time.time() - self.t_start) / 20.0), 0) + 1e0
        #roll_cost = max(1e6 * (1 - (time.time() - self.t_start) / 20.0), 0) + 3e4
        pitch_cost = max(1e6 * (1 - (time.time() - self.t_start) / 10.0), 0) + 1e0
        roll_cost = 3e4
        #roll_cost = 1e5
        pos_cost = 1e0
        x_cost_scale = (
            #[3e-1, 3e-1, 1e2]
            #[1e0, 1e0, 1e2]
            #[3e0, 3e0, 1e2]
            [pos_cost, pos_cost, 1e2]
            #[3e0 / (dist / 100), 3e0 / (dist / 100), 1e2]
            + [1e2, 1e0]
            #+ [1e0, 3e3, 1e4]
            #+ [1e0, 3e4, 1e3]
            + [pitch_cost, roll_cost, 1e0]
            + [1e-3, 1e-3, 1e-3]
            + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
            + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
        )

        # for heading target ##########################################
        # x_cost_scale = (
        #    [1e2, 1e2, 1e3]  # position
        #    + [1e2, 1e2]  # velocities
        #    + [1e2, 1e6, 1e6]  # angles
        #    + [1e3, 1e2, 1e2]  # angular rate
        #    + [1e-9, 1e-9, 1e3]  # integrated position
        #    #+ [1e-9, 3e5, 1e6]  # integrated angles
        #    + [1e-9, 3e4, 1e4]  # integrated angles
        #    #+ [1e-9, 1e-9, 1e-9]  # integrated angles
        # )

        # for landing #################################################
        # x_cost_scale = (
        #    [3e1, 3e1, 1e4]  # position
        #    + [1e2, 1e2]  # velocities
        #    + [1e2, 1e3, 1e3]  # angles
        #    + [1e3, 1e2, 1e2]  # angular rate
        #    + [1e-9, 1e-9, 1e-9]  # integrated position
        #    + [1e-9, 1e-9, 1e-9]  # integrated angles
        # )
        x_cost_scale = np.array(x_cost_scale) / 1e3
        q_diag = x_cost_scale

        ang = deg2rad(self.posi0[5])
        #ang = 2.0

        Q = np.diag(q_diag)
        #c, s = math.cos(ang), math.sin(ang)
        #Qx = np.diag([1e-1, 1e0]) / 1e3
        #R = np.array([[c, -s], [s, c]])
        #Q[:2, :2] = np.linalg.inv(R).T @ Qx @ np.linalg.inv(R)
        #Q[:2, :2] = R.T @ Qx @ R

        # x_ref[:2] = p.x0[:2]  # positions
        #ang = 4.0
        v_norm = np.array([math.cos(ang), math.sin(ang)])
        #x_ref[:2] = np.array(self.target[:2]) - (4.5e3 * (1.0 - (time.time() - self.t_start) / 90.0)) * v_norm
        dx = np.array(self.target[:2]) - np.array(p.x0[:2])
        v_par = np.sum(dx * v_norm) * v_norm
        v_perp = dx - v_par
        d_par = math.sqrt(max(5e2 ** 2 - np.linalg.norm(v_perp) ** 2, 0)) / np.linalg.norm(v_par)
        #d_par = np.linalg.norm(v_perp) / np.linalg.norm(v_par)



        x_ref[:2] = p.x0[:2] + max(np.linalg.norm(v_perp), 1e2) * v_perp / np.linalg.norm(v_perp) + d_par * v_par
        #v_target = v_perp + d_par * v_par / np.linalg.norm(v_par)
        #x_ref[:2] = p.x0[:2] + 2e3 * v_target / np.linalg.norm(v_target)



        #x_ref[:2] = self.target[:2]
        #x_ref[:2] = p.x0[:2]
        # x_ref[2] = max(0, self.params["pos_ref"][2] - 300 / 90.0 * (time.time() - self.t_start)) # altitude
        x_ref[2] = max(0, self.params["pos_ref"][2] * (dist / 5e3))  # altitude
        #x_ref[2] = 300.0  # altitude
        x_ref[3:5] = self.v_ref, 0.0  # velocities
        #x_ref[3:5] = 60.0, 0.0  # velocities
        x_ref[5:8] = self.params["ang_ref"]
        x_ref[8:11] = 0  # dangles
        x_ref[11:] = 0  # integrated errors

        p.X_ref = x_ref
        p.Q = Q
        # p.Q[:-1, :, :] *= 3e-1
        # p.Q[-1, :, :] *= 3e0

        # p.R = np.diag(np.array([1e3, 3e2, 3e4, 1e-1])) * 3e0
        # p.R = np.diag(np.array([1e3, 3e2, 3e4, 1e-1])) * 1e1
        #p.R = np.diag(np.array([1e0, 1e0, 1e3, 1e0])) * 3e-2
        p.R = np.diag(np.array([1e0, 1e0, 1e3, 1e0])) * 1e0
        p.U_ref = np.array([0.0, 0.0, 0.0, 0.0])
        p.slew_rate = 1e2
        # p.slew_rate = 1e0

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
        if self.controller == "pid" or (self.controller == "mpc" and self.X is None):
            state = self.get_curr_state()
            pitch, roll, heading = state[5:8]
            pitch_ref, roll_ref, heading_ref = deg2rad(5.0), 0.0, self.state0[7]
            u_pitch = -1.0 * (pitch - pitch_ref)
            u_roll = -1.0 * (roll - roll_ref)
            u_heading = -1.0 * (30.0 / state[3]) * (heading - heading_ref)
            throttle = 0.7

            # u = np.array([u_pitch, u_roll, u_heading, throttle]) + self.u_random[self.it, :]
            u = np.array([u_pitch, u_roll, u_heading, throttle])
        elif self.controller == "mpc":
            with self.lock:
                ts, X, U, Ls = self.ts, self.X, self.U, self.Ls
            curr_time, state = self.get_curr_time(), self.get_curr_state()
            # try:
            opts = dict(axis=0, fill_value="extrapolate")
            x = interp1d(ts, X, **opts)(curr_time)
            u = interp1d(ts[:-1], U, **opts)(curr_time)
            u = u + Ls[0, :, :] @ (state - x)
            # except ValueError:
            #    u = np.zeros(4)

            # plt.figure(45453)
            # plt.clf()
            ##plt.plot(self.ts, self.X[:, 5], label="pitch")
            ##plt.plot(self.ts, self.X[:, 6], label="roll")
            ##plt.plot(self.ts, self.X[:, 7], label="yaw")
            ##plt.scatter(curr_time, self.get_curr_state()[6], label="roll-x")
            # plt.plot(self.ts[:-1], self.U[:, 0], label="pitch")
            # plt.plot(self.ts[:-1], self.U[:, 1], label="roll")
            # plt.plot(self.ts[:-1], self.U[:, 2], label="yaw")
            # plt.plot(self.ts[:-1], self.U[:, 3], label="throttle")
            # plt.ylim([-1.2, 1.2])
            # plt.legend()
            # plt.draw()
            # plt.pause(1e-2)
            # self.v_ref = 35.0 / (1 + 0.05 * (time.time() - self.t_start))
        elif self.controller == "lqr":
            state = self.get_curr_state()
            p = self.construct_problem(state)
            Q, R, x_ref, u_ref = p.Q[0, :, :], p.R[0, :, :], p.X_ref[0, :], p.U_ref[0, :]
            print(f"Distance to x_ref: {np.linalg.norm(state[:2] - p.X_ref[0, :2]):.4e}")
            u0 = np.zeros(p.udim)
            f, fx, fu = p.f_fx_fu_fn(state, u0)
            A, B, d = fx, fu, f - fx @ state - fu @ u0
            L, l = design_LQR_controller(A, B, d, Q, R, x_ref, u_ref, T=10)
            u = L @ state + l
            #if abs(state[6]) > 0.1:
            #    u = np.array(u)
            #    u[1] = -0.3 * (state[6] - 0.1)

        # construct history
        # self.u_hist.append([u_pitch, u_roll, u_heading, throttle])
        # self.x_hist.append(state)
        # self.t_hist.append(self.sim_time)

        # print(f"pitch: {u_pitch}, roll: {u_roll}, heading: {u_heading}")
        u_pitch, u_roll, u_heading, throttle = np.clip(u, [-1, -1, -1, 0], [1, 1, 1, 1])
        if state[2] < 5.0:
            if not hasattr(self, "fixed_pitch"):
                self.fixed_pitch = u_pitch - 0.05
            u_pitch, u_roll, u_heading, throttle = self.fixed_pitch, 0.0, 0.0, 0.0
            self.brake(1)
        print(tabulate([state[:11]], floatfmt="+.1e"))
        ctrl = self.build_control(pitch=u_pitch, roll=u_roll, yaw=u_heading, throttle=throttle)
        print(tabulate([ctrl], floatfmt="+.1e"))
        self.xp.sendCTRL(ctrl)

    ################################################################################

    def close(self):
        self.flight_state.close()
        self.done = True
        self.mpc_thread.join()


####################################################################################################


class FlightState:
    def __init__(self):
        self.shm = Array("d", [math.nan for _ in range(2 + len(FULL_STATE))])
        self.lock = Lock()
        self.close_event = Event()
        xp = xpc.XPlaneConnect()
        self.ref = [
            x[0] for x in xp.getDREFs([SIM_TIME, STATE_REF["lat_ref"], STATE_REF["lon_ref"]])
        ]
        xp.close()
        self.start_process()
        self.real_times, self.sim_times = [], []
        time.sleep(1e-1)

    @property
    def last_sim_time_and_state(self):
        if not self.process.is_alive():
            self.start_process()
        with self.lock:
            real_time, sim_time, state = self.shm[0], self.shm[1], self.shm[2:]
        self.real_times.append(real_time)
        self.sim_times.append(sim_time)
        if len(self.real_times) > 100:
            self.real_times = self.real_times[-100:]
            self.sim_times = self.sim_times[-100:]
        return sim_time, state

    def estimated_time(self):
        median_diff = np.median(np.array(self.sim_times) - np.array(self.real_times))
        return time.time() + median_diff

    def start_process(self):
        self.process = Process(
            target=self._state_query_process, args=(self.lock, self.shm, self.ref, self.close_event)
        )
        self.process.start()

    def close(self):
        self.close_event.set()
        self.process.join()

    def posi2state_pos(self, posi):
        sim_time0, lat0, lon0 = self.ref
        state_pos = [None for _ in range(3)]
        state_pos[0] = (posi[0] - lat0) * DEG_TO_METERS
        state_pos[1] = (posi[1] - lon0) * DEG_TO_METERS
        state_pos[2] = posi[2]
        return state_pos

    def state_pos2posi(self, state_pos):
        sim_time0, lat0, lon0 = self.ref
        posi = [None for _ in range(3)]
        posi[0] = state_pos[0] / DEG_TO_METERS + lat0
        posi[1] = state_pos[1] / DEG_TO_METERS + lon0
        posi[2] = state_pos[2]
        return posi

    @staticmethod
    def _state_query_process(lock, shm, ref, close_event):
        xp = xpc.XPlaneConnect()
        sim_time0, lat0, lon0 = ref
        while not close_event.is_set():
            real_time = time.time()
            state_time = [x[0] for x in xp.getDREFs(list(FULL_STATE.values()) + [SIM_TIME])]
            real_time = (time.time() + real_time) / 2.0
            state, sim_time = state_time[:-1], state_time[-1] - sim_time0
            state[5:] = [deg2rad(x) for x in state[5:]]
            # print(f"Getting state took {time.time() - t:.4e} s")
            state[0] = (state[0] - lat0) * DEG_TO_METERS
            state[1] = (state[1] - lon0) * DEG_TO_METERS
            with lock:
                shm[:] = [real_time, sim_time] + state
            time.sleep(1e-3)
        xp.close()


####################################################################################################


####################################################################################################
def main():
    r = redis.Redis(password=os.environ["REDIS_PASSWORD"])
    controller = TakeoffController(rconn=r)
    controller.close()
    hist = {
        "x": controller.x_hist,
        "u": controller.u_hist,
        "t": controller.t_hist,
    }
    i = 0
    while len(r.keys(f"flight/new_hist{i}")) > 0:
        i += 1
    r.set(f"flight/new_hist{i}", json.dumps(hist))


if __name__ == "__main__":
    main()
