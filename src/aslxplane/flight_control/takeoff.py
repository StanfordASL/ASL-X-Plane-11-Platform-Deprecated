import sys
import os
import time
from pathlib import Path
import json
import math
from copy import copy
from pathlib import Path
from multiprocessing import Process, Lock, Array, Event, Value
from threading import Thread

from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import xpc
import dashing as dsh

from pmpc import Problem
from pmpc.remote import solve_problems, RegisteredFunction
from pmpc.experimental.jax.root import all_sensitivity_L

try:
    from . import dynamics
    from .lqr import design_LQR_controller
except ImportError:
    root_path = Path(__file__).parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    import dynamics
    from lqr import design_LQR_controller

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "CPU"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JFI_COPY_NUMPY"] = "0"

from jfi import jaxm

####################################################################################################

SPEEDS = {
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
    def __init__(self):
        self.xp = xpc.XPlaneConnect()
        self.flight_state = FlightState()
        time.sleep(0.3)
        self.int_state = np.zeros(6)
        self.state0, self.posi0 = self.get_curr_state(), self.xp.getPOSI()

        # touchdown
        self.v_ref = 50.0
        self.params = dict()
        self.params["pos_ref"] = np.array([0.0, 0.0, 300.0])
        self.params["ang_ref"] = np.array([deg2rad(0.0), 0.0, deg2rad(self.posi0[5])])

        self.brake(0)
        self.t_start = time.time()
        T, self.dt = 600.0, 0.5
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

        # define UI display
        self.ui = dsh.VSplit(
            dsh.HSplit(
                dsh.Text(title="State", text="", border_color=2, color=2),
                dsh.Text(title="Control", text="", border_color=2, color=2),
            ),
            dsh.Log(title="Debug", border_color=2, color=2),
        )
        self.last_ui_update = time.time()

        self.data = dict()

        dt_small = 1.0 / 50.0
        t_prev = 0.0
        try:
            while time.time() - self.t_start < T:
                t_prev = time.time()
                self.control()
                #self.advance_state(dt_small)
                sleep_for = max(0, dt_small - (time.time() - t_prev))
                # print(f"Sleeping for {sleep_for:.4e} s")
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
            self.xp.sendPOSI(self.posi0)
            self.xp.sendDREFs(list(SPEEDS.values()), [0 for _ in SPEEDS.values()])
            # arrest rotation
            self.xp.sendDREFs(list(ROTATION_SPEEDS.values()), [0 for _ in ROTATION_SPEEDS.values()])
            self.xp.sendPOSI(self.posi0)
            self.xp.sendCTRL(self.build_control())
            self.brake()

            posi = list(copy(self.posi0))
            posi[2] = 300
            dist = 5e3
            posi[0] += dist / DEG_TO_METERS * -math.cos(deg2rad(posi[5])) - 3e1 / DEG_TO_METERS
            posi[1] += dist / DEG_TO_METERS * -math.sin(deg2rad(posi[5])) + 3e3 / DEG_TO_METERS
            self.xp.sendPOSI(posi)
            v = 60.0
            vx, vz = v * math.sin(deg2rad(self.posi0[5])), v * -math.cos(deg2rad(self.posi0[5]))
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
        x_l, x_u = -1e5 * np.ones(p.xdim), 1e5 * np.ones(p.xdim)
        x_ref = np.copy(p.x0)

        dist = np.linalg.norm(self.target[:2] - p.x0[:2])

        # for lqr #####################################################
        q_diag = (
            np.array(
                [1e0, 1e0, 1e2]
                + [1e3, 1e0]
                + [1e0, 3e4, 1e5]
                + [1e-3, 1e-3, 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
                + [0 * 1e-3, 0 * 1e-3, 0 * 1e-3]
            )
            / 1e3
        )

        approach_ang = deg2rad(self.posi0[5])
        Q = np.diag(q_diag)
        v_norm = np.array([math.cos(approach_ang), math.sin(approach_ang)])

        dx = np.array(self.target[:2]) - np.array(p.x0[:2])
        v_par = np.sum(dx * v_norm) * v_norm
        v_perp = dx - v_par
        self.ui.items[1].append(f"Distance to approach line: {np.linalg.norm(v_perp):.4e} m")
        d_par = math.sqrt(max(5e2**2 - np.linalg.norm(v_perp) ** 2, 0)) / np.linalg.norm(v_par)
        x_ref[:2] = (
            p.x0[:2]
            + max(np.linalg.norm(v_perp), 1e2) * v_perp / np.linalg.norm(v_perp)
            + d_par * v_par
        )

        if not hasattr(self, "cost_approx"):

            def cost_fn(x0, target, v_norm):
                """Compute a position cost as a scalar."""
                dx = target[:2] - x0[:2]
                v_par = jaxm.sum(dx * v_norm) * v_norm
                v_perp = dx - v_par
                v_perp_norm2 = jaxm.sum(v_perp**2)
                # Huber-like loss on the y-position distance from the approach line
                Jvp = jaxm.minimum(jaxm.linalg.norm(v_perp), 1e-3 * v_perp_norm2)
                return 1e3 * Jvp + 3e1 * jaxm.linalg.norm(v_par)

            @jaxm.jit
            def cost_approx(x0, target, v_norm):
                """Develop a quadratic approximation of the cost function based on a scalar cost."""
                g = jaxm.grad(cost_fn, argnums=0)(x0, target, v_norm)
                H = jaxm.hessian(cost_fn, argnums=0)(x0, target, v_norm)
                Q = H + 1e-3 * jaxm.eye(H.shape[-1])
                ref = x0 - jaxm.linalg.solve(Q, g)
                return Q, ref

            self.cost_approx = cost_approx

        x_ref[2] = min(
            max(self.posi0[2], self.params["pos_ref"][2] * (dist / 5e3)), 300.0
        )  # altitude
        x_ref[3:5] = self.v_ref, 0.0  # velocities
        x_ref[5:8] = self.params["ang_ref"]
        x_ref[8:11] = 0  # dangles
        x_ref[11:] = 0  # integrated errors

        Qx, refx = self.cost_approx(p.x0[:2], np.array(self.target)[:2], np.array(v_norm))
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
        if state[2] < 5.0:
            self.data.setdefault("fixed_pitch", u_pitch - 0.05)
            u_pitch, u_roll, u_heading, throttle = self.data["fixed_pitch"], 0.0, 0.0, 0.0
            self.brake(1)
        ctrl = self.build_control(pitch=u_pitch, roll=u_roll, yaw=u_heading, throttle=throttle)

        self.xp.sendCTRL(ctrl)

        if time.time() - self.last_ui_update > 0.33:
            state_names = list(FULL_STATE.keys())[:11]
            self.ui.items[0].items[0].text = "\n".join(
                [f"{name:<10s} {s:+07.4e}" for (name, s) in zip(state_names, state[:11])]
            )
            ctrl_names = ["pitch", "roll", "yaw", "throttle"]
            self.ui.items[0].items[1].text = "\n".join(
                [f"{name:<10s} {c:+07.4e}" for (name, c) in zip(ctrl_names, ctrl)]
            )
            self.last_ui_update = time.time()
            self.ui.display()

    ################################################################################

    def close(self):
        self.flight_state.close()
        self.done = True
        if self.mpc_thread is not None:
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
    controller = TakeoffController()
    controller.close()
    # hist = {
    #    "x": controller.x_hist,
    #    "u": controller.u_hist,
    #    "t": controller.t_hist,
    # }
    # i = 0
    # while len(r.keys(f"flight/new_hist{i}")) > 0:
    #    i += 1
    # r.set(f"flight/new_hist{i}", json.dumps(hist))


if __name__ == "__main__":
    main()
