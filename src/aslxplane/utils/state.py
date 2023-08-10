import sys
import math
import time
from multiprocessing import Array, Event, Lock, Process
from pathlib import Path

import numpy as np

try:
    from .robust_xpc import RobustXPlaneConnect
except ImportError:
    root_path = Path(__file__).absolute().parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    from aslxplane.utils.robust_xpc import RobustXPlaneConnect

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

LATLON_DEG_TO_METERS = 1852 * 60

####################################################################################################

def rad2deg(x):
    return x * 180 / math.pi


def deg2rad(x):
    return x * math.pi / 180

####################################################################################################


class FlightState:
    def __init__(self, ref=None):
        self.shm = Array("d", [math.nan for _ in range(2 + len(FULL_STATE))])
        self.lock = Lock()
        self.close_event = Event()
        xp = RobustXPlaneConnect()
        if ref is None:
            self.ref = [
                x[0] for x in xp.getDREFs([SIM_TIME, STATE_REF["lat_ref"], STATE_REF["lon_ref"]])
            ]
        else:
            self.ref = ref
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
        if self.process is not None:
            self.process.join()
            self.process = None

    def posi2state_pos(self, posi):
        sim_time0, lat0, lon0 = self.ref
        state_pos = [None for _ in range(3)]
        state_pos[0] = (posi[0] - lat0) * LATLON_DEG_TO_METERS
        state_pos[1] = (posi[1] - lon0) * LATLON_DEG_TO_METERS
        state_pos[2] = posi[2]
        return state_pos

    def state_pos2posi(self, state_pos):
        sim_time0, lat0, lon0 = self.ref
        posi = [None for _ in range(3)]
        posi[0] = state_pos[0] / LATLON_DEG_TO_METERS + lat0
        posi[1] = state_pos[1] / LATLON_DEG_TO_METERS + lon0
        posi[2] = state_pos[2]
        return posi

    @staticmethod
    def _state_query_process(lock, shm, ref, close_event):
        xp = RobustXPlaneConnect()
        sim_time0, lat0, lon0 = ref
        while not close_event.is_set():
            real_time = time.time()
            state_time = [x[0] for x in xp.getDREFs(list(FULL_STATE.values()) + [SIM_TIME])]
            real_time = (time.time() + real_time) / 2.0
            state, sim_time = state_time[:-1], state_time[-1] - sim_time0
            state[5:] = [deg2rad(x) for x in state[5:]]
            state[0] = (state[0] - lat0) * LATLON_DEG_TO_METERS
            state[1] = (state[1] - lon0) * LATLON_DEG_TO_METERS
            with lock:
                shm[:] = [real_time, sim_time] + state
            time.sleep(1e-3)
        xp.close()
