import math
import struct
import traceback
import time
import json
from pathlib import Path
from typing import Union
import socket
from functools import partial
from multiprocessing import Array, Event, Lock, Process, Queue

import cv2
import numpy as np

import xpc

try:
    from ..simulation import corruptions
except ImportError:
    root_path = Path(__file__).parents[2]
    from aslxplane.simulation import corruptions

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


def rad2deg(x):
    return x * 180 / math.pi


def deg2rad(x):
    return x * math.pi / 180


####################################################################################################


class XPlaneConnectWrapper:
    """An XPlaneConnect wrapper that automatically reconnects if the socket breaks."""

    def __init__(self):
        self.xp = xpc.XPlaneConnect()
        self.sendCTRL = partial(self.wrap_call, "sendCTRL")
        self.getDREF = partial(self.wrap_call, "getDREF")
        self.getDREFs = partial(self.wrap_call, "getDREFs")
        self.sendDREF = partial(self.wrap_call, "sendDREF")
        self.sendDREFs = partial(self.wrap_call, "sendDREFs")
        self.getPOSI = partial(self.wrap_call, "getPOSI")
        self.sendPOSI = partial(self.wrap_call, "sendPOSI")
        self.sendUDP = partial(self.wrap_call, "sendUDP")
        self.sendVIEW = partial(self.wrap_call, "sendVIEW")

    def wrap_call(self, fn_name, *args, **kwargs):
        while True:
            try:
                ret = getattr(self.xp, fn_name)(*args, **kwargs)
                return ret
            except socket.timeout:
                self.xp = xpc.XPlaneConnect()

    def close(self):
        self.xp.close()


####################################################################################################


def reset_flight(xp: XPlaneConnectWrapper, on_crash_only=True):
    """Send a command COMM to the XPlaneConnect plugin in the format of (COMM, len(cmd), cmd)."""
    is_crashed = xp.getDREF("sim/flightmodel2/misc/has_crashed")[0] > 0.0
    if is_crashed or not on_crash_only:
        for cmd in ["sim/operation/reset_flight", "sim/operation/close_windows"]:
            buffer = struct.pack(f"<4sxB{len(cmd)}s".encode(), b"COMM", len(cmd), cmd.encode())
            xp.sendUDP(buffer)
    time.sleep(4.0)


####################################################################################################


class FlightState:
    def __init__(self):
        self.shm = Array("d", [math.nan for _ in range(2 + len(FULL_STATE))])
        self.lock = Lock()
        self.close_event = Event()
        xp = XPlaneConnectWrapper()
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
        xp = XPlaneConnectWrapper()
        sim_time0, lat0, lon0 = ref
        while not close_event.is_set():
            real_time = time.time()
            state_time = [x[0] for x in xp.getDREFs(list(FULL_STATE.values()) + [SIM_TIME])]
            real_time = (time.time() + real_time) / 2.0
            state, sim_time = state_time[:-1], state_time[-1] - sim_time0
            state[5:] = [deg2rad(x) for x in state[5:]]
            state[0] = (state[0] - lat0) * DEG_TO_METERS
            state[1] = (state[1] - lon0) * DEG_TO_METERS
            with lock:
                shm[:] = [real_time, sim_time] + state
            time.sleep(1e-3)
        xp.close()


####################################################################################################


class VideoRecorder:
    def __init__(
        self,
        recording_name: Union[str, Path],
        flight_state: FlightState,
        record_id: int = 2,
        display: bool = True,
    ):
        self.name = Path(recording_name).with_suffix("")
        self.movie_name = self.name.with_suffix(".avi")
        self.meta_name = self.name.with_suffix(".json")
        self.process, self.close_event = None, Event()
        self.flight_state, self.record_id, self.display = flight_state, record_id, display
        self.start_process()

    def start_process(self):
        self.process = Process(
            target=self._record_process,
            args=(
                self.movie_name,
                self.meta_name,
                self.flight_state,
                self.record_id,
                self.display,
                self.close_event,
            ),
        )
        self.process.start()

    def _visualization_process(self):
        pass

    @staticmethod
    def _record_process(movie_name, meta_name, flight_state, record_id, display, close_event):
        flight_state = FlightState()
        data, reader, writer, frame_id = [], cv2.VideoCapture(record_id), None, 0
        corr = corruptions.Rain()
        while not close_event.is_set():
            ret, frame = reader.read()
            if not ret:
                break  # end the recording if there's a problem with the virtual camera
            if writer is None:
                fcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(
                    str(movie_name), fcc, 60.0, (frame.shape[1], frame.shape[0])
                )
                continue  # we're ignoring the first frame since it might be badly timed
            sim_time, state = flight_state.last_sim_time_and_state
            data.append(
                {"frame_id": frame_id, "time": time.time(), "sim_time": sim_time, "state": state}
            )
            writer.write(frame)
            if display:
                try:
                    cv2.imshow("frame", cv2.resize(frame, (0, 0), fx=0.2, fy=0.2))
                    cv2.waitKey(1)
                except:
                    traceback.print_exc()
            frame_id += 1
        Path(meta_name).write_text(json.dumps(data))
        flight_state.close()

    def close(self):
        self.close_event.set()
        self.process.join()


####################################################################################################
