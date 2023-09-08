from __future__ import annotations

import torch
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import math
import time
import sys
from pathlib import Path
from torch.multiprocessing import Process, Lock, Array

import cv2
import torch
from torch import nn
from torchvision.models.resnet import resnet50
from torchvision import transforms as T

try:
    from .state import FlightState
except ImportError:
    root_path = Path(__file__).absolute().parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    from aslxplane.utils.state import FlightState

from xplane_vision import transform_eval, get_model, kalman_filter, get_dynamics, R_default

####################################################################################################


class FlightStateWithVision(FlightState):
    def __init__(
        self,
        camera_id: int,
        model_path: str | Path,
        device: torch.device | str | None = None,
        ref: list | tuple | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.camera_id = camera_id
        self.model_path = model_path
        self.vis_lock = Lock()
        self.vis_shm = Array("d", [math.nan for _ in range(3)])
        self.vis_shm[:3] = [math.nan, math.nan, math.nan]
        super().__init__(ref)

    def start_process(self):
        super().start_process()
        self.vis_process = Process(
            target=self._vis_process,
            args=(
                self.lock,
                self.shm,
                self.vis_lock,
                self.vis_shm,
                self.close_event,
                self.camera_id,
                self.model_path,
                self.device,
            ),
        )
        self.vis_process.start()

    def close(self):
        super().close()
        if self.vis_process is not None:
            self.vis_process.join()
            self.vis_process = None

    @staticmethod
    def _vis_process(
        lock: Lock,
        shm: Array,
        vis_lock: Lock,
        vis_shm: Array,
        close_event,
        camera_id: int,
        model_path: Path | str | None = None,
        device: torch.device | str | None = None,
    ):
        # initialization phase ##########################################################
        f_fx_fu_fn, dyn_params = get_dynamics()
        with lock:
            P, x_estimate, t_prev = None, torch.as_tensor(shm[2:]), float(shm[1])
        P, x_estimate, t_prev = None, torch.zeros(3) * math.nan, None
        t_proc_start = time.time()
        while not torch.isfinite(x_estimate).all():
            with lock:
                P, x_estimate, t_prev = None, torch.as_tensor(shm[2:]), float(shm[1])
            time.sleep(1e-1)
        model = get_model(model_path, pretrained=True, device=device)
        print("#" * 40)
        print(f"device = {next(model.parameters()).device}")
        print("#" * 40)
        video = cv2.VideoCapture(camera_id)
        t_true_update = shm[-1]
        # initialization phase ##########################################################

        # loop phase ####################################################################
        while not close_event.is_set():
            ret, frame = video.read()
            if ret is False:
                with lock:
                    vis_shm[:3] = [math.nan, math.nan, math.nan]
                continue
            else:
                frame = (
                    transform_eval(torch.as_tensor(frame)).to(device).to(torch.float32)[None, ...]
                )
                with torch.no_grad():
                    z_obs = model(frame).reshape(-1).cpu() * torch.tensor([1e3, 1e3, 1e2])
                u_prev = torch.zeros(4).to(z_obs)
                x_prev = x_estimate
                with lock:
                    t_curr = float(shm[1])
                dyn_params["dt_sqrt"] = math.sqrt(max(0.0, t_curr - t_prev))
                if dyn_params["dt_sqrt"] ** 2 > 1e-5:
                    R = 1e0 * R_default
                    x_estimate, _ = kalman_filter(
                        lambda x, u: f_fx_fu_fn(x, u, dyn_params), z_obs, x_prev, u_prev, P=P, R=R
                    )
                    t_prev = t_curr

                # react to state estimate corruption #########################
                if P is not None and (not torch.all(torch.isfinite(P)) or torch.norm(P) > 1e4):
                    print("#" * 40)
                    print("Resetting Kalman filter")
                    print("#" * 40, flush=True)
                    with lock:
                        P, x_estimate = None, torch.as_tensor(shm[2:]) # reset
                if shm[1] - t_true_update > 5.0:
                    t_true_update = shm[1]
                    with lock:
                        x_estimate = torch.as_tensor(shm[2:])
                        t_prev = t_curr
                    print("Update state to true", flush=True)
                if time.time() - t_proc_start < 50.0:
                    print("Resetting to true still", flush=True)
                    with lock:
                        x_estimate = torch.as_tensor(shm[2:])
                        t_prev = t_curr

                with vis_lock:
                    vis_shm[:3] = list(x_estimate.detach().cpu().numpy())[:3]
            time.sleep(1.0 / 60.0)
        # loop phase ####################################################################

        video.release()

    @property
    def last_sim_time_and_state(self):
        if (
            self.process is None
            or not self.process.is_alive()
            or self.vis_process is None
            or not self.vis_process.is_alive()
        ):
            self.start_process()
        with self.lock:
            real_time, sim_time, state = self.shm[0], self.shm[1], self.shm[2:]
        with self.vis_lock:
            vis = self.vis_shm[:3]
        if any(not math.isfinite(x) for x in vis[:3]):
            state = list(state)
        else:
            state = list(vis)[:3] + list(state)[3:]
        self.real_times.append(real_time)
        self.sim_times.append(sim_time)
        if len(self.real_times) > 100:
            self.real_times = self.real_times[-100:]
            self.sim_times = self.sim_times[-100:]
        return sim_time, state
