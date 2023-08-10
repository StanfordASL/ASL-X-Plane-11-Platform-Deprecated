from __future__ import annotations

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

####################################################################################################


def load_model(model_path: Path | str, device: torch.device | str = "cuda") -> nn.Module:
    model = resnet50()
    model.fc = nn.Linear(model.fc.weight.shape[-1], 3)
    model.conv1 = nn.Sequential(nn.BatchNorm2d(3), model.conv1)
    model.to(device)
    model.load_state_dict(torch.load(Path(model_path).absolute(), map_location=device))
    model.eval()
    return model


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
        super().__init__(ref)

    def start_process(self):
        super().start_process()
        self.vis_process = Process(
            target=self._vis_process,
            args=(
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
    def _vis_process(lock, shm, close_event, camera_id, model_path, device):
        model = load_model(model_path, device)
        video = cv2.VideoCapture(camera_id)
        transform = T.Compose([T.Lambda(lambda x: x.permute(2, 0, 1)), T.Resize((360, 480))])
        while not close_event.is_set():
            ret, frame = video.read()
            if ret is False:
                with lock:
                    shm[:3] = [math.nan, math.nan, math.nan]
                continue
            else:
                frame = transform(torch.as_tensor(frame)).to(device).to(torch.float32)[None, ...]
                with torch.no_grad():
                    output = model(frame).reshape(-1).cpu()
                    output = output * torch.tensor([1e3, 1e3, 1e2])
                    output = output.numpy().tolist()
                with lock:
                    shm[:3] = output
            time.sleep(1.0 / 60.0)
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
        state = list(vis)[:3] + list(state)[3:]
        self.real_times.append(real_time)
        self.sim_times.append(sim_time)
        if len(self.real_times) > 100:
            self.real_times = self.real_times[-100:]
            self.sim_times = self.sim_times[-100:]
        return sim_time, state
