import time
import sys
from pathlib import Path
from typing import Union
from multiprocessing import Event, Process
import traceback
import json

import cv2

try:
    from .state import FlightState
except ImportError:
    root_path = Path(__file__).absolute().parents[2]
    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    from aslxplane.utils.state import FlightState

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
                self.flight_state.ref,
                self.record_id,
                self.display,
                self.close_event,
            ),
        )
        self.process.start()

    def _visualization_process(self):
        pass

    @staticmethod
    def _record_process(movie_name, meta_name, flight_state_ref, record_id, display, close_event):
        flight_state = FlightState(flight_state_ref)
        data, reader, writer, frame_id = [], cv2.VideoCapture(record_id), None, 0
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
