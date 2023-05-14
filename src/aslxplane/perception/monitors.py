import torch
import numpy as np
import torch.nn as nn
import xpc3.xpc3 as xpc3
import xpc3.xpc3_helper as xpc3_helper
import torchvision.transforms as tform
import aslxplane.perception.models as xplane_models
from abc import ABC, abstractmethod

class RuntimeMonitor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def monitor(self, observation, estimate):
        raise(NotImplementedError)
    
class OODDetector(RuntimeMonitor):


    def __init__(self, model_file, threshold):
        super(OODDetector, self).__init__()

        self.model, image_size = xplane_models.load_checkpoint(model_file)
        self.downsample = tform.Compose([
            tform.Resize(size=image_size),
            tform.Grayscale(),
            tform.ToTensor()
        ])
        self.estop_triggered = False
        self.threshold = threshold

    def monitor(self, observation, estimate):
        anomaly_score = self.get_anomaly_score(observation, estimate)
        if anomaly_score >= self.threshold or self.estop_triggered:
            self.estop_triggered = True
        return anomaly_score, self.estop_triggered
    
    @abstractmethod
    def get_anomaly_score(self, observation, estimate):
        raise(NotImplementedError)

class AutoEncoderMonitor(OODDetector):

    def __init__(self, model_file, threshold):
        super(AutoEncoderMonitor, self).__init__(model_file, threshold)

    def get_anomaly_score(self, observation, estimate):
        return torch.norm(self.downsample(observation) - self.model(self.downsample(observation))).detach().item()

        
