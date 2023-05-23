import torch
import numpy as np
import torch.nn as nn
import xpc3.xpc3 as xpc3
import xpc3.xpc3_helper as xpc3_helper
import torchvision.transforms as tform
import aslxplane.perception.models as xplane_models
from abc import ABC, abstractmethod


class Estimator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_estimate(self, observation):
        raise(NotImplementedError)
    
class GroundTruthEstimator(Estimator):

    def __init__(self, xplane):
        super(GroundTruthEstimator, self).__init__()
        self.xplane = xplane


    def get_estimate(self, observation):
        """ Returns the true crosstrack error (meters) and
        heading error (degrees) to simulate fully 
        oberservable control

        Args:
            observation: throwaway argument, queries ground truth from xpc3 client
        """
        cte, he, dtp, speed = self.xplane.get_ground_truth_state()
        return cte, he

class TaxiNet(Estimator):

    def __init__(self, model_file, normalization_constants):
        self.model, image_size = xplane_models.load_checkpoint(model_file)
        self.downsample = tform.Compose([
            tform.Resize(size=image_size),
            tform.Grayscale(),
            tform.ToTensor()
        ])
        self.cte_norm_const = normalization_constants["cte_constant"]
        self.he_norm_const = normalization_constants["he_constant"]

    def get_estimate(self, observation):
        img = self.downsample(observation)
        with torch.no_grad():
            pred = self.model(img.unsqueeze(0)).squeeze()
        cte = pred[0].item() * self.cte_norm_const
        he = pred[1].item() * self.he_norm_const
        return cte, he

