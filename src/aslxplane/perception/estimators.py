import torch
import torch.nn as nn
import torchvision.transforms as tform
import numpy as np

from abc import ABC, abstractmethod
import xpc3.xpc3 as xpc3
import xpc3.xpc3_helper as xpc3_helper


class Estimator:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_estimate(self, observation):
        raise(NotImplementedError)
    
class GroundTruthEstimator(Estimator):

    def __init__(self, client):
        super(GroundTruthEstimator, self).__init__()
        self.client = client

    @staticmethod
    def get_estimate(self, observation):
        """ Returns the true crosstrack error (meters) and
        heading error (degrees) to simulate fully 
        oberservable control

        Args:
            observation: throwaway argument, queries ground truth from xpc3 client
        """
        cte, dtp, he = xpc3_helper.getHomeState(self.client)
        return cte, dtp, he

IMG_SIZE = (128,256)

class TaxiNet(Estimator):

    def __init__(self, model_file):
        self.model = get_encoder()
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        self.downsample = tform.Compose([
            tform.Resize(size=IMG_SIZE),
            tform.Grayscale(),
            tform.ToTensor()
        ])
        self.cte_norm_const = 10
        self.he_norm_const = 30

    def get_estimate(self, observation):
        img = self.downsample(observation)
        with torch.no_grad():
            pred = self.model(img).squeeze()
        cte = pred[0].item() * self.cte_norm_const
        he = pred[1].item() * self.he_norm_const
        return cte, he

def get_encoder():

    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(np.prod(IMG_SIZE), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    return encoder
