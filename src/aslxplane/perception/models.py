import torch
import torch.nn as nn
import numpy as np

def get_encoder(params):
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(np.prod(params["image_size"]), params["hidden_layers"][0]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][0], params["hidden_layers"][1]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][1], 2)
    )
    return encoder

#     cnn = nn.Sequential(
#                 nn.Conv2d(1,5,5),
#                 nn.ReLU(),
#                 nn.Conv2d(5,5,5),
#                 nn.ReLU(),
#                 nn.Flatten(),
#                 nn.Linear(120 * 248 * 5, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1))
#     return cnn
