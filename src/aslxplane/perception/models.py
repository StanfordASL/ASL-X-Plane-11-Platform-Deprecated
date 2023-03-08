import torch
import datetime
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

def save_checkpoint(model_params, model, save_dir):
    assert(next(model.parameters()).is_cuda == False)
    model_file = save_dir + "taxinet" + datetime.datetime.now().strftime("%m-%d_%H-%M") + ".pt"
    checkpoint = {"model_state_dict": model.state_dict()}
    checkpoint.update(model_params)
    torch.save(checkpoint, model_file)

def load_checkpoint(model_file):
    checkpoint = torch.load(model_file)
    model = None
    if checkpoint["type"] == "encoder":
        model_params = {k:v for k, v in checkpoint.items() if k in ["image_size", "hidden_layers"]}
        model = get_encoder(model_params)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return model, checkpoint["image_size"]

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
