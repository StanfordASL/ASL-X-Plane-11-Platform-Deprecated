import torch
import datetime
import torch.nn as nn
import numpy as np

def get_encoder(params, num_outputs=2):
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(np.prod(params["image_size"]), params["hidden_layers"][0]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][0], params["hidden_layers"][1]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][1], num_outputs)
    )
    return encoder

def get_decoder(params):
    decoder = nn.Sequential(
        nn.Linear(params["hidden_layers"][-1], params["hidden_layers"][1]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][1], params["hidden_layers"][0]),
        nn.ReLU(),
        nn.Linear(params["hidden_layers"][0], np.prod(params["image_size"])),
        nn.Sigmoid(),
        nn.Unflatten(1, params["image_size"])
    )
    return decoder

def get_conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return conv_block

class SimpleConvNet(nn.Module):

    def __init__(self, params, num_outputs=2):
        super().__init__()
        self.conv_layers = nn.Sequential(*[
            get_conv_block(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride
            ) for in_channels, out_channels, kernel_size, stride in
            zip(
                params["channel_sizes"][:-1], 
                params["channel_sizes"][1:], 
                params["kernel_sizes"], 
                params["strides"]
            )
        ])
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(params["output_size"], num_outputs)
        self.params = params

    def forward(self, x):
        conv_output = self.conv_layers(x)
        # xx = x 
        # for conv in self.conv_layers:
        #     xx = conv(xx)
        #     print(xx.shape)
        flattened_conv_output = self.flatten(conv_output)
        # print(flattened_conv_output.shape)
        output = self.output_layer(flattened_conv_output)
        return output

class AutoEncoder(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.encoder = get_encoder(params, num_outputs=params["hidden_layers"][-1])
        self.decoder = get_decoder(params)
        self.params = params

    def get_latent(self, x):
        encoding = self.encoder(x)
        latent_representation = encoding
        return latent_representation

    def forward(self, x):
        latent_representation = self.get_latent(x)
        decoding = self.decoder(latent_representation)
        return decoding

def save_checkpoint(model_params, model, save_dir, model_title="taxinet"):
    assert(next(model.parameters()).is_cuda == False)
    model_file = save_dir + model_title + datetime.datetime.now().strftime("%m-%d_%H-%M") + ".pt"
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
    elif checkpoint["type"] == "auto-model":
        model_params = {k:v for k, v in checkpoint.items() if k in ["image_size", "hidden_layers"]}
        model = AutoEncoder(model_params)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    elif checkpoint["type"] == "simple_convnet":
        model_params = {k:v for k, v in checkpoint.items() if k in ["image_size", "kernel_sizes", "strides", "channel_sizes", "output_size"]}
        model = SimpleConvNet(model_params)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return model, checkpoint["image_size"]

