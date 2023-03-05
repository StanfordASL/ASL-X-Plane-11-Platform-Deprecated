# from nnet import *
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tform

import numpy as np
import time
import matplotlib.pyplot as plt

import mss
import cv2
import os

# Read in the network
filename = "../../models/taxinet03-02_14-53.nn"
img_size = (128,256)

def get_encoder():

    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(np.prod(img_size), 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    return encoder

downsample = tform.Compose([
    tform.Resize(size=img_size),
    tform.Grayscale(),
    tform.ToTensor()
])

network = get_encoder()
network.load_state_dict(torch.load(filename))
network.eval()

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16             # Size of square of pixels downsampled to one grayscale value
# During downsampling, average the numPix brightest pixels in each square
numPix = 16
width = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image

screenShot = mss.mss()
monitor = {'top': 100, 'left': 100, 'width': 3740, 'height': 2060}
screen_width = 360  # For cropping
screen_height = 200  # For cropping

def getCurrentImage(corruption=None, debug=False):
    """ Returns a downsampled image of the current X-Plane 11 image
        compatible with the TinyTaxiNet neural network state estimator

        NOTE: this is designed for screens with 1920x1080 resolution
        operating X-Plane 11 in full screen mode - it will need to be adjusted
        for other resolutions
    """
    # Get current screenshot
    img = cv2.cvtColor(np.array(screenShot.grab(monitor)),
                       cv2.COLOR_BGRA2BGR)[230:, :, :]
    img = cv2.resize(img, (screen_width, screen_height))
    img = img[:, :, ::-1]
    
    img = np.array(img)
    
    if corruption:
        img = corruption(img)
        img = np.array(img)

    if debug:
        plt.imshow(img)
        plt.show()
    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
    # values range between 0 and 1
    
    img = Image.fromarray(img)
    input = downsample(img)

    if debug:
        plt.imshow(input.numpy().squeeze(0), cmap="gray")
        plt.show()

    return input

def getStateTinyTaxiNet(client, corruption=None):
    """ Returns an estimate of the crosstrack error (meters)
        and heading error (degrees) by passing the current
        image through TinyTaxiNet

        Args:
            client: XPlane Client
    """
    image = getCurrentImage(corruption=corruption)
    with torch.no_grad():
        pred = network(image).squeeze()
    cte = pred[0].item() * 10
    he = pred[1].item() * 30

    return cte, he
