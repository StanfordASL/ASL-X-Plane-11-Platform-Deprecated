import numpy as np
import imgaug.augmenters as iaa
from abc import ABC, abstractmethod

class ImageCorruption(ABC):
    
    @abstractmethod
    def __init__(self):
        self.aug = None
    
    def add_corruption(self, img):
        assert(self.aug is not None)
        img = self.aug(image=img)
        return img
     
class Rain(ImageCorruption):

    def __init__(self, speed_scale=1, size_scale=1):
        super(Rain, self).__init__()
        aug = iaa.Rain(speed=(.1,.1) * speed_scale, drop_size=(.6,.4) * size_scale, seed=1)
        # self.aug = aug.to_deterministic()

    def __str__(self):
        return "Rain"


class Noise(ImageCorruption):

    def __init__(self, scale=3):
        super(Noise, self).__init__()
        self.aug = iaa.imgcorruptlike.GaussianNoise(severity=scale)

    def __str__(self):
        return "Noise"

class Motionblur(ImageCorruption):

    def __init__(self, scale=1, angle=90):
        super(Motionblur, self).__init__()
        self.aug = iaa.MotionBlur(k=100 * scale, angle=[angle])
    
    def __str__(self):
        return "Motion Blur"
    
class RainyBlur(ImageCorruption):

    def __init__(self, scale=1):
        super(RainyBlur, self).__init__()
        self.aug = iaa.Sequential([
            iaa.MotionBlur(k=100 * scale, angle=[30]),
            iaa.Rain(speed=(.1,.1) * scale, drop_size=(.3,.3) * scale, seed=1), #.to_deterministic(),
            ])
        
    def __str__(self):
        return "Rain and Motion Blur"
        
class Snow(ImageCorruption):

    def __init__(self, scale=2.5, thresh=140):
        super(Snow, self).__init__()
        self.aug = iaa.FastSnowyLandscape(
            lightness_threshold=thresh,
            lightness_multiplier=scale
            )
        
    def __str__(self):
        return "Snow"
