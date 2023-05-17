import numpy as np
import imgaug.augmenters as iaa
from abc import ABC, abstractmethod
    
class ImageCorruption(ABC):
    
    @abstractmethod
    def __init__(self, transient_range=None, num_buildup_steps=None):
        self.aug = None
        self.augs = None
        self.transient_range = transient_range
        self.num_buildup_steps = num_buildup_steps
        self.t = 0

    @abstractmethod
    def get_corruption(self, severity):
        raise(NotImplementedError)
    
    def build(self, severity):
        if self.transient_range and self.num_buildup_steps:
            self.augs = [self.get_corruption(severity / self.num_buildup_steps * t_ref) for t_ref in range(0, self.num_buildup_steps + 1)]
        else:
            self.aug = self.get_corruption(severity)
    
    def add_corruption(self, img):

        if self.transient_range:
            if self.t >= self.transient_range[0] and self.t <= self.transient_range[1]:
                if self.num_buildup_steps:
                    t_ref = max(min(self.t - self.transient_range[0], self.num_buildup_steps), 0)
                    img = self.augs[t_ref](image=img)
                else:
                    img = self.aug(image=img)
        else:
            img = self.aug(image=img)

        self.t += 1
        return img
     
class Rain(ImageCorruption):

    def __init__(self, severity=1, transient_range=None, num_buildup_steps=None):
        super(Rain, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.severity = severity
        self.build(self.severity)

    def get_corruption(self, severity):
        aug = iaa.Rain(speed=np.array([.1,.1]), drop_size=np.array([.3,.3]) * severity + 1e-3, seed=1)
        return aug

    def __str__(self):
        return "Rain"


class Noise(ImageCorruption):

    def __init__(self, severity=3, transient_range=None, num_buildup_steps=None):
        super(Noise, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.severity=severity
        self.build(self.severity - 1)

    def get_corruption(self, severity):
        return iaa.imgcorruptlike.GaussianNoise(severity=int(severity) + 1)

    def __str__(self):
        return "Noise"

class Motionblur(ImageCorruption):

    def __init__(self, severity=1, angle=90, transient_range=None, num_buildup_steps=None):
        super(Motionblur, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.severity = severity
        self.angle = angle
        self.build(self.severity)
    
    def get_corruption(self, severity):
        aug = iaa.MotionBlur(k=int(100 * severity) + 3, angle=[self.angle])
        return aug
    
    def __str__(self):
        return "Motion Blur"
    
class RainyBlur(ImageCorruption):

    def __init__(self, severity=1, transient_range=None, num_buildup_steps=None):
        super(RainyBlur, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.severity = severity
        self.build(self.severity)

    def get_corruption(self, severity):
        aug = iaa.Sequential([
            iaa.MotionBlur(k=int(100 * severity) + 3, angle=[30]),
            iaa.Rain(speed=(.1,.1), drop_size=np.array([.3,.3]) * severity + 1e-3, seed=1), #.to_deterministic(),
        ])
        return aug
        
    def __str__(self):
        return "Rain and Motion Blur"
        
class Snow(ImageCorruption):

    def __init__(self, severity=2.5, thresh=140, transient_range=None, num_buildup_steps=None):
        super(Snow, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.severity = severity
        self.thresh = thresh
        self.build(self.severity - 1)

    def get_corruption(self, severity):
        aug = iaa.FastSnowyLandscape(
            lightness_threshold=self.thresh,
            lightness_multiplier=severity + 1
        )
        return aug
        
    def __str__(self):
        return "Snow"

class RainySnow(ImageCorruption):

    def __init__(self, 
                snow_scale=2.5, 
                snow_thresh=100, 
                rain_speed_scale=1, 
                rain_size_scale=1, 
                num_buildup_steps=None, 
                transient_range=None
            ):
        super(RainySnow, self).__init__(transient_range=transient_range, num_buildup_steps=num_buildup_steps)
        self.snow_scale = snow_scale
        self.snow_thresh = snow_thresh
        self.rain_speed_scale = rain_speed_scale
        self.rain_size_scale = rain_size_scale

        self.rain_aug = iaa.Rain(speed=(.1,.1) * self.rain_speed_scale, drop_size=(.3,.3) * self.rain_size_scale, seed=1)
        self.build(snow_scale - 1)

    def get_corruption(self, severity):
        aug = iaa.Sequential([
            iaa.FastSnowyLandscape(
                lightness_threshold=self.snow_thresh, 
                lightness_multiplier= 1 + severity
            ),
            self.rain_aug, #.to_deterministic(),
        ])
        return aug
    
    def __str__(self):
        return "Snowing"
