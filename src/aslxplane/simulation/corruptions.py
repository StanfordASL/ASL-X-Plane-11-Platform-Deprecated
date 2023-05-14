import numpy as np
import imgaug.augmenters as iaa
from abc import ABC, abstractmethod

class ImageCorruption(ABC):
    
    @abstractmethod
    def __init__(self, transient_range=None):
        self.aug = None
        self.transient_range = transient_range
        self.t = 0
    
    def add_corruption(self, img):
        # assert(self.aug is not None)
        if self.transient_range is not None:
            if self.t >= self.transient_range[0] and self.t <= self.transient_range[1]:
                img = self.aug(image=img)
        else:
            img = self.aug(image=img)

        self.t += 1
        return img
     
class Rain(ImageCorruption):

    def __init__(self, speed_scale=1, size_scale=1, transient_range=None):
        super(Rain, self).__init__(transient_range=transient_range)
        self.aug = iaa.Rain(speed=np.array([.1,.1]) * speed_scale, drop_size=np.array([.3,.3]) * size_scale, seed=1)

    def __str__(self):
        return "Rain"


class Noise(ImageCorruption):

    def __init__(self, scale=3, transient_range=None):
        super(Noise, self).__init__(transient_range=transient_range)
        self.aug = iaa.imgcorruptlike.GaussianNoise(severity=scale)

    def __str__(self):
        return "Noise"

class Motionblur(ImageCorruption):

    def __init__(self, scale=1, angle=90,transient_range=None):
        super(Motionblur, self).__init__(transient_range=transient_range)
        self.aug = iaa.MotionBlur(k=100 * scale, angle=[angle])
    
    def __str__(self):
        return "Motion Blur"
    
class RainyBlur(ImageCorruption):

    def __init__(self, scale=1, transient_range=None):
        super(RainyBlur, self).__init__(transient_range=transient_range)
        self.aug = iaa.Sequential([
            iaa.MotionBlur(k=100 * scale, angle=[30]),
            iaa.Rain(speed=(.1,.1) * scale, drop_size=(.3,.3) * scale, seed=1), #.to_deterministic(),
        ])
        
    def __str__(self):
        return "Rain and Motion Blur"
        
class Snow(ImageCorruption):

    def __init__(self, scale=2.5, thresh=140, transient_range=None):
        super(Snow, self).__init__(transient_range=transient_range)
        self.aug = iaa.FastSnowyLandscape(
            lightness_threshold=thresh,
            lightness_multiplier=scale
        )
        
    def __str__(self):
        return "Snow"

class RainySnow(ImageCorruption):

    def __init__(self, 
                snow_scale=2.5, 
                snow_thresh=100, 
                rain_speed_scale=1, 
                rain_size_scale=1, 
                num_buildup_steps=5, 
                transient_range=None
            ):
        super(RainySnow, self).__init__(transient_range=transient_range)
        self.snow_scale = snow_scale
        self.snow_thresh = snow_thresh
        self.rain_speed_scale = rain_speed_scale
        self.rain_size_scale = rain_size_scale
        self.num_buildup_steps = num_buildup_steps

        self.rain_aug = iaa.Rain(speed=(.1,.1) * self.rain_speed_scale, drop_size=(.3,.3) * self.rain_size_scale, seed=1)
        self.augs = [
            iaa.Sequential([
                iaa.FastSnowyLandscape(
                    lightness_threshold=self.snow_thresh, 
                    lightness_multiplier= 1 + (self.snow_scale - 1) / self.num_buildup_steps * t_ref
                ),
                self.rain_aug, #.to_deterministic(),
            ])
            for t_ref in range(0, self.num_buildup_steps + 1)
        ]
        self.aug = self.augs[0]

    def add_corruption(self, img):
        t_ref = max(min(self.t - self.transient_range[0], self.num_buildup_steps), 0)
        self.aug = self.augs[t_ref]
        return super().add_corruption(img)
    
    def __str__(self):
        return "Snowing"
