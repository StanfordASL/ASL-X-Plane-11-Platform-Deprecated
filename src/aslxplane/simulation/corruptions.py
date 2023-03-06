import numpy as np
import imgaug.augmenters as iaa

class Rain:

    def __init__(self, speed_scale=1, size_scale=1):
        aug = iaa.Rain(speed=(.1,.1) * speed_scale, drop_size=(.6,.4) * size_scale, seed=1)
        self.aug = aug.to_deterministic()


    def add_rain(self, img):
        return self.aug(image=img)

class Noise:

    def __init__(self, scale=3):
        self.aug = iaa.imgcorruptlike.GaussianNoise(severity=scale)

    def add_noise(self, img):
        # import pdb; pdb.set_trace()
        img = self.aug(image=img)
        return img


class Motionblur:

    def __init__(self, scale=1, angle=90) -> None:
        self.aug = iaa.MotionBlur(k=100 * scale, angle=[angle])

    def add_blur(self, img):
        # import pdb; pdb.set_trace()
        img = self.aug(image=img)
        return img

class RainyBlur:

    def __init__(self, scale=1) -> None:
        self.aug = iaa.Sequential([
            iaa.MotionBlur(k=100 * scale, angle=[30]),
            iaa.Rain(speed=(.1,.1) * scale, drop_size=(.3,.3) * scale, seed=1).to_deterministic(),
            ])
        

    def add_blur(self, img):
        # import pdb; pdb.set_trace()
        img = self.aug(image=img)
        return img
    

class Snow:

    def __init__(self, scale=2.5, thresh=140) -> None:
        self.aug = iaa.FastSnowyLandscape(
            lightness_threshold=thresh,
            lightness_multiplier=scale
            )

    def add_snow(self, img):
        img = self.aug(image=img)
        return img  
