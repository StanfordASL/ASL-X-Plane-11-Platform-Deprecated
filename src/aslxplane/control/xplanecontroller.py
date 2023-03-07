import numpy as np
import aslxplane.control.controllers as controllers

class XPlaneController(controllers.Controller):

    def __init__(self, steering_params, speed_params, dt):
        super(XPlaneController, self).__init__(
            dt, 
            [np.array([c1, c2]) for c1, c2 in zip(steering_params["input_constraints"], speed_params["input_constraints"])]
            )
        self.he_ratio = steering_params["he_ratio"]
        self.steering_controller = controllers.PID(
            steering_params["P"],
            steering_params["I"],
            steering_params["D"],
            0,
            0.008, # constant offset input reference to compensate for bias in the simulator
            dt,
            steering_params["input_constraints"]
            )
        self.speed_controller = controllers.BangBang(
            speed_params["low_u"],
            speed_params["high_u"],
            speed_params["nominal_u"],
            speed_params["low_speed"],
            speed_params["high_speed"],
            speed_params["input_constraints"],
        )
        
    def reset(self):
        self.steering_controller.reset()
        self.speed_controller.reset()

    def solve(self, state):
        cte, he, speed = state
        rudder = self.steering_controller.get_input(cte + self.he_ratio * he)
        throttle = self.speed_controller.get_input(speed)
        return rudder, throttle
    