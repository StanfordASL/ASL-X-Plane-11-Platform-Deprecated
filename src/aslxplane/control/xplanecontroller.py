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
            steering_params["bias"], # constant offset input reference to compensate for bias in the simulator
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

    def solve(self, state, estop=False):
        cte, he, speed = state
        rudder = self.steering_controller.get_input(cte + self.he_ratio * he)
        throttle = self.speed_controller.get_input(speed)

        if estop:
            rudder = 0
            throttle = 0

        return rudder, throttle
    

class SinusoidController(controllers.Controller):

    def __init__(self, steering_params, speed_params, dt):
        super(SinusoidController, self).__init__(
            dt, 
            [np.array([c1, c2]) for c1, c2 in zip(steering_params["input_constraints"], speed_params["input_constraints"])]
        )
        self.speed_controller = controllers.BangBang(
            speed_params["low_u"],
            speed_params["high_u"],
            speed_params["nominal_u"],
            speed_params["low_speed"],
            speed_params["high_speed"],
            speed_params["input_constraints"],
        )
        self.steering_params = steering_params
        self.cte_bias = None
        self.turn_gain = None
        self.he_limit = None
        self.rudder_bias = self.steering_params["bias"]

    def reset(self):
        self.speed_controller.reset()
        self.cte_bias = np.random.uniform(
            low=self.steering_params["cte_bias_range"][0],
            high=self.steering_params["cte_bias_range"][1]
        )
        self.he_limit = np.random.uniform(
            low=self.steering_params["he_limit_range"][0],
            high=self.steering_params["he_limit_range"][1]
        )
        self.turn_gain = self.steering_params["turn_min"] + self.steering_params["turn_gain"] * self.he_limit

    def solve(self, state):
        cte, he, speed = state
        throttle = self.speed_controller.get_input(speed)

        rudder = self.rudder_bias
        if he < self.he_limit and cte < self.cte_bias:
            rudder -= self.turn_gain
        elif he > - self.he_limit and cte > self.cte_bias:
            rudder += self.turn_gain
        
        return rudder, throttle

