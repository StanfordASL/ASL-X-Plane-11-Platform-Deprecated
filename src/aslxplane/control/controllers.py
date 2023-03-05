import numpy as np

from abc import ABC, abstractmethod

class Controller(ABC):

    @abstractmethod
    def __init__(self, dt, input_constraints):
        self.dt = dt
        self.input_constraints = input_constraints
        self.state_reference = None
        self.input_reference = None

    @abstractmethod
    def reset(self):
        raise(NotImplementedError)

    @abstractmethod
    def solve(self, state):
        raise(NotImplementedError)

    def get_input(self, state):
        input = self.solve(state)
        return np.clip(input, self.input_constraints[0], self.input_constraints[1])

    def set_reference(self, state_reference, input_reference):
        self.state_reference = state_reference
        self.input_reference = input_reference

class XPlaneProportionalControl(Controller):

    def __init__(self, cte_gain, he_gain, input_constraints):
        super(XPlaneProportionalControl, self).__init__(None, input_constraints)
        self.cte_gain = cte_gain
        self.he_gain = he_gain

    def reset(self):
        return

    def solve(self, state):
        cte, he = state
        return self.input_reference + self.cte_gain * cte + self.he_gain * he
