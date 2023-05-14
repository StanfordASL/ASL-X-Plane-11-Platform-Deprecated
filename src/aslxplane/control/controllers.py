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

    def get_input(self, state, estop=False):
        input = self.solve(state, estop=estop)
        return np.clip(input, self.input_constraints[0], self.input_constraints[1])

    def set_reference(self, state_reference, input_reference):
        self.state_reference = state_reference
        self.input_reference = input_reference

class PID(Controller):
	"""
	SISO PID controller that tracks a state and input reference, with anti-windup on the input.
	"""
	def __init__(self, P, I, D, state_reference, input_reference, dt, input_constraints):
		super(PID, self).__init__(dt, input_constraints)
		self.Kp = P
		self.Ki = I
		self.Kd = D
		self.state_reference = state_reference
		self.input_reference = input_reference

		self.e  = None
		self.de = None
		self.ei = None

	def reset(self):
		self.e  = 0
		self.de = 0
		self.ei = 0

	def solve(self, x, estop=False):
		e = self.state_reference - x
		de = (e - self.e) / self.dt
		ei = self.ei + e * self.dt

		u = self.Kp * e + self.Kd * de + self.Ki * ei + self.input_reference

		self.e = e
		self.de = de

		if u < self.input_constraints[0]:
			return self.input_constraints[0]
		elif u > self.input_constraints[1]:
			return self.input_constraints[1]
		else:
			self.ei = ei
			return u

class BangBang(Controller):
      
    def __init__(self, low_u, high_u, nominal_u, low_x, high_x, input_constraints):
        super(BangBang, self).__init__(None, input_constraints)
        self.low_u = low_u
        self.high_u = high_u
        self.nominal_u = nominal_u
        self.low_x = low_x
        self.high_x = high_x
    
    def reset(self):
        pass

    def solve(self, x, estop=False):
        if self.low_x > x:
            return self.high_u
        elif self.high_x < x:
            return self.low_u
        else:
            return self.nominal_u
