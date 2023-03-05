# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)

# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '../simulation')

import controllers
import fully_observable
import tiny_taxinet_pytorch as tiny_taxinet
# import tiny_taxinet

from corruptions import *

# Type of state estimation
# 'fully_observable' - true state is known
# 'tiny_taxinet'     - state is estimated using the tiny taxinet neural network from
#                      image observations of the true state
STATE_ESTIMATOR = 'tiny_taxinet'

## Type of image corruption to add
## Current options: None, Motion Blur, Rain, Rain + Blur, Noise, Snow

CORRUPTION = "None"

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""

# Directory to save output data
# NOTE: CSV file and images will be overwritten if already exists in that directory, but
# extra images (for time steps that do not occur in the new episodes) will not be deleted
OUT_DIR = "../../../Xplane-data-dir/conformal-shifts-data/taxinet-02-05-23/"

# # Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
# TIME_OF_DAY = 8.0

# Start and end of range of possible time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
# For each sinsoidal trajectory in the run, the time of day will be sampled uniformly
# from this range
TIME_OF_DAY_START = 8
TIME_OF_DAY_END = 8.1


# Cases to run (determines how other variables are set)
# example    - runs 2 short trajectories (used for initial testing)
# smallset   - runs 5 sinusoidal trajectories centered at zero crosstrack error with 
#              varying amplitude and frequency (ideal for collecting OoD data)
# largeset   - runs 20 sinusoidal trajectories with varying amplitude and frequency
#              and centered at different crosstrack errors
# validation - runs 5 sinusoidal trajectories centered at zero crosstrack error with 
#              varying amplitude and frequency
# test       - runs 3 sinusoidal trajectories center at 3 different crosstrack errors
# the last five trajectories of the largeset have the same parameter settings as
# the smallset

# Frequency with which to record data
# NOTE: this is approximate due to computational overhead
FREQUENCY = 5# Hz



"""
Other parameters
    - NOTE: you should not need to change any of these unless you want to create
    additional scenarios beyond the ones provided
"""
NUM_TRAJECTORIES = 1
# # Case indices to run (see getParams in sinusoidal.py for the specifics of each case)
# if case == 'example':
#     CASE_INDS = [18, 19]
# elif case == 'smallset':
#     CASE_INDS = [*range(15, 20)]
# elif case == 'largeset':
#     CASE_INDS = [*range(0, 20)]
# elif case == 'validation':
#     CASE_INDS = [*range(20, 25)]
# elif case == 'test':
#     CASE_INDS = [*range(25, 28)]
# else:
#     print('invalid case name, running the example set...')
#     CASE_INDS = [18, 19]

END_PERC = 20.0
START_PERC = 0.5
# Screenshot parameters
MONITOR = {'top': 100, 'left': 100, 'width': 3740, 'height': 2060}
# Width and height of final image
WIDTH = 360
HEIGHT = 200

# Dictionary of weather types (used when selecting a name for each output image)
CLOUD_TYPES = {0: 'clear', 1: 'cirrus', 2: 'scattered', 3: 'broken', 4: 'overcast'}
ANG_LIMIT_RANGE = [5,20] #[5,20]
CENTER_CTE_RANGE = [0,6]
TURN_GAIN_RANGE = [0.05, 0.15] #[0.05, 0.15]

# Simulation Constants
RUNWAY_LENGTH = 2982

# Frequency to get new control input 
# (e.g. if DT=0.5, CTRL_EVERY should be set to 20 to perform control at a 1 Hz rate)
CTRL_EVERY = 1.0

"""
Other parameters
    - NOTE: you should not need to change any of these unless you want to create
    additional scenarios beyond the ones provided
"""

# Tells simulator which proportional controller to use based on dynamics model
GET_CONTROL = controllers.getProportionalControl

if CORRUPTION == "None":
    corruption = None
elif CORRUPTION == "Noise":
    noise = Noise(scale=5)
    corruption = noise.add_noise
elif CORRUPTION == "Rain":
    rain = Rain()
    corruption = rain.add_rain
elif CORRUPTION == "Motion Blur":
    blur = Motionblur()
    corruption = blur.add_blur
elif CORRUPTION == "Rain and Motion Blur":
    blur = RainyBlur()
    corruption = blur.add_blur
elif CORRUPTION == "Snow":
    snow = Snow()
    corruption = snow.add_snow


# Tells simulator which function to use to estimate the state
if STATE_ESTIMATOR == 'tiny_taxinet':
    GET_STATE = lambda client: tiny_taxinet.getStateTinyTaxiNet(client, corruption=corruption)
elif STATE_ESTIMATOR == 'fully_observable':
    GET_STATE = fully_observable.getStateFullyObservable
else:
    print("Invalid state estimator name - assuming fully observable")
    GET_STATE = fully_observable.getStateFullyObservable

