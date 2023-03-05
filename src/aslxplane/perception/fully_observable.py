import sys
import os

import xpc3
import xpc3.xpc3_helper as xpc3_helper



def getStateFullyObservable(client):
    """ Returns the true crosstrack error (meters) and
        heading error (degrees) to simulate fully 
        oberservable control

        Args:
            client: XPlane Client
    """
    cte, _, he = xpc3_helper.getHomeState(client)
    return cte, he
