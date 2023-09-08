import socket
from functools import partial

import xpc

class RobustXPlaneConnect:
    """An XPlaneConnect wrapper that automatically reconnects if the socket breaks."""

    def __init__(self):
        self.xp = xpc.XPlaneConnect(timeout=100.0)
        self.sendCTRL = partial(self.wrap_call, "sendCTRL")
        self.getDREF = partial(self.wrap_call, "getDREF")
        self.getDREFs = partial(self.wrap_call, "getDREFs")
        self.sendDREF = partial(self.wrap_call, "sendDREF")
        self.sendDREFs = partial(self.wrap_call, "sendDREFs")
        self.getPOSI = partial(self.wrap_call, "getPOSI")
        self.sendPOSI = partial(self.wrap_call, "sendPOSI")
        self.sendUDP = partial(self.wrap_call, "sendUDP")
        self.sendVIEW = partial(self.wrap_call, "sendVIEW")

    def wrap_call(self, fn_name, *args, **kwargs):
        while True:
            try:
                ret = getattr(self.xp, fn_name)(*args, **kwargs)
                return ret
            except socket.timeout:
                self.xp.socket.close()
                self.xp = xpc.XPlaneConnect(timeout=100.0)

    def close(self):
        self.xp.close()