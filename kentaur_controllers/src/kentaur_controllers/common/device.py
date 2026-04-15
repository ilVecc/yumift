from typing_extensions import override, Tuple

import numpy as np, quaternion as quat

from dynamicals.utils.geometry import Frame
from dynamicals.utils.jacobians import jacobian_change_base_frame
from dynamicals.common.devices import AbstractDevice, AbstractDeviceState, AbstractDeviceCommand

from sleipner_controllers.common.device import SleipnerCartesianDevice, SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand
from sleipner_controllers.misc.utils import SleipnerCartesianDeviceState_to_Frame
from yumift_controllers.common.device import YumiDevice, YumiDualDeviceCommand, YumiDualDeviceState



class KentaurDeviceState(AbstractDeviceState):
    def __init__(self, state_yumi = YumiDualDeviceState(), state_sleipner = SleipnerCartesianDeviceState()) -> None:
        super().__init__()
        self.state_yumi = state_yumi
        self.state_sleipner = state_sleipner

class KentaurDeviceCommand(AbstractDeviceCommand):
    def __init__(self, command_yumi = YumiDualDeviceCommand(), command_sleipner = SleipnerCartesianDeviceCommand()) -> None:
        super().__init__()
        self.command_yumi = command_yumi
        self.command_sleipner = command_sleipner

class KentaurDevice(AbstractDevice[KentaurDeviceState, KentaurDeviceCommand]):
    def __init__(self):
        super().__init__()
        self.device_yumi = YumiDevice()
        self.device_sleipner = SleipnerCartesianDevice()
        
        # initial pose in world (avoids recurrent use of the "world" frame, relying on old odometry and thus possibly very biased)
        self.wX_home = SleipnerCartesianDeviceState_to_Frame(self.device_sleipner.read())
        self.homeXw = self.wX_home.inv()
        print("HOME", self.homeXw)
        
        # yumi to base transform
        alpha = np.deg2rad(180)
        self.sT_y = np.array([-0.049, 0, 0.050])
        self.sXy = Frame(self.sT_y, quat.from_rotation_vector(np.array([0, 0, alpha])))
        self.sRy = quat.as_rotation_matrix(self.sXy.rot)
        
        self.wJ_s = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]])
        self.homeJ_s = jacobian_change_base_frame(self.homeXw.rot, self.wJ_s)
    
    def state_to_home(self, state : KentaurDeviceState) -> Tuple[Frame, Frame]:
        # base transform in home (i.e. sleipner's pose at startup, based on odometry)
        wXs = SleipnerCartesianDeviceState_to_Frame(state.state_sleipner)
        homeXs = self.homeXw @ wXs
        homeXy = homeXs @ self.sXy
        return homeXs, homeXy
    
    @override
    def reset(self) -> bool:
        return self.device_yumi.reset() and self.device_sleipner.reset()
    
    @override
    def is_ready(self) -> bool:
        return self.device_yumi.is_ready() and self.device_sleipner.is_ready()
    
    @override
    def read(self) -> KentaurDeviceState:
        return KentaurDeviceState(self.device_yumi.read(), self.device_sleipner.read())
    
    @override
    def send(self, command: KentaurDeviceCommand):
        self.device_yumi.send(command.command_yumi)
        self.device_sleipner.send(command.command_sleipner)
