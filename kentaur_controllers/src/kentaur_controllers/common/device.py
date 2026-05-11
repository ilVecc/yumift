from typing_extensions import override, Tuple

import rospy
import numpy as np, quaternion as quat

from dynamicals.utils.geometry import Frame
from dynamicals.utils.jacobians import jacobian_change_base_frame
from dynamicals.common.devices import AbstractDevice, AbstractDeviceState, AbstractDeviceCommand

from sleipner_controllers.common.device import SleipnerCartesianDevice, SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand
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
        
        # yumi to sleipner transform
        self.sXy = Frame(
            position=np.array([-0.049, 0, 0.050]), 
            rotation=quat.from_rotation_vector([0, 0, np.deg2rad(180)]))
    
    def state_to_home(self, state : KentaurDeviceState) -> Tuple[Frame, Frame]:
        # base transform in home (i.e. sleipner's pose at startup, based on odometry)
        homeXs = self.device_sleipner.state_wrt_home(state.state_sleipner)
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
