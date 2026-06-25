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
        
        # yumi to sleipner surface transform
        self.sXy = Frame(
            position=np.array([-0.390, 0, 0.500]), 
            rotation=quat.from_rotation_vector([0, 0, np.deg2rad(180)]))
        # cache yumi to odometry transform
        self.odomXyumi = self.device_sleipner.odomXs @ self.sXy
    
    def robots_wrt_home(self, state : KentaurDeviceState) -> Tuple[Frame, Frame]:
        # base transform in home (i.e. sleipner's pose at startup, based on odometry)
        homeXodom = self.device_sleipner.state_wrt_home(state.state_sleipner)
        homeXyumi = homeXodom @ self.odomXyumi
        return homeXodom, homeXyumi
    
    def grippers_wrt_home(self, state : KentaurDeviceState) -> Tuple[Frame, Frame]:
        _, homeXy = self.robots_wrt_home(state)
        homeX_r = homeXy @ state.state_yumi.pose_gripper_r
        homeX_l = homeXy @ state.state_yumi.pose_gripper_l
        return homeX_r, homeX_l
    
    def jacobian_to_home(self, state : KentaurDeviceState, alpha : float = 2.75):
        """ alpha > 1 is more yumi
            alpha < 1 is more sleipner
        """
        homeXodom, homeXyumi = self.robots_wrt_home(state)
        
        # task jacobian
        homeJy_r = jacobian_change_base_frame(homeXyumi.rot, state.state_yumi.jacobian_gripper_r)
        homeJs_r = jacobian_change_base_frame(homeXodom.rot, state.state_sleipner.jacobian_SE3)
        homeJy_l = jacobian_change_base_frame(homeXyumi.rot, state.state_yumi.jacobian_gripper_l)
        homeJs_l = homeJs_r
        
        beta = 1/alpha
        homeJ = np.zeros((6+6,7+7+3))
        # right arm
        homeJ[ 0:6, 0:7 ] = alpha * homeJy_r
        homeJ[ 0:6,14:17] = beta  * homeJs_r
        # left arm
        homeJ[6:12, 7:14] = alpha * homeJy_l
        homeJ[6:12,14:17] = beta  * homeJs_l
        
        return homeJ
    
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
