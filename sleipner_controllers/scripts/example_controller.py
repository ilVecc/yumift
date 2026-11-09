#!/usr/bin/env python3
from typing_extensions import override

import rospy

from dynamicals.impl import AbstractROSController, CartesianVelocityControlLaw
from sleipner_controllers.common import SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand, SleipnerCartesianDevice, SE2TwistAction
from sleipner_controllers.common.parameters import ControllerParameters


class DummySleipnerController(
    AbstractROSController[SleipnerCartesianDeviceState, SE2TwistAction, SleipnerCartesianDeviceCommand]
):
    """ SE2 twist controller for the Sleipner mobile robot.
    """

    def __init__(self, sleipner_device : SleipnerCartesianDevice):
        super().__init__(sleipner_device)
        self._device : SleipnerCartesianDevice
        self.control_law = CartesianVelocityControlLaw(k_p=1, k_o=1)
        # Move 2 meters forward from the starting pose (home).
        # Use `self.device_read().to_Frame()` if you prefer to use poses in the 
        # odometry reference frame (no homing).
        self.target = self._device.state_wrt_home()  
        self.target.pos[0] += 2
    
    @override
    def reset(self, state: SleipnerCartesianDeviceState):
        self.control_law.clear()
    
    @override
    def policy(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        # take the delta between the current wall time and the time of the robot state
        dt = ControllerParameters.dt  # self.dt(state.time)
        # convert the robot state to a SE3 frame
        pose_now = self._device.state_wrt_home()
        # compute the velocity to bring the current pose to the target
        vel = self.control_law.update_and_compute(pose_now, self.target, dt)
        # finally, convert the SE3 twist in a SE2 twist
        return SE2TwistAction.from_twist_SE3(vel)
    
    @override
    def solve_action(self, state: SleipnerCartesianDeviceState, action: SE2TwistAction) -> SleipnerCartesianDeviceCommand:
        return SleipnerCartesianDeviceCommand(action.twist_SE2)


if __name__ == "__main__":
    rospy.init_node("sleipner_dummy_controller", anonymous=False)
    
    sleipner = SleipnerCartesianDevice()
    controller = DummySleipnerController(sleipner)
    
    controller.ready()
    controller.start(ControllerParameters.update_freq)  # locking
