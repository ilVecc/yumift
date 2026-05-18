#!/usr/bin/env python3
from typing_extensions import override

import rospy

from dynamicals.impl import AbstractROSController, CartesianVelocityControlLaw
from sleipner_controllers.common import SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand, SleipnerCartesianDevice, SE2TwistAction


class DummySleipnerController(
    AbstractROSController[SleipnerCartesianDeviceState, SE2TwistAction, SleipnerCartesianDeviceCommand]
):
    """ SE2 twist controller for the Sleipner mobile robot.
    """

    def __init__(self, sleipner_device : SleipnerCartesianDevice):
        super().__init__(sleipner_device)
        self.control_law = CartesianVelocityControlLaw(k_p=1, k_o=1)
        # move 2 meters forward
        self.target = self.device_read().to_Frame()
        self.target.pos[0] += 2
    
    @override
    def reset(self, state: SleipnerCartesianDeviceState):
        self.control_law.clear()
    
    @override
    def policy(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        # take the delta between the current wall time and the time of the robot state
        dt = self.dt(state.time)
        # convert the robot state to a SE3 frame
        pose_now = state.to_Frame()
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
    controller.start(250)  # locking
