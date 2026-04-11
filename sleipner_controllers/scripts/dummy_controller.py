#!/usr/bin/env python3
from abc import ABCMeta
from typing_extensions import override

import rospy
import numpy as np

from dynamicals.common.controllers import AbstractController
from dynamicals.impl import CartesianVelocityControlLaw

from sleipner_controllers.common.device import SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand, SleipnerCartesianDevice
from sleipner_controllers.common.controller_base import SE2TwistAction
from sleipner_controllers.misc.utils import SleipnerCartesianDeviceState_to_Frame


class DummySleipnerController(
    AbstractController[SleipnerCartesianDeviceState, SE2TwistAction, SleipnerCartesianDeviceCommand], 
    metaclass=ABCMeta
):
    """ Cartesian 2D twist controller for the Sleipner mobile robot.
    """

    def __init__(self, sleipner_device : SleipnerCartesianDevice):
        super().__init__(sleipner_device)
        self.control_law = CartesianVelocityControlLaw(1, 1, None, np.array([1, 1]))
        self.target = SleipnerCartesianDeviceState_to_Frame(self.device_read())
        self.target.pos[0] += 2  # move 2 meters in the x-axis (forward)
    
    @override
    def start(self):
        self.reset(self.device_read())  # init trajectory
        super().start(250) # Hz
    
    @override
    def stop(self):
        print("Controller shutting down")
        super().stop()
    
    @override
    def spin(self, control_rate: float):
        """ ROS implementation of the original function.
        """
        rate = rospy.Rate(control_rate)
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()
        # when the controller is shut down, send a stop command
        stop_commands = 3
        for i in range(stop_commands):
            self.device_send(SleipnerCartesianDeviceCommand())
            print(f"Sent stop command ({i+1}/{stop_commands})")
    
    @override
    def reset(self, state: SleipnerCartesianDeviceState):
        self.control_law.clear()
    
    @override
    def fallback(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        return SE2TwistAction(np.zeros(3))
    
    @override
    def policy(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        
        real_now = rospy.Time.now()
        state_now: rospy.Time = state.time
        dt = (real_now - state_now).to_sec()
        
        pose_now = SleipnerCartesianDeviceState_to_Frame(state)
        
        vel = self.control_law.update_and_compute(pose_now, self.target, dt)
        vel = np.array([vel[0], vel[1], vel[5]])
      
        return SE2TwistAction(vel)
    
    @override
    def solve_action(self, state: SleipnerCartesianDeviceState, action: SE2TwistAction) -> SleipnerCartesianDeviceCommand:
        return SleipnerCartesianDeviceCommand(action.twist_SE2)


if __name__ == "__main__":
    rospy.init_node("sleipner_cartesian_trajectory_controller", anonymous=False)
    
    sleipner = SleipnerCartesianDevice()
    controller = DummySleipnerController(sleipner)
    rospy.on_shutdown(controller.stop)
    
    controller.ready()
    controller.start()  # locking
