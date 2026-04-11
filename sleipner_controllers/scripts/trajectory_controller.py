#!/usr/bin/env python3
from abc import ABCMeta
from typing_extensions import override

import rospy

import numpy as np, quaternion as quat

from dynamicals.utils import Frame
from dynamicals.common.controllers import AbstractController
from dynamicals.impl import CartesianVelocityControlLaw
from pathfinder.polynomial import CubicPoseTrajectory, PoseParam

from sleipner_controllers.common.device import SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand, SleipnerCartesianDevice
from sleipner_controllers.common.controller_base import SE2TwistAction
from sleipner_controllers.misc.utils import SleipnerCartesianDeviceState_to_PoseParam, SleipnerCartesianDeviceState_to_Frame

from geometry_msgs.msg import PointStamped


class SleipnerCartesianTrajectoryController(
    AbstractController[SleipnerCartesianDeviceState, SE2TwistAction, SleipnerCartesianDeviceCommand], 
    metaclass=ABCMeta
):
    """ Velocity controller for Sleipner robot.
    """

    def __init__(self, robot_handle : SleipnerCartesianDevice):
        self._device : SleipnerCartesianDevice
        super().__init__(robot_handle)
        self.control_law = CartesianVelocityControlLaw(k_p=3, k_o=1, 
                                                       min_actionable_error=[0.05, 0.02],
                                                       max_allowed_deviation=[1.5, np.pi/2])
        self.trajectory = CubicPoseTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        rospy.Subscriber("/base/target", PointStamped, self._callback_received_target, queue_size=1)
        self.target_tra = np.zeros(3)
        self.target_rot = quat.one
        self.target_time = 0.0001
    
    @override
    def start(self):
        # TODO make this a default behaviour?
        rospy.loginfo("Controller will start up soon")
        self.reset(self.device_read())  # init trajectory
        super().start(250) # Hz
    
    @override
    def stop(self):
        rospy.loginfo("Controller is shutting down")
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
            self._device.send(SleipnerCartesianDeviceCommand())
            print(f"Sent stop command ({i+1}/{stop_commands})")
    
    def _callback_received_target(self, msg : PointStamped):
        rospy.loginfo(f"New target received: [{msg.point.x}, {msg.point.y}, {msg.point.z}]")
        # msg.header.
        self.target_tra = np.array([msg.point.x, msg.point.y, 0])
        self.target_rot = quat.from_rotation_vector(np.array([0, 0, msg.point.z]))
        self.target_time = 5.0
        # initialize new trajectory
        self.reset(self.device_read())
    
    @override
    def reset(self, state: SleipnerCartesianDeviceState):
        self.control_law.clear()
        
        param_init = SleipnerCartesianDeviceState_to_PoseParam(state)
        frame_init = SleipnerCartesianDeviceState_to_Frame(state)
        frame_final = Frame(self.target_tra, self.target_rot) @ frame_init
        param_final = PoseParam(frame_final.pos, frame_final.rot, np.zeros(6))
        
        self.trajectory.update(param_init, param_final, self.target_time)
        self.trajectory_initial_time = rospy.Time.now()
        
        with np.printoptions(precision=2):
            rospy.loginfo(f"Initial pose: {frame_init.pos} {quat.as_float_array(frame_init.rot)}")
            rospy.loginfo(f"Final pose:   {frame_final.pos} {quat.as_float_array(frame_final.rot)}")
    
    @override
    def fallback(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        return SE2TwistAction(np.zeros(3))
    
    @override
    def policy(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        
        real_now = rospy.Time.now()
        state_now: rospy.Time = state.time
        dt = (real_now - state_now).to_sec()
        traj_time = (real_now - self.trajectory_initial_time).to_sec()
        
        pose_now = SleipnerCartesianDeviceState_to_Frame(state)
        
        param_next = self.trajectory.compute(traj_time)
        pose_next = Frame(param_next.pos, param_next.rot, param_next.vel)
        
        vel = self.control_law.update_and_compute(pose_now, pose_next, dt)
        vel = np.array([vel[0], vel[1], vel[5]])
        
        return SE2TwistAction(vel)
    
    @override
    def solve_action(self, state: SleipnerCartesianDeviceState, action: SE2TwistAction) -> SleipnerCartesianDeviceCommand:
        return SleipnerCartesianDeviceCommand(action.twist_SE2)


if __name__ == "__main__":
    rospy.init_node("sleipner_cartesian_trajectory_controller", anonymous=False)
    
    sleipner = SleipnerCartesianDevice()
    controller = SleipnerCartesianTrajectoryController(sleipner)
    rospy.on_shutdown(controller.stop)
    
    controller.ready()
    controller.start()  # locking
