#!/usr/bin/env python3
from abc import ABCMeta
from typing_extensions import override

import rospy

import numpy as np, quaternion as quat

from dynamicals.utils import Frame
from dynamicals.common.devices import AbstractDevice, AbstractDeviceState, AbstractDeviceCommand
from dynamicals.common.controllers import AbstractControllerAction, AbstractController
from dynamicals.impl import CartesianVelocityControlLaw

from pathfinder.polynomial import CubicPoseTrajectory, PoseParam

from sleipner_IROS26.drivers import Sleipner8DOFDeviceState

from geometry_msgs.msg import Twist as TwistMsg
from sensor_msgs.msg import JointState as JointStateMsg
from nav_msgs.msg import Odometry as OdometryMsg


class Parameters():
    
    # degrees of freedom for the robot (DO NOT TOUCH, obv.)
    DOF_JOINTS = 8
    DOF = DOF_JOINTS
    DOF_EE = 3
    EE = DOF_EE
    
    # calibration configuration
    CONFIG_HOME = np.array([0.0, 0.0, 
                            0.0, 0.0, 
                            0.0, 0.0, 
                            0.0, 0.0])
    

# TODO same as in `yumift_common.robot_state`
class SleipnerDeviceState(AbstractDeviceState, Sleipner8DOFDeviceState):
    def __init__(self) -> None:
        super().__init__()

class SleipnerDeviceCommand(AbstractDeviceCommand):

    def __init__(self, pose_velocity_target : np.ndarray = np.zeros(Parameters.DOF_EE)) -> None:
        super().__init__()
        self._pose_velocity_target = pose_velocity_target
        
    def pose_velocity_target(self, target : np.ndarray):
        assert target.shape == (Parameters.DOF_EE,)
        self._pose_velocity_target = target

class SleipnerDevice(AbstractDevice[SleipnerDeviceState, SleipnerDeviceCommand]):
    """ Sleipner is a 8-DOF pseudo-omnidirectional mobile robot.
        For simplicity, we represent it as a classical planar omnidirectional 3-DOF robot.
    """
    
    def __init__(self):
        super().__init__()
        # sleipner command publisher
        self._pub_vel = rospy.Publisher("/base/twist_mux/command_teleop_keyboard", TwistMsg, queue_size=1, tcp_nodelay=False)
        # sleipner state subscriber
        self._cache_state = SleipnerDeviceState()
        self._device_ready = False
        self._device_ready_changed = False
        rospy.Subscriber("/base/joint_states", JointStateMsg, self._callback_received_joints, queue_size=1, tcp_nodelay=False)
        rospy.Subscriber("/base/odometry_controller/odometry", OdometryMsg, self._callback_received_odom, queue_size=1, tcp_nodelay=False)
        # ensure to start the controller with a real robot state 
        # (no wait means default state (all zeros), very bad)
        # TODO use me
        # rospy.wait_for_message("/base/joint_states", JointStateMsg)
        # rospy.wait_for_message("/base/odometry_controller/odometry", OdometryMsg)
    
    def _callback_received_joints(self, data: JointStateMsg):
        self._cache_state.joint_pos = np.array(data.position)
        self._cache_state.joint_vel = np.array(data.velocity)
        self._cache_state.time = rospy.Time.now()
    
    def _callback_received_odom(self, data: OdometryMsg):
        self._cache_state.effector_pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        self._cache_state.effector_rot = quat.from_float_array([data.pose.pose.orientation.w, data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z])
        self._cache_state.effector_vel = np.zeros(6)
        self._cache_state.effector_vel[:3] = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z])
        self._cache_state.effector_vel[3:] = np.array([data.twist.twist.angular.x, data.twist.twist.angular.y, data.twist.twist.angular.z])
        self._cache_state.time = rospy.Time.now()
    
    @override
    def is_ready(self) -> bool:
        return True
    
    @override
    def read(self) -> SleipnerDeviceState:
        return self._cache_state
       
    @override 
    def send(self, command: SleipnerDeviceCommand):
        msg = TwistMsg()
        msg.linear.x=command._pose_velocity_target[0]
        msg.linear.y=command._pose_velocity_target[1]
        msg.linear.z=0
        msg.angular.x=0
        msg.angular.y=0
        msg.angular.z=command._pose_velocity_target[2]
        self._pub_vel.publish(msg)


class SleipnerDeviceAction(AbstractControllerAction, dict):
    
    def __init__(self) -> None:
        super().__init__()
    
    def velocity_joints(self, velocity: np.ndarray):
        assert velocity.shape == (Parameters.DOF,)
        self["velocity_joints"] = velocity
    
    def velocity_pose(self, pseudotwist: np.ndarray):
        assert pseudotwist.shape == (Parameters.DOF_EE,)
        self["velocity_pose"] = pseudotwist

class SleipnerController(
    AbstractController[SleipnerDeviceState, SleipnerDeviceAction, SleipnerDeviceCommand], 
    metaclass=ABCMeta
):
    """ Velocity controller for Sleipner robot.
    """

    def __init__(self, robot_handle = SleipnerDevice()):
        self._device : SleipnerDevice
        super().__init__(robot_handle)
        self.control_law = CartesianVelocityControlLaw(1, 1, np.array([1, 1]))
        self.trajectory = CubicPoseTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        
    def start(self):
        self.reset(self.device_read())  # init trajectory
        super().start(250) # Hz

    def stop(self):
        print("Controller shutting down")
        super().stop()

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
            self._device.send(SleipnerDeviceCommand())
            print(f"Sent stop command ({i+1}/{stop_commands})")
    
    def reset(self, state: SleipnerDeviceState):
        self.control_law.clear()
        
        diff = np.array([-0.5, 0, 0])
        param_init = PoseParam(state.effector_pos, state.effector_rot, state.effector_vel)
        param_final = PoseParam(state.effector_pos + diff, state.effector_rot, state.effector_vel)
        self.trajectory.update(param_init, param_final, 5)
        self.trajectory_initial_time = rospy.Time.now()
    
    def fallback(self, state: SleipnerDeviceState) -> SleipnerDeviceAction:
        action = SleipnerDeviceAction()
        action.velocity_pose(np.zeros(Parameters.DOF_EE))
        return action
        
    def policy(self, state: SleipnerDeviceState) -> SleipnerDeviceAction:
        # timing
        real_now = rospy.Time.now()
        state_now: rospy.Time = state.time
        dt = (real_now - state_now).to_sec()
        time = (real_now - self.trajectory_initial_time).to_sec()
        
        pose_now = Frame(state.effector_pos, state.effector_rot, state.effector_vel)
        
        param_final = self.trajectory.compute(time)
        pose_final = Frame(param_final.pos, param_final.rot, param_final.vel)
        
        vel = self.control_law.update_and_compute(pose_now, pose_final, dt)
        vel = np.array([vel[0], vel[1], vel[5]])
      
        # calculate new target velocities for this time step
        action = SleipnerDeviceAction()
        action.velocity_pose(vel)
    
        return action
    
    def solve_action(self, state: SleipnerDeviceState, action: SleipnerDeviceAction) -> SleipnerDeviceCommand:
        
        if "velocity_joints" in action:
            # dq_target = action["velocity_joints"]
            raise NotImplementedError("We don't know how to do this yet")  # TODO fix me
        else:
            pose_target = action["velocity_pose"]
        
        return SleipnerDeviceCommand(pose_target)


def main():
    
    # starting ROS node
    rospy.init_node("sleipner_controller", anonymous=False)
    
    controller = SleipnerController()
    rospy.on_shutdown(controller.stop)
    
    controller.ready()
    controller.start()  # locking


if __name__ == "__main__":
    main()
