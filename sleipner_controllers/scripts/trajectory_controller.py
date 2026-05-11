#!/usr/bin/env python3
from typing_extensions import override

import rospy
import numpy as np, quaternion as quat

from dynamicals.utils import Frame
from dynamicals.impl import AbstractROSController, CartesianVelocityControlLaw
from pathfinder.polynomial import CubicPoseTrajectory, PoseParam

from sleipner_controllers.common import SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand, SleipnerCartesianDevice, SE2TwistAction

from geometry_msgs.msg import Quaternion


class SleipnerCartesianTrajectoryController(
    AbstractROSController[SleipnerCartesianDeviceState, SE2TwistAction, SleipnerCartesianDeviceCommand]
):
    """ Cartesian velocity controller for Sleipner robot.
    
        Try the controller by sending a command like
        
        `rostopic pub /base/target geometry_msgs/Quaternion "{x: 1, y: 0, z: 0, w: 5}" --once`
        
        where `(x,y,z)` are the SE2 position and orientation coordinates, while `w` 
        is interpreted as the desired time.
    """

    def __init__(self, robot_handle : SleipnerCartesianDevice):
        self._device : SleipnerCartesianDevice
        super().__init__(robot_handle)
        self.control_law = CartesianVelocityControlLaw(k_p=2, k_o=1.5, 
                                                       min_actionable_error=[0.05, 0.02],
                                                       max_allowed_deviation=[1.5, np.pi/2])
        self.trajectory = CubicPoseTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        rospy.Subscriber("/base/target", Quaternion, self._callback_received_target, queue_size=1)
    
    def update_traj(self, initial_state : SleipnerCartesianDeviceState, target_pose : Frame, target_time : float):
        # initialize new trajectory
        frame_init = self._device.state_wrt_home(initial_state)
        frame_final = (frame_init @ target_pose).motionless()
        
        self.trajectory.update(PoseParam.from_Frame(frame_init), PoseParam.from_Frame(frame_final), target_time)
        self.trajectory_initial_time = rospy.Time.now()
        
        with np.printoptions(precision=2, suppress=True):
            rospy.loginfo(f"Init pose:  {frame_init.pos} {quat.as_float_array(frame_init.rot)}")
            rospy.loginfo(f"Final pose: {frame_final.pos} {quat.as_float_array(frame_final.rot)}")
    
    def _callback_received_target(self, msg : Quaternion):
        rospy.loginfo(f"New target received: [{msg.x}, {msg.y}, {msg.z}] in {msg.w} sec")
        target_tra = np.array([msg.x, msg.y, 0])
        target_rot = quat.from_rotation_vector(np.array([0, 0, msg.z]))
        target_time = msg.w
        self.update_traj(self.device_read(), Frame(target_tra, target_rot), target_time)
        
    @override
    def reset(self, state: SleipnerCartesianDeviceState):
        self.control_law.clear()
        self.update_traj(state, Frame(), 0.001)
    
    @override
    def policy(self, state: SleipnerCartesianDeviceState) -> SE2TwistAction:
        state_dt = self.dt(state.time)
        traj_dt = self.dt(self.trajectory_initial_time)
        
        frame_now = self._device.state_wrt_home(state)
        frame_next = PoseParam.to_Frame(self.trajectory.compute(traj_dt))
        
        twist = self.control_law.update_and_compute(frame_now, frame_next, state_dt)
        
        return SE2TwistAction.from_twist_SE3(twist)
    
    @override
    def solve_action(self, state: SleipnerCartesianDeviceState, action: SE2TwistAction) -> SleipnerCartesianDeviceCommand:
        return SleipnerCartesianDeviceCommand(action.twist_SE2)


if __name__ == "__main__":
    rospy.init_node("sleipner_cartesian_trajectory_controller", anonymous=False)
    
    sleipner = SleipnerCartesianDevice()
    controller = SleipnerCartesianTrajectoryController(sleipner)
    
    controller.ready()
    controller.start()  # locking
