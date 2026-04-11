from typing_extensions import override

import rospy
import numpy as np, quaternion as quat

from geometry_msgs.msg import Twist as TwistMsg
from nav_msgs.msg import Odometry as OdometryMsg

from .constants import SleipnerRobotConstants

from dynamicals.common.devices import AbstractDevice, AbstractDeviceCommand, AbstractDeviceState
from dynamicals.common.robotics import RobotState


###############################################################################
###                              WHEELS STATE                               ###
###############################################################################

# TODO make use of or delete me
class Sleipner8DOFDeviceState(AbstractDeviceState, RobotState):
    
    def __init__(self,
        joint_pos: np.ndarray = np.zeros(SleipnerRobotConstants.DOF),
        joint_vel: np.ndarray = np.zeros(SleipnerRobotConstants.DOF),
        joint_torque: np.ndarray = np.zeros(SleipnerRobotConstants.DOF),
        pose_xy: np.ndarray = np.zeros(2),
        pose_th: float = 0,
        pose_pseudovel: np.ndarray = np.zeros(SleipnerRobotConstants.EE),
        pose_pseudowrc: np.ndarray = np.zeros(SleipnerRobotConstants.EE),
        jacobian: np.ndarray = np.zeros((6, SleipnerRobotConstants.DOF))
    ):
        super()
        AbstractDeviceState.__init__(self)
        RobotState.__init__(self,
            SleipnerRobotConstants.EE,
            joint_pos, joint_vel, None, joint_torque,
            np.array([pose_xy[0], pose_xy[1], 0]), quat.from_rotation_vector(np.array([0, 0, pose_th])), pose_pseudovel, None, pose_pseudowrc,
            jacobian, None)
        self.obs_time = rospy.Time.now()
        self.obs_vect = np.zeros((3,))

    ####### FRONT RIGHT #######
    @property
    def wheel_fr_pos(self):
        return self.joint_pos[0:2]

    @property
    def joint_fr_vel(self):
        return self.joint_vel[0:2]

    @property
    def joint_fr_acc(self):
        return self.joint_acc[0:2]

    @property
    def joint_fr_torque(self):
        return self.joint_tau[0:2]

    ####### FRONT LEFT #######
    @property
    def wheel_fl_pos(self):
        return self.joint_pos[2:4]

    @property
    def joint_fl_vel(self):
        return self.joint_vel[2:4]

    @property
    def joint_fl_acc(self):
        return self.joint_acc[2:4]

    @property
    def joint_fl_torque(self):
        return self.joint_tau[2:4]

    ####### BACK RIGHT #######
    @property
    def wheel_br_pos(self):
        return self.joint_pos[4:6]

    @property
    def joint_br_vel(self):
        return self.joint_vel[4:6]

    @property
    def joint_br_acc(self):
        return self.joint_acc[4:6]

    @property
    def joint_br_torque(self):
        return self.joint_tau[4:6]

    ####### BACK LEFT #######
    @property
    def wheel_bl_pos(self):
        return self.joint_pos[6:8]

    @property
    def joint_bl_vel(self):
        return self.joint_vel[6:8]

    @property
    def joint_bl_acc(self):
        return self.joint_acc[6:8]

    @property
    def joint_bl_torque(self):
        return self.joint_tau[6:8]


###############################################################################
###                             CARTESIAN STATE                             ###
###############################################################################

class SleipnerCartesianDeviceState(AbstractDeviceState):
    
    def __init__(self,
        pose_SE2: np.ndarray = np.zeros(3),
        twist_SE2: np.ndarray = np.zeros(3)
    ):
        super().__init__()
        self.pose_SE2 = pose_SE2
        self.twist_SE2 = twist_SE2

class SleipnerCartesianDeviceCommand(AbstractDeviceCommand):

    def __init__(self, pose_velocity_target : np.ndarray = np.zeros(SleipnerRobotConstants.DOF_EE)) -> None:
        super().__init__()
        self._pose_vel_tgt = pose_velocity_target

    def pose_velocity_target(self, target : np.ndarray):
        assert target.shape == (SleipnerRobotConstants.DOF_EE,)
        self._pose_vel_tgt = target

class SleipnerCartesianDevice(AbstractDevice[SleipnerCartesianDeviceState, SleipnerCartesianDeviceCommand]):
    """ Sleipner is a 8-DOF pseudo-omnidirectional mobile robot.
        For simplicity and control hardware restrictions, we represent it as a 
        classical planar omnidirectional 3-DOF robot.
    """

    def __init__(self):
        super().__init__()
        self._cache_state = SleipnerCartesianDeviceState()
        # sleipner command publisher
        self._pub_vel = rospy.Publisher("/base/twist_mux/command_teleop_keyboard", TwistMsg, queue_size=1, tcp_nodelay=False)
        # sleipner state subscriber
        rospy.Subscriber("/base/odometry_controller/odometry", OdometryMsg, self._callback_received_odom, queue_size=1, tcp_nodelay=False)
        # ensure to start the controller with a real robot state 
        # (zero-wait-time means default state (all zeros), which is very bad)
        rospy.wait_for_message("/base/odometry_controller/odometry", OdometryMsg)

    def _callback_received_odom(self, data: OdometryMsg):
        pose, twist = data.pose.pose, data.twist.twist
        theta = 2*np.arctan2(pose.orientation.z, pose.orientation.w)  # quick quaterion-to-angle conversion
        self._cache_state.pose_SE2 = np.array([pose.position.x, pose.position.y, theta])
        self._cache_state.twist_SE2 = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
        self._cache_state.time = rospy.Time.now()
    
    @override
    def reset(self) -> bool:
        return True
    
    @override
    def is_ready(self) -> bool:
        # TODO actually look for state change in Sleipner
        return True

    @override
    def read(self) -> SleipnerCartesianDeviceState:
        return self._cache_state

    @override
    def send(self, command: SleipnerCartesianDeviceCommand):
        msg = TwistMsg()
        msg.linear.x=command._pose_vel_tgt[0]
        msg.linear.y=command._pose_vel_tgt[1]
        msg.angular.z=command._pose_vel_tgt[2]
        self._pub_vel.publish(msg)
