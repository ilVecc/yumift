import numpy as np
import quaternion as quat

import rospy

from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped

from .robot_state import YumiCoordinatedRobotState
from .trajectory import YumiParam
from dynamics.utils import Frame


def _WrenchStampedMsg_to_ndarray(wrench_stamped: WrenchStamped):
    fx = wrench_stamped.wrench.force.x
    fy = wrench_stamped.wrench.force.y
    fz = wrench_stamped.wrench.force.z
    mx = wrench_stamped.wrench.torque.x
    my = wrench_stamped.wrench.torque.y
    mz = wrench_stamped.wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])


def _PoseMsg_to_frame(pose_msg: Pose):
    return Frame(
        position=np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]),
        quaternion=np.quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z))


def YumiParam_to_YumiCoordinatedRobotState(yumi_param: YumiParam):
    """ Transforms a desired Yumi parameter into a Yumi state.
    """
    yumi_state = YumiCoordinatedRobotState(
        grip_r=yumi_param.grip_right, 
        grip_l=yumi_param.grip_left)
    # Since the `pose_*` arguments of `YumiCoordinatedRobotState` are the ones 
    # for the robot's flanges, a small workaround is needed. This is totally 
    # fine though because this transformation will only be used by the control 
    # law, which doesn't require the state to be sound nor complete
    yumi_state.pose_gripper_r = Frame(
        yumi_param.position[:3],
        yumi_param.rotation[0],
        yumi_param.velocity[:6])
    yumi_state.pose_gripper_l = Frame(
        yumi_param.position[3:],
        yumi_param.rotation[1],
        yumi_param.velocity[6:])
    return yumi_state


def _Frame_to_PoseStamped(pose: Frame, parent: str = "yumi_base_link"):
    msg = PoseStamped()
    msg.header.frame_id = parent
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = pose.pos[0]
    msg.pose.position.y = pose.pos[1]
    msg.pose.position.z = pose.pos[2]
    msg.pose.orientation.w = pose.rot.w
    msg.pose.orientation.x = pose.rot.x
    msg.pose.orientation.y = pose.rot.y
    msg.pose.orientation.z = pose.rot.z
    return msg
