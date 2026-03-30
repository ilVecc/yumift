from typing import Union

import numpy as np, quaternion as quat

from dynamicals.utils import Frame

from geometry_msgs.msg import Pose as PoseMsg, Wrench as WrenchMsg, Twist as TwistMsg
from yumift_msgs.msg import Jacobian as JacobianMsg


def TwistMsg_to_ndarray(twist: TwistMsg, out: Union[np.ndarray, None] = None):
    a = [twist.linear.x, twist.linear.y, twist.linear.z, 
         twist.angular.x, twist.angular.y, twist.angular.z]
    if out is None:
        return np.array(a)
    out[0:6] = a
    return out

def WrenchMsg_to_ndarray(wrench: WrenchMsg, out: Union[np.ndarray, None] = None):
    a = [wrench.force.x, wrench.force.y, wrench.force.z,
         wrench.torque.x, wrench.torque.y, wrench.torque.z]
    if out is None:
        return np.array(a)
    out[0:6] = a
    return out

def PoseMsg_to_Frame(pose_msg: PoseMsg, out: Union[Frame, None] = None):
    pos = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    rot = np.quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z)
    if out is None:
        return Frame(pos, rot)
    else:
        out.pos = pos
        out.rot = rot
        return out

def JacobianMsg_to_ndarray(jacobian: JacobianMsg, out: Union[np.ndarray, None] = None):
    if out is None:
        out = np.zeros((6, jacobian.dof), dtype=np.float64)
    out[0, :] = jacobian.vx
    out[1, :] = jacobian.vy
    out[2, :] = jacobian.vz
    out[3, :] = jacobian.wx
    out[4, :] = jacobian.wy
    out[5, :] = jacobian.wz
    return out
