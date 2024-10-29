#!/usr/bin/env python3
import rospy, tf

import numpy as np

from collections import deque

from geometry_msgs.msg import WrenchStamped, PoseStamped
from nav_msgs.msg import Path

board_weight = 0.245  # kg
board_width = 0.2845  # x, m
board_depth = 0.199  # y, m
board_height = 0.199  # y, m
board_offset = 0.031 + 0.034
board_com = np.array([board_width, board_depth, board_height]) / 2
_board_bias = board_weight * 9.81  # world-z N


def _ndarray_to_WrenchStampedMsg(wrench: np.ndarray, frame: str):
    msg = WrenchStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame
    msg.wrench.force.x = wrench[0]
    msg.wrench.force.y = wrench[1]
    msg.wrench.force.z = wrench[2]
    msg.wrench.torque.x = wrench[3]
    msg.wrench.torque.y = wrench[4]
    msg.wrench.torque.z = wrench[5]
    return msg

def _WrenchStampedMsg_to_ndarray(msg: WrenchStamped):
    fx = msg.wrench.force.x
    fy = msg.wrench.force.y
    fz = msg.wrench.force.z
    mx = msg.wrench.torque.x
    my = msg.wrench.torque.y
    mz = msg.wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])

_wrenches = np.zeros(12)  # [right, left]

_pub_r = rospy.Publisher("/wrench_r", WrenchStamped, queue_size=1, tcp_nodelay=False)
_pub_l = rospy.Publisher("/wrench_l", WrenchStamped, queue_size=1, tcp_nodelay=False)

def _callback_set_wrench(msg: WrenchStamped, arm: str):
    if arm == "r":
        _wrenches[0:6] = _WrenchStampedMsg_to_ndarray(msg)
        _wrenches[1] -= _board_bias / 2
        _pub_r.publish(_ndarray_to_WrenchStampedMsg(_wrenches[0:6], msg.header.frame_id))
    elif arm == "l":
        _wrenches[6:12] = _WrenchStampedMsg_to_ndarray(msg)
        _wrenches[7] += _board_bias / 2
        _pub_l.publish(_ndarray_to_WrenchStampedMsg(_wrenches[6:12], msg.header.frame_id))


_pub_path = rospy.Publisher("path", Path, tcp_nodelay=True, queue_size=1)

def _coord_to_PoseStamped(x, y, parent="gripper_r_tip"):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = parent
    msg.pose.position.x = 0
    msg.pose.position.y = x
    msg.pose.position.z = y
    msg.pose.orientation.w = 1
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0
    return msg

def main():
    # starting ROS node and subscribers
    rospy.init_node("contact_point_estimation", anonymous=False)
    
    rospy.Subscriber("/ftsensor_r/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="r", queue_size=1, tcp_nodelay=False)
    rospy.Subscriber("/ftsensor_l/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="l", queue_size=1, tcp_nodelay=False)
    
    
    history = deque(maxlen=100)
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        # x1 + x2
        # y1 - y2
        # z1 - z2
        f1, f2 = _wrenches[0:3], np.array([1, -1, -1]) * _wrenches[6:9]
        f1, f2 = f1[0], f2[0]
        t1, t2 = _wrenches[3:6], np.array([1, -1, -1]) * _wrenches[9:12]
        t1, t2 = t1[2], t2[2]
        
        fa = f1 + f2
        r1 = 1 - np.linalg.norm(f1) / np.linalg.norm(fa)
        r2 = 1 - np.linalg.norm(f2) / np.linalg.norm(fa)
        ra = t1 / np.linalg.norm(fa)
        
        # store the value
        history.append(_coord_to_PoseStamped(ra * 2.5, (r1 - 0.2) * board_width))
        
        # publish everything
        path = Path()
        path.header.frame_id = "gripper_r_tip"
        path.header.stamp = rospy.Time.now()
        path.poses = list(history)
        _pub_path.publish(path)
        
        rate.sleep()


if __name__ == "__main__":
    main()


