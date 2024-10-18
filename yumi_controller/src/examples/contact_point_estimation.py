#!/usr/bin/env python
import rospy

import numpy as np

from geometry_msgs.msg import WrenchStamped


_wrenches = np.zeros(12)  # [right, left]

def _WrenchStampedMsg_to_ndarray(msg: WrenchStamped):
    fx = msg.wrench.force.x
    fy = msg.wrench.force.y
    fz = msg.wrench.force.z
    mx = msg.wrench.torque.x
    my = msg.wrench.torque.y
    mz = msg.wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])

def _callback_set_wrench(msg: WrenchStamped, arm: str):
    if arm == "right":
        _wrenches[0:6] = _WrenchStampedMsg_to_ndarray(msg)
    elif arm == "left":
        _wrenches[6:12] = _WrenchStampedMsg_to_ndarray(msg)


def main():
    # starting ROS node and subscribers
    rospy.init_node("contact_point_estimation", anonymous=True)
    
    rospy.Subscriber("/ftsensor_r/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="r", queue_size=10)
    rospy.Subscriber("/ftsensor_l/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="l", queue_size=10)
    
    
    
    rospy.sleep(0.1)


if __name__ == "__main__":
    main()


