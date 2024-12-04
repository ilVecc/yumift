#!/usr/bin/env python3
import rospy, tf

import numpy as np

from collections import deque

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import WrenchStamped, PoseStamped, Pose, Point, Vector3, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker


DEBUG = False


board_weight = 0.245  # kg
board_width = 0.2845  # x, m
board_depth = 0.199  # y, m
board_height = 0.199  # y, m
board_offset = 0.031 + 0.034
board_com = np.array([board_width, board_depth, board_height]) / 2
_board_bias = board_weight * 9.81  # world-z N



if DEBUG:
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

    _pub_r = rospy.Publisher("/wrench_r", WrenchStamped, queue_size=1, tcp_nodelay=False)
    _pub_l = rospy.Publisher("/wrench_l", WrenchStamped, queue_size=1, tcp_nodelay=False)


def _WrenchStampedMsg_to_ndarray(msg: WrenchStamped):
    fx = msg.wrench.force.x
    fy = msg.wrench.force.y
    fz = msg.wrench.force.z
    mx = msg.wrench.torque.x
    my = msg.wrench.torque.y
    mz = msg.wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])

_wrenches = np.zeros(12)  # [right, left]
def _callback_set_wrench(msg: WrenchStamped, arm: str):
    if arm == "r":
        _wrenches[0:6] = _WrenchStampedMsg_to_ndarray(msg)
        _wrenches[1] -= _board_bias / 2
        if DEBUG:
            _pub_r.publish(_ndarray_to_WrenchStampedMsg(_wrenches[0:6], msg.header.frame_id))
    elif arm == "l":
        _wrenches[6:12] = _WrenchStampedMsg_to_ndarray(msg)
        _wrenches[7] += _board_bias / 2
        if DEBUG:
            _pub_l.publish(_ndarray_to_WrenchStampedMsg(_wrenches[6:12], msg.header.frame_id))


_pub_trace = rospy.Publisher("/trace", Marker, queue_size=1, tcp_nodelay=False)

class Timer:
    init_time = None

    def tic(self):
        self.init_time = rospy.Time.now()

    def toc(self):
        return (rospy.Time.now() - self.init_time).to_sec()


def main():
    # starting ROS node and subscribers
    rospy.init_node("contact_point_estimation", anonymous=False)
    
    rospy.Subscriber("/ftsensor_r/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="r", queue_size=1, tcp_nodelay=False)
    rospy.Subscriber("/ftsensor_l/tool_tip", WrenchStamped, _callback_set_wrench, callback_args="l", queue_size=1, tcp_nodelay=False)


    n = 5
    lp_filter_x = deque(maxlen=n)
    lp_filter_y = deque(maxlen=n)
    lp_filter_x.append(0)
    lp_filter_y.append(0)
    history_points = deque(maxlen=2500)
    new_stroke = True
    timer = Timer()
    timer.tic()


    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        # x = x1 + x2    y = y1 - y2    z = z1 - z2
        f1, f2 = _wrenches[0:3], np.array([1, -1, -1]) * _wrenches[6:9]

        if np.linalg.norm(f1 + f2) > 3.1:
            f1, f2 = f1[0], f2[0]
            t1, t2 = _wrenches[3:6], np.array([1, -1, -1]) * _wrenches[9:12]
            t1, t2 = t1[2], t2[2]

            fa = f1 + f2
            r1 = 1 - np.linalg.norm(f1) / np.linalg.norm(fa)
            r2 = 1 - np.linalg.norm(f2) / np.linalg.norm(fa)
            ra = t1 / np.linalg.norm(fa)

            # store the value
            if not new_stroke:
                print("cleaning")
                history_points.clear()
                lp_filter_x.clear()
                lp_filter_y.clear()
                lp_filter_x.append(0)
                lp_filter_y.append(0)
                new_stroke = True

            timer.tic()
            x = ra*2.5
            y = (r1-0.2)*board_width
            l = np.linalg.norm([x-lp_filter_x[-1], y-lp_filter_y[-1]])
            if l > 0.01:
                lp_filter_x.clear()
                lp_filter_y.clear()
            lp_filter_x.append(x)
            lp_filter_y.append(y)
            if len(lp_filter_x) == n:
                x_fil = sum(lp_filter_x) / len(lp_filter_x)
                y_fil = sum(lp_filter_y) / len(lp_filter_y)
                history_points.append(Point(x=0, y=x_fil, z=y_fil))
        else:
            if timer.toc() > 2.0 and new_stroke:
                new_stroke = False

        # publish everything
        markers = Marker()
        markers.header.frame_id = "gripper_r_tip"
        markers.header.stamp = rospy.Time.now()
        markers.id = 100
        markers.type = Marker.SPHERE_LIST
        markers.action = Marker.ADD
        markers.lifetime = rospy.Duration.from_sec(0)
        markers.pose.position = Vector3(x=0, y=0, z=0)
        markers.pose.orientation = Quaternion(w=1, x=0, y=0, z=0)
        markers.color = ColorRGBA(1, 1, 0, 1)
        markers.scale = Vector3(x=0.005, y=0.005, z=0.005)
        markers.points = list(history_points)
        _pub_trace.publish(markers)

        
        rate.sleep()


if __name__ == "__main__":
    main()
