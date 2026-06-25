#!/usr/bin/env python3
import argparse
import math

import rospy
import tf2_ros
from geometry_msgs.msg import Twist, TransformStamped, Pose
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


class EmulationBase(object):
    def __init__(self, odom_frame):
        # this node emulates the controllers of a base including twist_controller and odometry_controller
        #
        # interfaces
        # - subscribers:
        #   - /base/twist_controller/command [geometry_msgs/Twist]
        # - publishers:
        #   - /base/odometry_controller/odometry [nav_msgs/Odometry]
        #   - /base/joint_states [sensor_msgs/JointState]
        #   - tf (odom_frame --> base_footprint)

        # TODO
        # - speed factor

        self._update_period = 0.1  # 10Hz
        self._odom_frame = odom_frame
        self._foot_frame = "base_footprint"

        self._last_twist = Twist()
        self._last_twist_timestamp = rospy.Time(0)
        rospy.Subscriber("/base/twist_controller/command", Twist, self._callback_twist, queue_size=1)
        
        # pub base_footprint --> odom_frame
        self._cache_trans = TransformStamped()
        self._cache_trans.header.frame_id = self._odom_frame
        self._cache_trans.child_frame_id = self._foot_frame
        self.pub_trans = tf2_ros.TransformBroadcaster()

        self._cache_odom = Odometry()
        self._cache_odom.header.frame_id = self._odom_frame
        self._cache_odom.child_frame_id = self._foot_frame
        self._cache_odom.pose.pose.orientation.w = 1 # initialize orientation with a valid quaternion
        self._cache_angle = 0
        self.pub_odom = rospy.Publisher("/base/odometry_controller/odometry", Odometry, queue_size=1)
        
        self._cache_joint_state = JointState()
        self._cache_joint_state.name = [
            "fl_caster_rotation_joint", "fl_caster_r_wheel_joint",
            "fr_caster_rotation_joint", "fr_caster_r_wheel_joint",
            "bl_caster_rotation_joint", "bl_caster_r_wheel_joint",
            "br_caster_rotation_joint", "br_caster_r_wheel_joint"]
        self._cache_joint_state.position = [0] * 8
        self._cache_joint_state.velocity = [0] * 8
        self.pub_jointstate = rospy.Publisher("/base/joint_states", JointState, queue_size=1)

        rospy.Service("/base/odometry_controller/reset_odometry", Trigger, self._srv_reset_odometry)

        self._last_update_timestamp = rospy.Time.now()
        rospy.Timer(rospy.Duration(self._update_period), self._callback_update_everything)
        
        rospy.loginfo("Emulation for Sleipner running")

    def _srv_reset_odometry(self, req : TriggerRequest):
        self._cache_odom.pose.pose = Pose()
        self._cache_odom.pose.pose.orientation.w = 1
        return TriggerResponse(True, "odometry reset")

    def _callback_twist(self, msg : Twist):
        self._last_twist = msg
        self._last_twist_timestamp = rospy.Time.now()

    def _callback_update_everything(self, event):
        
        # move robot (calculate new pose)
        dt_update = (rospy.Time.now() - self._last_update_timestamp).to_sec()
        dt_twist = (rospy.Time.now() - self._last_twist_timestamp).to_sec()
        
        # we assume we're not moving any more if there is no new twist after 0.1 sec
        if dt_twist < self._update_period:
            self._cache_angle += self._last_twist.angular.z * dt_update
            self._cache_odom.pose.pose.orientation.w = math.cos(self._cache_angle * 0.5)
            self._cache_odom.pose.pose.orientation.z = math.sin(self._cache_angle * 0.5)
            cos = math.cos(self._cache_angle)
            sin = math.sin(self._cache_angle)
            dx = (self._last_twist.linear.x * cos - self._last_twist.linear.y * sin) * dt_update
            dy = (self._last_twist.linear.x * sin + self._last_twist.linear.y * cos) * dt_update
            self._cache_odom.pose.pose.position.x += dx
            self._cache_odom.pose.pose.position.y += dy
            # we're moving, so we set a non-zero twist
            self._cache_odom.twist.twist.linear.x = self._last_twist.linear.x
            self._cache_odom.twist.twist.linear.y = self._last_twist.linear.y
            self._cache_odom.twist.twist.angular.z = self._last_twist.angular.z
        else:
            # reset twist as we're not moving anymore
            self._cache_odom.twist.twist = Twist()

        # publish joint state
        self._cache_joint_state.header.stamp = rospy.Time.now()
        # TODO actually update state 
        self.pub_jointstate.publish(self._cache_joint_state)
        
        # publish odometry
        self._cache_odom.header.stamp = rospy.Time.now()
        self.pub_odom.publish(self._cache_odom)

        # publish odom -> base on /tf
        self._cache_trans.header.stamp = rospy.Time.now()
        self._cache_trans.transform.translation = self._cache_odom.pose.pose.position
        self._cache_trans.transform.rotation = self._cache_odom.pose.pose.orientation
        self.pub_trans.sendTransform(self._cache_trans)
        
        self._last_update_timestamp = rospy.Time.now()
        

if __name__ == '__main__':
    rospy.init_node('sleipner_emulator', anonymous=False)
    
    parser = argparse.ArgumentParser(conflict_handler='resolve',
                                     description="Tool for emulating base by publishing odometry and propagating base_footprint.")
    parser.add_argument('-o', '--odom_frame', help='odom frame name (default: \'odom_combined\')', default='odom_combined')
    args, unknown = parser.parse_known_args()
    
    EmulationBase(args.odom_frame)
    
    rospy.spin()
