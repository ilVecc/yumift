#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1)
    rospy.sleep(0.1)
    
    # def shutdown_callback():
    #     msg = YumiTrajectory()
    #     msg.header.stamp = rospy.Time.now()
    #     msg.mode = "individual"
    #     msg.trajectory = [YumiPosture(pointTime = 0.5)]
    #     pub.publish(msg)
    #     rospy.sleep(0.1)
    #     rospy.signal_shutdown("interrupting trajectories")
    #     print("Interrupting trajectories")
        
    # rospy.on_shutdown(shutdown_callback)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        # YumiPosture(
        #     positionAbsolute = [0.55, 0.0, 0.2],
        #     positionRelative = [0, 0.10, 0],
        #     orientationAbsolute = [0, 1, 0, 0],
        #     orientationRelative = [1, 0, 0, 0],
        #     pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.55, 0.0, 0.2],
            positionRelative = [0.05, 0.05, 0.05],
            orientationAbsolute = [0,1,0,0], #trans.quaternion_about_axis(np.radians(135), [1, 0, 0])
            orientationRelative = np.roll(trans.quaternion_about_axis(np.radians(90), [1, 0, 0]), 1),
            pointTime = 8.0),
        # YumiPosture(
        #     positionAbsolute = [0.55, 0.0, 0.2],
        #     positionRelative = [0, 0.50, 0],
        #     orientationAbsolute = [0, 1, 0, 0],
        #     orientationRelative = np.roll(trans.quaternion_about_axis(np.radians(45), [1, 0, 0]), 1),
        #     pointTime = 8.0),
        # YumiPosture(
        #     positionAbsolute = [0.55, 0.0, 0.2]
        #     positionRelative = [0, 0.30, 0]
        #     orientationAbsolute = [0, 1, 0, 0]
        #     orientationRelative = [1, 0, 0, 0]
        #     pointTime = 8.0),
    ]
    pub.publish(msg)
    
    print("Sent message")
    rospy.sleep(8)


if __name__ == "__main__":
    main()


