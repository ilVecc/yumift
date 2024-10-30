#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("routine_face_inside", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.45, -0.15, 0.5],
            orientationRight = [ 0.7071068, -0.7071068, 0, 0 ],
            positionLeft  = [0.45,  0.15, 0.5],
            orientationLeft  = [ 0.7071068,  0.7071068, 0, 0 ],
            pointTime = 4.0),
    ]
    pub.publish(msg)
    print("message sent")
    rospy.sleep(4)


if __name__ == "__main__":
    main()


