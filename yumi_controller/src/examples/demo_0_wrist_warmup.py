#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------
    # 1. Z down
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.45, -0.3, 0.3],
            positionLeft = [0.45, 0.3, 0.3],
            orientationRight = [0, 1, 0, 0],
            orientationLeft = [0, 1, 0, 0],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent Z down ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")

    # --------------------------------------------------
    # 2. Y down
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.60, -0.3, 0.4],
            positionLeft = [0.60, 0.3, 0.4],
            orientationRight = [0.5, -0.5, 0.5, -0.5],
            orientationLeft = [0.5, -0.5, 0.5, -0.5],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent Y down ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")

    # --------------------------------------------------
    # 3. X up
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.60, -0.3, 0.4],
            positionLeft = [0.60, 0.3, 0.4],
            orientationRight = [0.0, 0.707, 0.0, 0.707],
            orientationLeft = [0.0, 0.707, 0.0, 0.707],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent X up ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")

    # --------------------------------------------------
    # 4. Y up
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.60, -0.3, 0.4],
            positionLeft = [0.60, 0.3, 0.4],
            orientationRight = [0.5, 0.5, 0.5, 0.5],
            orientationLeft = [0.5, 0.5, 0.5, 0.5],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent Y up ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")

    # --------------------------------------------------
    # 5. X down
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.60, -0.3, 0.4],
            positionLeft = [0.60, 0.3, 0.4],
            orientationRight = [0.707, 0.0, 0.707, 0.0],
            orientationLeft = [0.707, 0.0, 0.707, 0.0],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent X down ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")

    # --------------------------------------------------
    # 6. Z up
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"  # control mode
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.60, -0.3, 0.4],
            positionLeft = [0.60, 0.3, 0.4],
            orientationRight = [0.5, 0.5, 0.5, 0.5],
            orientationLeft = [0.5, 0.5, 0.5, 0.5],
            pointTime = 2.0),
        YumiTrajectoryPoint(
            positionRight = [0.45, -0.3, 0.5],
            positionLeft = [0.45, 0.3, 0.5],
            orientationRight = [0, 0, 0, -1],
            orientationLeft = [0, 0, 0, -1],
            pointTime = 2.0)
    ]
    pub.publish(msg)

    print("sent Z up ... ", end="", flush=True)
    rospy.sleep(10)
    print("done")
    

if __name__ == "__main__":
    main()
