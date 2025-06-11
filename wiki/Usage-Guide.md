# Usage

### Using hardware:
When using real hardware and inverse kinematics it is important that the robot is in a good initial configuration, especially for trajectory control, it is therefore recommended to reset the pose before use. 

warning: this activates EGM and joint controllers and also closes EGM and rapid when set_yumi_settings_and_start.py is closed (In our lab, Ip: 192.168.125.1) 

```
roslaunch abb_robot_bringup_examples ex3_rws_and_egm_yumi_robot.launch robot_ip:=<ROBOT_IP> 
rosrun controller set_yumi_settings_and_start.py
```

### Using the trajectory controller:
Then use roslaunch to start the trajectory controller. 
``` 
roslaunch controller yumiTrajectoryControl.launch 
``` 
Then to send example trajectories to the trajectory controller run.
``` 
rosrun controller testTrajectories.py 
``` 


# Create trajectories (Assumes you are using the yumiTrajectoryControl.launch)
The robot is controlled by sending a list of trajectory parameters that the controller should follow. The list has to contain at least 1 trajectory point (not for resetPose), and if there are more the controller follows the trajectory points in the order of the list. The trajectory message is defined at /yumi/controller/msg/Trajectory_msg.msg, this message can hold a list of trajectory points. These trajectory points are defined at /yumi/controller/msg/Trajectory_point.msg.

Some imported points
* The control mode (i.e. individual, coordinated or resetPose) is defined in Trajectory_msg.msg. This means that a single trajectory can not have multiple control modes. 
* units are [m], [rad] and [s]
* Any current trajectory in the controller will be immediately overwritten when a new one is received. 
* The time argument in the Trajectory_point.msg describes how long time it should take to get from the previous trajectory point (or current pose if first in trajectory list) to the pose describes by the trajectory point. This is the main form for controlling velocity.
* Only the fields relevant for the control mode has to be set in the Trajectory_point.msg. For individual control, both right and left has to be set for each trajectory point in the list. For coordinated manipulation, all fields for absolute and relative control has to be set. 

* Take a careful look at how the frames are set up, they can be visualized in rviz with the tf frames. The trajectory parameters describe the pose in the yumi_base_link frame for the grippers. The grippers z-axis points in the direction of the grippers (i.e. normally down) while the base fame has the z-axis up. This means that the orientation of the grippers to be pointing towards the workspace they have to be flipped 180 degrees around the x-axis (or y-axis). Also, be careful about making 180-degree orientation changes between two trajectory points as the controller will take the shortest path with is ambiguous for 180-degree orientation changes. 

* For coordinated manipulation, the absolute frame is the average of the grippers and the relative frame is the difference. The relative frame should therefore not be flipped 180 degrees as that would entail that the grippers should have a relative orientation difference of 180 degrees.  

# Parameters
There are some parameters that can easily be tuned in the controller. In the file /controller/src/parameters.py the important parameters can be found. Some important points for the parameters.
* gripperRightLocal (also left), this sets the local transformation from the wrist of the yumi arm to the point that should be controlled with inverse kinematics. I will also be published on the tf-tree, these parameters can also be set and changed during operation through the self.tfFrames class instance of utils.TfBroadcastFrames(). 
* feasibilityObjectives, this sets with extra feasibility objectives that should be added to the stack of tasks for the HQP solver, these can also be overwritten by setting in the action command with the same keys. To note the "gripperCollision" can cause problems with finding a valid solution in some cases. 
* the velocity bound is for the joints in rad/s, assumed identical for both arms. There are also velocity bounds in the ABB driver.
* neutral pose is only for the joint potential task and not for the reset pose. 
* update rate has to be changed in multiple places, both in parameters and in the kdl_kinematics.cpp 

# Offset joint position (Calibration)
A very basic way to set offset values for each joint can be found in kdl_kinematics.cpp. A variable "joint_offset" is defined and can be used for simple tuning of yumi to increase accuracy.

# YuMi EGM settings
The yumi EGM settings can be found in /controller/src/egm_setup/set_yumi_settings_and_start.py, read the ABB manual for in-depth information. Important parts 

* For velocity control the pos_corr_gain = 0
* max_speed_deviation determines the max speed for each joint on the robot side in [deg/s], there is one more place in the ABB driver where joint speed is limited.
* object mass and other parameters can be set. 

