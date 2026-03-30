uint8 OFF=0
uint8 LOCAL=1
uint8 GLOBAL=2

uint8 RIGHT=1
uint8 LEFT=2
uint8 INDIVIDUAL=3
uint8 ABSOLUTE=4
uint8 RELATIVE=8
uint8 COORDINATED=12

# execution time
duration time_to_execute

# primary is RIGHT/ABSOLUTE, secondary is LEFT/RELATIVE
geometry_msgs/Pose pose_primary
geometry_msgs/Twist twist_primary
geometry_msgs/Wrench wrench_primary
geometry_msgs/Pose pose_secondary
geometry_msgs/Twist twist_secondary
geometry_msgs/Wrench wrench_secondary

# gripper position (minimum and maximum defined by gripper calibration)
float64 gripper_right
float64 gripper_left

# operational mode (eg. INDIVIDUAL, COORDINATED, ...)
uint8 mode

# incremental mode (eg. OFF, LOCAL, ...)
# wether to interpret this message w.r.t. the previous posture, with the difference 
# being specified in the world or in the local reference frame 
uint8 incremental
