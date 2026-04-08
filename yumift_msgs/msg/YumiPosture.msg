# Incremental mode constants
# Given the initial pose as `sPi` (pose P in frame s), compute `sPf` using
# homogenous transformations `aTb` (transform from frame b to frame a)
uint8 OFF=0     # sPf (nothing to compute)
uint8 LOCAL=1   # ePf (must compute sPf = sTe @ ePf)
uint8 GLOBAL=2  # iTf (must compute sPf = sPi @ iTf)

# Control mode constants
# How to interpret "primary" and "secondary"
uint8 INDIVIDUAL=0
uint8 RIGHT=1
uint8 LEFT=2
uint8 COORDINATED=3
uint8 ABSOLUTE=4
uint8 RELATIVE=5

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
