# REFERENCE TYPES CONSTANTS
# -------------------------
# Given the initial pose as `bX_i` (pose X in base frame), compute target pose 
# `bX_t` using the provided transformation `xHy` (homogenous from frame `y` to 
# frame `x`) or poses `zX_t` (target pose X in frame `z`)
#
# {b} <--+--0-- [t] := bX_t
#  4     |      |||     
#        |      bHt  (given)
#        |
#        +----- {e} <--2-- [t] := eX_t
#        |      |||        |||
#        |      |||        eHt  (given)
#        |      |||
#        |      bHe := bX_i  (known)
#        |
#        `--3-- {f} <--1-- [t] := fX_t
#               |||        |||
#               |||        fHt  (given)
#               |||
#               bHf := bX_f  (known)
#
uint8 REFERENCE_BASE=0      # provided  `bX_t`  compute nothing (since bHb = I)    changing pose IN base/static frame   (same as REFERENCE_LOCAL with `ref_frame="base"`)
uint8 REFERENCE_TOOL=1      # provided  `eX_t`  compute bX_t = bHe @ eX_t          changing pose IN end-effector frame  (same as REFERENCE_LOCAL with `ref_frame="tool"`)
uint8 REFERENCE_LOCAL=2     # provided  `fX_t`  compute bX_t = bHf @ fX_t          changing pose IN custom/object frame (essentially locally move the initial pose to the new frame)
uint8 REFERENCE_GLOBAL=4    # provided  `bHb'`  compute bX_t = bHb' @ bX_i         changing pose OF base/static frame   (same as REFERENCE_FRAME with `ref_frame="base"`)
uint8 REFERENCE_FRAME=8     # provided  `fHf'`  compute bX_t = bHf @ fHf' @ fX_i   changing pose OF custom/object frame (essentially reinterpret the initial pose in the new frame)
#                                                            = (bHf @ fHf' / bHf) @ bX_i

# MOTION TYPES CONSTANTS
# ----------------------
# How to interpret "primary" and "secondary" poses.
# RIGHT, LEFT, ABSOLUTE, RELATIVE imply that their counterpart can be either 
# ignored or set to 0 depending on the chosen controller.
uint8 MOTION_INDIVIDUAL=3
uint8 MOTION_RIGHT=1
uint8 MOTION_LEFT=2
uint8 MOTION_COORDINATED=12
uint8 MOTION_ABSOLUTE=4
uint8 MOTION_RELATIVE=8




# execution time
duration time_to_execute

# reference type (eg. BASE, TOOL, LOCAL, ...)
# how to interpret the provided poses
uint8 reference
# reference frame of the provided posture
# must be set when REFERENCE_LOCAL or REFERENCE_FRAME are chosen
# the frame is used for both poses (except for RELATIVE)
string ref_frame

# motion type (eg. INDIVIDUAL, COORDINATED, ...)
uint8 motion

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
