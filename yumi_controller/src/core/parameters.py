import numpy as np
import quaternion as quat

from dynamics.utils import Frame

# TODO make these loadable

# This class stores all tunable parameters for the controller (only for the parts written in python)
class YumiControllerParameters():
    # Hz the controller is running, is also defined in `kdl_kinematics.cpp` (both needs to be the same!)
    update_rate = 50 
    dt = 1/update_rate
    
    # degrees of freedom for the robot (DO NOT TOUCH, obv.)
    dof = 14
    
    # initial state
    init_pos = np.array([ 1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0,   # right arm
                         -1.0, -2.0,  1.2, 0.6,  2.0, 1.0, 0.0])  # left arm
    init_vel = np.zeros(14)

    # reset configuration 
    reset_pos = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                          -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
    
    # calibration configuration
    calibration_pos = np.array([ 0.0, -2.269,  2.356, 0.524, -0.001, 0.699, 0.002,
                                 0.0, -2.269, -2.356, 0.524,  0.001, 0.664, 0.031])

    ######################     HQP INVERSE KINEMATICS     #####################
    # extra objectives that should be included in HQP, if not already required
    feasibility_objectives = {
        "joint_position_bound": True,  # UNSAFE if False
        "joint_velocity_bound": True,  # UNSAFE/CLIPPING if False
        "elbow_collision": True,       # UNSAFE if False
        "gripper_collision": False,    # UNSAFE if False (only used for individual control)
        "joint_potential": False        # WEIRD CONFIGURATION if False
    }
    
    # max values before joints becomes saturated, values are from
    # https://search.abb.com/library/Download.aspx?DocumentID=3HAC052982-001&LanguageCode=en
    # (scale to 0.99 as an extra safety boundary)
    joint_position_bound_upper = 0.99 * np.radians([ 168.5,   43.5,  168.5,     80,  290, 138,  229])
    joint_position_bound_lower = 0.99 * np.radians([-168.5, -143.5, -168.5, -123.5, -290, -88, -229])
    
    # joint velocity limit [rad/s]
    # (beware that ABB Documentation naming follows [1,2,7,3,4,5,6], but in 
    #  this entire codebase and `abb_drivers` use [1,2,3,4,5,6,7])
    joint_velocity_bound = 1.5 * np.array([1., 1., 1., 1., 1., 1., 1.])

    # gripper collision avoidance (Only for individual motion and not coordinated motion)
    gripper_min_distance = 0.120  # closet allowed distance in [m]

    # elbow collision avoidance  
    elbow_min_distance = 0.200  # closes the elbows can be to each other in [m]
    
    # For joint potential, defining a neutral pose to move towards
    neutral_pos = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                            -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
    ###########################################################################

    # TODO move these frames to the urdf (?)
    # set local offset between "yumi_link_7_*" and "yumi_grip_*" which becomes the frame controlled
    frame_local_arm_to_gripper_right = Frame(position=np.array([0, 0, 0.166]), quaternion=quat.one)
    frame_local_arm_to_gripper_left  = Frame(position=np.array([0, 0, 0.166]), quaternion=quat.one)
    frame_local_yumi_to_world        = Frame(position=np.array([0.181, 0, 0]), quaternion=quat.from_rotation_vector([0, 0, -np.pi/2]))
