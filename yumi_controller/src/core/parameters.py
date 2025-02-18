import numpy as np

# TODO make these loadable

class Parameters():
    """ This class stores all tunable parameters for the controller.
    """
    
    # degrees of freedom for the robot (DO NOT TOUCH, obv.)
    dof = 14
    
    # controller rate in Hz, also defined in `kdl_kinematics.cpp` (both needs to be the same!)
    # this is a desired value, thus unreliable! check the actual rate
    update_rate = 500
    dt = 1/update_rate
    
    # initial configuration
    init_pos = np.array([ 1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0,   # right arm
                         -1.0, -2.0,  1.2, 0.6,  2.0, 1.0, 0.0])  # left arm
    init_vel = np.zeros(14)

    # reset configuration 
    reset_pos = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                          -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
    
    # calibration configuration
    calib_pos = np.array([ 0.0, -2.270, -2.356, 0.524, 0.0, 0.670, 0.0,
                           0.0, -2.270,  2.356, 0.524, 0.0, 0.670, 0.0])

    ######################     HQP INVERSE KINEMATICS     #####################
    # extra objectives that should be included in HQP, if not already required
    safety_objectives = {
        "joint_position_bound": True,  # UNSAFE if False
        "joint_velocity_bound": True,  # UNSAFE/CLIPPING if False
        "elbow_collision": True,       # UNSAFE if False
        "gripper_collision": False,    # UNSAFE if False (only used for individual control)
        "joint_potential": True        # WEIRD CONFIGURATION if False
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
    grippers_min_distance = 0.120  # closet allowed distance in [m]

    # elbow collision avoidance  
    elbows_min_distance = 0.200  # closes the elbows can be to each other in [m]
    
    # for joint potential, defining a neutral pose to move towards
    neutral_pos = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                            -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
    
    # for joint potential, be less strict on the last wrist joints
    potential_weight = np.array([1., 1., 1., 1., 1., 1., 0.5, 
                                 1., 1., 1., 1., 1., 1., 0.5])
    ###########################################################################
    
    ####################     SIMPLE INVERSE KINEMATICS     ####################
    q_avg = np.concatenate([(joint_position_bound_upper + joint_position_bound_lower) / 2, 
                            (joint_position_bound_upper + joint_position_bound_lower) / 2])
    q_span = np.concatenate([joint_position_bound_upper - joint_position_bound_lower,
                             joint_position_bound_upper - joint_position_bound_lower])
    k0 = 10
    
    @classmethod
    def secondary_nothing(cls, q, dq): 
        return np.zeros(cls.dof)
    
    @classmethod
    def secondary_neutral(cls, q, dq): 
        return - cls.k0 * (1/cls.dof) * (q - cls.neutral_pos) / cls.q_span ** 2
    
    @classmethod
    def secondary_center(cls, q, dq): 
        return - cls.k0 * (1/cls.dof) * (q - cls.q_avg) / cls.q_span ** 2
    ###########################################################################
