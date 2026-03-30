import numpy as np


class HQPParameters():
    
    # extra objectives that should be included in HQP, if not already required
    safety_objectives = {
        "joint_position_bound": True,  # UNSAFE if False
        "joint_velocity_bound": True,  # UNSAFE/CLIPPING if False
        "elbow_collision": True,       # UNSAFE if False
        "gripper_collision": False,    # UNSAFE if False (only used for individual control)
        "joint_potential": False        # WEIRD CONFIGURATION if False
    }
    
    # gripper collision avoidance (Only for individual motion and not coordinated motion)
    grippers_min_distance = 0.120  # closet allowed distance in [m]

    # elbow collision avoidance  
    elbows_min_distance = 0.200  # closes the elbows can be to each other in [m]
    
    # for joint potential, be less strict on the last wrist joints
    potential_weight = np.array([1., 1., 1., 1., 1., 1., 0.25, 
                                 1., 1., 1., 1., 1., 1., 0.25])
