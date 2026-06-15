import numpy as np


class ControllerParameters():
    # (beware that ABB Documentation naming follows [1,2,7,3,4,5,6], but in 
    #  this entire codebase and `abb_drivers` use [1,2,3,4,5,6,7])
        
    # controller rate in Hz, also defined in `kdl_kinematics.cpp` (both needs to be the same!)
    # this is a desired value, thus unreliable! check the actual rate
    update_rate = 250
    dt = 1/update_rate
    
    # initial configuration
    CONFIG_INIT_POS = np.array([ 1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0,   # right arm
                                -1.0, -2.0,  1.2, 0.6,  2.0, 1.0, 0.0])  # left arm

    # reset configuration 
    CONFIG_READY_POS = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                                 -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
    