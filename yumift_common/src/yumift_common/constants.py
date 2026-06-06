import numpy as np


class YumiRobotConstants():
    """ This class stores all constants of Yumi. 
        
        NOTHING CAN BE TOUCHED HERE, THESE ARE PHYSICAL CONSTRAITS
        
    """

    DOF_JOINTS_RIGHT = 7
    DOF_JOINTS_LEFT = 7
    DOF_JOINTS = DOF_JOINTS_RIGHT + DOF_JOINTS_LEFT
    DOF = DOF_JOINTS
    DOF_EE_RIGHT = 6
    DOF_EE_LEFT = 6
    DOF_EE = DOF_EE_RIGHT + DOF_EE_LEFT
    EE = DOF_EE

    # calibration configuration (red marks on Yumi)
    JOINT_POS_CALIB = np.array([ 0.0, -2.270, -2.356, 0.524, 0.0, 0.670, 0.0,
                                 0.0, -2.270,  2.356, 0.524, 0.0, 0.670, 0.0])
    
    # TODO move me
    # for joint potential, defining a neutral pose to move towards
    JOINT_POS_NEUTRAL = np.array([ 0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0, 
                                  -0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0])
        
    # max values before joints becomes saturated, values are from
    # https://search.abb.com/library/Download.aspx?DocumentID=3HAC052982-001&LanguageCode=en
    # (scale to 0.99 as an extra safety boundary)
    JOINT_POS_LB = 0.99 * np.radians([-168.5, -143.5, -168.5, -123.5, -290, -88, -229])
    JOINT_POS_UB = 0.99 * np.radians([ 168.5,   43.5,  168.5,     80,  290, 138,  229])
    
    # joint velocity limit [rad/s] (absolute value)
    JOINT_VEL_AB = 1.5 * np.array([1., 1., 1., 1., 1., 1., 1.])
