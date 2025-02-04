import numpy as np


###############################################################################
###   CLOSED LOOP INVERSE KINEMATICS
###############################################################################
# Gains for individual right control
CLIK_K_P_IR = 2.0   # Gain for positional error
CLIK_K_O_IR = 8.0   # Gain for angular error
# Gains for individual right control
CLIK_K_P_IL = CLIK_K_P_IR
CLIK_K_O_IL = CLIK_K_O_IR
# Gains for coordinated absolute control
CLIK_K_P_CA = 2.0
CLIK_K_O_CA = 4.0
# Gains for coordinated relative control
CLIK_K_P_CR = 2.0
CLIK_K_O_CR = 4.0
# Maximum deviation in target
CLIK_MAX_DEVIATION = np.array([0.015, 0.25]) * 100  # [lin, ang]  # TODO too high values

###############################################################################
###   DIRECT FORCE FEEDBACK
###############################################################################
# Gains for individual right feedback
DFF_K_F_IR = 0.05  # Gain for force feedback
DFF_K_T_IR = 1.5   # Gain for torque feedback
# Gains for individual right feedback
DFF_K_F_IL = DFF_K_F_IR
DFF_K_T_IL = DFF_K_T_IR
# Gains for coordinated absolute feedback
DFF_K_F_CA = 0.02
DFF_K_T_CA = 0.4
# Gains for coordinated relative feedback
DFF_K_F_CR = 0.02
DFF_K_T_CR = 0.4

###############################################################################
###   ADMITTANCE FORCE FEEDBACK
###############################################################################
# TODO fix these values
# Gains for individual right admittances
ADM_F_M_IR = np.eye(3) * 2.5
ADM_F_D_IR = None
ADM_F_K_IR = np.eye(3) * 10
ADM_T_M_IR = np.diag([0.01, 0.01, 0.02]) * 10
ADM_T_D_IR = None
ADM_T_K_IR = np.diag([0.2, 0.2, 0.2]) * 10
# Gains for individual left admittances
ADM_F_M_IL = ADM_F_M_IR
ADM_F_D_IL = ADM_F_D_IR
ADM_F_K_IL = ADM_F_K_IR
ADM_T_M_IL = ADM_T_M_IR
ADM_T_D_IL = ADM_T_D_IR
ADM_T_K_IL = ADM_T_K_IR
# Gains for coordinated absolute admittances
ADM_F_M_CA = np.eye(3) * 3
ADM_F_D_CA = None
ADM_F_K_CA = np.eye(3) * 30
ADM_T_M_CA = np.diag([0.01, 0.01, 0.002]) * 1000
ADM_T_D_CA = None
ADM_T_K_CA = np.diag([2, 2, 0.2]) * 10000
# Gains for coordinated relative admittances
ADM_F_M_CR = np.eye(3) * 1000
ADM_F_D_CR = None
ADM_F_K_CR = np.eye(3) * 10000
ADM_T_M_CR = np.diag([0.01, 0.01, 0.002]) * 1000
ADM_T_D_CR = None
ADM_T_K_CR = np.diag([2, 2, 0.2]) * 10000


# TODO this structure is ugly
GAINS = {
    "individual": {
        "right": {
            "clik": {
                "position": CLIK_K_P_IR,
                "rotation": CLIK_K_O_IR,
                "max_deviation": CLIK_MAX_DEVIATION,
            },
            "direct_force": {
                "force": DFF_K_F_IR,
                "torque": DFF_K_T_IR,
            },
            "admittance": {
                "force": {
                    "m": ADM_F_M_IR,
                    "d": ADM_F_D_IR,
                    "k": ADM_F_K_IR,
                },
                "torque": {
                    "m": ADM_T_M_IR,
                    "d": ADM_T_D_IR,
                    "k": ADM_T_K_IR,
                }
            },
        },
        "left": {
            "clik": {
                "position": CLIK_K_P_IL,
                "rotation": CLIK_K_O_IL,
                "max_deviation": CLIK_MAX_DEVIATION,
            },
            "direct_force": {
                "force": DFF_K_F_IL,
                "torque": DFF_K_T_IL,
            },
            "admittance": {
                "force": {
                    "m": ADM_F_M_IL,
                    "d": ADM_F_D_IL,
                    "k": ADM_F_K_IL,
                },
                "torque": {
                    "m": ADM_T_M_IL,
                    "d": ADM_T_D_IL,
                    "k": ADM_T_K_IL,
                }
            },
        }
    },
    "coordinated": {
        "absolute": {
            "clik": {
                "position": CLIK_K_P_CA,
                "rotation": CLIK_K_O_CA,
                "max_deviation": CLIK_MAX_DEVIATION,
            },
            "direct_force": {
                "force": DFF_K_F_CA,
                "torque": DFF_K_T_CA,
            },
            "admittance": {
                "force": {
                    "m": ADM_F_M_CA,
                    "d": ADM_F_D_CA,
                    "k": ADM_F_K_CA,
                },
                "torque": {
                    "m": ADM_T_M_CA,
                    "d": ADM_T_D_CA,
                    "k": ADM_T_K_CA,
                }
            },
        },
        "relative": {
            "clik": {
                "position": CLIK_K_P_CR,
                "rotation": CLIK_K_O_CR,
                "max_deviation": CLIK_MAX_DEVIATION,
            },
            "direct_force": {
                "force": DFF_K_F_CR,
                "torque": DFF_K_T_CR,
            },
            "admittance": {
                "force": {
                    "m": ADM_F_M_CR,
                    "d": ADM_F_D_CR,
                    "k": ADM_F_K_CR,
                },
                "torque": {
                    "m": ADM_T_M_CR,
                    "d": ADM_T_D_CR,
                    "k": ADM_T_K_CR,
                }
            },
        }
    }
}

