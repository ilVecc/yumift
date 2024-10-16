import numpy as np

# Maximu deviation in trajectory
MAX_DEVIATION = np.array([0.015, 0.25])*100  # [lin, ang]

# Gain for individual right control
K_P_IR = 4.0   # Gain for positional error
K_O_IR = 8.0   # Gain for angular error
K_F_IR = 0.05  # Gain for force error
K_T_IR = 2.5   # Gain for torque error
# Gain for individual right control
K_P_IL = K_P_IR
K_O_IL = K_O_IR
K_F_IL = K_F_IR
K_T_IL = K_T_IR
# Gain for coordinated absolute control
K_P_CA = 4.0
K_O_CA = 8.0
K_F_CA = 0.02
K_T_CA = 0.4
# Gain for coordinated relative control
K_P_CR = 4.0
K_O_CR = 8.0
K_F_CR = 0.02
K_T_CR = 0.4

# Gains for individual right admittances
ADM_F_M_IR = np.diag([1.0, 1.0, 1.0])
ADM_F_D_IR = None
ADM_F_K_IR = np.diag([30, 30, 30])
ADM_T_M_IR = np.diag([0.01, 0.01, 0.002])
ADM_T_D_IR = None
ADM_T_K_IR = np.diag([2, 2, 0.2])
# Gains for individual left admittances
ADM_F_M_IL = ADM_F_M_IR
ADM_F_D_IL = ADM_F_D_IR
ADM_F_K_IL = ADM_F_K_IR
ADM_T_M_IL = ADM_T_M_IR
ADM_T_D_IL = ADM_T_D_IR
ADM_T_K_IL = ADM_T_K_IR
# Gains for coordinated absolute admittances
ADM_F_M_CA = np.diag([0.75, 0.75, 0.75])
ADM_F_D_CA = None
ADM_F_K_CA = np.diag([30, 30, 30])
ADM_T_M_CA = np.diag([0.01, 0.01, 0.002])
ADM_T_D_CA = None
ADM_T_K_CA = np.diag([2, 2, 0.2])
# Gains for coordinated relative admittances
ADM_F_M_CR = np.diag([0.75, 0.75, 0.75])
ADM_F_D_CR = None
ADM_F_K_CR = np.diag([30, 30, 30])
ADM_T_M_CR = np.diag([0.01, 0.01, 0.002])
ADM_T_D_CR = None
ADM_T_K_CR = np.diag([2, 2, 0.2])

GAINS = {
    "individual": {
        "right": {
            "position": K_P_IR,
            "rotation": K_O_IR,
            "force": K_F_IR,
            "torque": K_T_IR,
            #
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
            #
            "max_deviation": MAX_DEVIATION
        },
        "left": {
            "position": K_P_IL,
            "rotation": K_O_IL,
            "force": K_F_IL,
            "torque": K_T_IL,
            #
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
            #
            "max_deviation": MAX_DEVIATION
        }
    },
    "coordinated": {
        "absolute": {
            "position": K_P_CA,
            "rotation": K_O_CA,
            "force": K_F_CA,
            "torque": K_T_CA,
            #
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
            #
            "max_deviation": MAX_DEVIATION
        },
        "relative": {
            "position": K_P_CR,
            "rotation": K_O_CR,
            "force": K_F_CR,
            "torque": K_T_CR,
            #
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
            #
            "max_deviation": MAX_DEVIATION
        }
    }
}

