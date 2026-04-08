import numpy as np
import quaternion as quat

from ..common.device import SleipnerCartesianDeviceState
from dynamicals.utils import Frame
from pathfinder.base_impl import PoseParam


def SleipnerCartesianDeviceState_to_PoseParam(state : SleipnerCartesianDeviceState) -> PoseParam:
    pos = np.array([state.pose_SE2[0], state.pose_SE2[1], 0], copy=False)
    ori = quat.from_rotation_vector([0, 0, state.pose_SE2[2]])
    twist = np.array([state.twist_SE2[0], state.twist_SE2[1], 0, 0, 0, state.twist_SE2[2]], copy=False)
    return PoseParam(pos, ori, twist)

def SleipnerCartesianDeviceState_to_Frame(state : SleipnerCartesianDeviceState) -> Frame:
    pos = np.array([state.pose_SE2[0], state.pose_SE2[1], 0], copy=False)
    ori = quat.from_rotation_vector([0, 0, state.pose_SE2[2]])
    twist = np.array([state.twist_SE2[0], state.twist_SE2[1], 0, 0, 0, state.twist_SE2[2]], copy=False)
    return Frame(pos, ori, twist)
