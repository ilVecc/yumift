from typing import Tuple

import numpy as np

from ..common.controller_base import YumiDualDeviceAction
from ..common.routine_sm import Routine

from ..common.parameters import ControllerParameters
from yumift_common.constants import YumiRobotConstants

from dynamicals.common.robotics import RobotState
from pathfinder.polynomial import CubicTrajectory


class JointStateRoutine(Routine):

    def __init__(self, name: str, joint_position: np.ndarray, min_time: float = 2) -> None:
        super().__init__(name)
        self._des_joint_pos = joint_position
        self._final_time_min = min_time
        self._final_time = self._final_time_min
        self._time = 0
        self._max_speed = 0.5  # rad/s  TODO never actually used in the trajectory creation

    def init(self, robot_state_init: RobotState) -> None:
        current_joint_position = robot_state_init.joint_pos
        max_error = np.max(np.abs(self._des_joint_pos - current_joint_position))
        min_time = max_error / self._max_speed
        self._final_time = max(min_time, self._final_time_min)
        self._time = 0
        self._coeffs = CubicTrajectory.calculate_coefficients(
            current_joint_position, np.zeros(14),
            self._des_joint_pos, np.zeros(14),
            self._final_time)

    def action(self, robot_state_curr: RobotState) -> Tuple[dict, bool]:
        current_joint_position = robot_state_curr.joint_pos

        # advance by time step
        self._time += ControllerParameters.dt  # TODO super wrong, use real data

        # if final time is reached, exit with "done" state
        if self._time <= self._final_time:
            q, dq, _ = CubicTrajectory.evaluate_at(self._coeffs, self._time)
            vel = dq + (q - current_joint_position)
            done = False
        else:
            vel = np.zeros(YumiRobotConstants.DOF)
            done = True

        action = YumiDualDeviceAction()
        action.control_space(YumiDualDeviceAction.ControlSpace.JOINT_SPACE)
        action.velocity_joints(vel)
        return action, done

    def finish(self, robot_state_final: RobotState) -> None:
        pass


class CalibPoseRoutine(JointStateRoutine):
    def __init__(self) -> None:
        super().__init__("calib_pose", YumiRobotConstants.CONFIG_CALIB)

class ReadyPoseRoutine(JointStateRoutine):
    def __init__(self) -> None:
        super().__init__("ready_pose", ControllerParameters.CONFIG_READY_POS)
