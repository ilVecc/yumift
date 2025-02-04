from typing import Tuple

import numpy as np

from .routine_sm import Routine
from .parameters import Parameters
from dynamics.utils import RobotState
from trajectory.polynomial import CubicTrajectory


class JointStateRoutine(Routine):

    def __init__(self, name: str, joint_position: np.ndarray, min_time: float = 2) -> None:
        super().__init__(name)
        self._des_joint_pos = joint_position
        self._final_time_min = min_time
        self._final_time = self._final_time_min
        self._time = 0
        self._max_speed = 1  # rad/s

    def init(self, robot_state_init: RobotState) -> None:
        current_joint_position = robot_state_init.joint_pos
        max_error = np.max(self._des_joint_pos - current_joint_position)
        min_time = max_error / self._max_speed
        self._final_time = max(min_time, self._final_time_min)
        self._time = 0
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(
            current_joint_position, np.zeros(14),
            self._des_joint_pos, np.zeros(14),
            self._final_time)

    def action(self, robot_state_curr: RobotState) -> Tuple[dict, bool]:
        current_joint_position = robot_state_curr.joint_pos

        # advance by time step
        self._time += Parameters.dt  # TODO super wrong, use real data

        # if final time is reached, exit with "done" state
        if self._time <= self._final_time:
            q, dq, _ = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, self._time)
            vel = dq + (q - current_joint_position)
            done = False
        else:
            vel = np.zeros(Parameters.dof)
            done = True

        action = {
            "control_space": "joint_space",
            "joint_velocities": vel}
        return action, done

    def finish(self, robot_state_final: RobotState) -> None:
        pass


class CalibPoseRoutine(JointStateRoutine):
    def __init__(self) -> None:
        super().__init__("calib_pose", Parameters.calib_pos)

class ReadyPoseRoutine(JointStateRoutine):
    def __init__(self) -> None:
        super().__init__("ready_pose", Parameters.reset_pos)