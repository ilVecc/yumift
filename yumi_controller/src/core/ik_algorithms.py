from typing import Dict

import numpy as np

from . import hqp_tasks
from .ik_solver import IKAlgorithm
from .parameters import Parameters
from .robot_state import YumiCoordinatedRobotState
from dynamics.hqp import HQPSolver, HQPTaskError


class HQPIKAlgorithm(IKAlgorithm):

    def __init__(self):
        super().__init__(name="hqp", can_init_late=True)

    # TODO handle this with the usual  .register_task("name", obj_task)  APIs
    #      though this time an ordered dict is needed
    def init(self):
        """ Sets up the HQP solver and the desired tasks
        """
        self._hqp_solver = HQPSolver()
        self._tasks: Dict[str, hqp_tasks.Task] = {}

        # joint position limit
        self._tasks["joint_position_bound"] = hqp_tasks.JointPositionBoundsTask(
            dof=Parameters.dof,
            bounds_lower=np.hstack([Parameters.joint_position_bound_lower, Parameters.joint_position_bound_lower]),
            bounds_upper=np.hstack([Parameters.joint_position_bound_upper, Parameters.joint_position_bound_upper]),
            timestep=Parameters.dt)

        # joint velocity limit
        self._tasks["joint_velocity_bound"] = hqp_tasks.JointVelocityBoundsTask(
            dof=Parameters.dof,
            bounds_lower=-np.hstack([Parameters.joint_velocity_bound, Parameters.joint_velocity_bound]),
            bounds_upper=np.hstack([Parameters.joint_velocity_bound, Parameters.joint_velocity_bound])).compute()  # constant

        # control objective
        self._tasks["individual_control"] = hqp_tasks.IndividualControl(dof=Parameters.dof)
        self._tasks["right_control"] = hqp_tasks.RightControl(dof=Parameters.dof)
        self._tasks["left_control"] = hqp_tasks.LeftControl(dof=Parameters.dof)
        self._tasks["coordinated_control"] = hqp_tasks.CoordinatedControl(dof=Parameters.dof)
        self._tasks["absolute_control"] = hqp_tasks.AbsoluteControl(dof=Parameters.dof)
        self._tasks["relative_control"] = hqp_tasks.RelativeControl(dof=Parameters.dof)

        # elbow collision avoidance
        self._tasks["self_collision_elbow"] = hqp_tasks.ElbowProximity(
            dof=Parameters.dof,
            min_dist=Parameters.elbows_min_distance,
            timestep=Parameters.dt)

        # end effector collision avoidance
        self._tasks["end_effector_collision"] = hqp_tasks.EndEffectorProximity(
            dof=Parameters.dof,
            min_dist=Parameters.grippers_min_distance,
            timestep=Parameters.dt)

        # joint potential 
        self._tasks["joint_position_potential"] = hqp_tasks.JointPositionPotential(
            dof=Parameters.dof,
            default_pos=Parameters.neutral_pos,
            weights=Parameters.potential_weight,
            timestep=Parameters.dt)

    def solve(self, action: dict, state: YumiCoordinatedRobotState):
        """ Sets up stack of tasks and solves the inverse kinematics problem for
            individual or coordinated manipulation
        """
        # add extra feasibility tasks (if not already included)
        for key, value in Parameters.safety_objectives.items():
            if key not in action:
                action[key] = value

        dt = action["timestep"]

        # stack of tasks, in descending hierarchy
        SoT = []
        # (1) velocity bound
        if action["joint_velocity_bound"]:
            SoT.append(self._tasks["joint_velocity_bound"])

        # (2) position bound
        if action["joint_position_bound"]:
            SoT.append(self._tasks["joint_position_bound"].compute(joint_position=state.joint_pos, timestep=dt))

        # (3) elbow proximity limit task
        if action["elbow_collision"]:
            SoT.append(self._tasks["self_collision_elbow"].compute(
                jacobian_elbows=state.jacobian_elbows,
                pose_elbow_r=state.pose_elbow_r,
                pose_elbow_l=state.pose_elbow_l,
                timestep=dt))

        # (4) velocity command task
        if action["control_space"] == "individual":
            # (4.0) gripper collision avoidance
            if action["gripper_collision"]:
                SoT.append(self._tasks["end_effector_collision"].compute(
                    jacobian_grippers=state.jacobian_grippers,
                    pose_gripper_r=state.pose_gripper_r,
                    pose_gripper_l=state.pose_gripper_l,
                    timestep=dt))

            # (4.1) individual control
            if "right_velocity" in action and "left_velocity" in action:
                SoT.append(self._tasks["individual_control"].compute(
                    control_vel=np.concatenate([action["right_velocity"], action["left_velocity"]]),
                    jacobian_grippers=state.jacobian_grippers))
            # (4.2) right motion
            elif "right_velocity" in action:
                SoT.append(self._tasks["right_control"].compute(
                    control_vel_right=action["right_velocity"],
                    jacobian_grippers_right=state.jacobian_gripper_r))
            # (4.3) left motion
            elif "left_velocity" in action:
                SoT.append(self._tasks["left_control"].compute(
                    control_vel_left=action["left_velocity"],
                    jacobian_grippers_left=state.jacobian_gripper_l))

            else:
                print(f"When using individual control mode, \"right_velocity\" and/or \"left_velocity\" must be specified")
                return np.zeros(Parameters.dof)

        elif action["control_space"] == "coordinated":
            # (4.1) coordinated motion
            if "relative_velocity" in action and "absolute_velocity" in action:
                SoT.append(self._tasks["coordinated_control"].compute(
                    control_vel=np.concatenate([action["absolute_velocity"], action["relative_velocity"]]),
                    jacobian_coordinated=state.jacobian_coordinated))
            # (4.2) relative motion
            elif "relative_velocity" in action:
                SoT.append(self._tasks["relative_control"].compute(
                    control_vel_rel=action["relative_velocity"],
                    jacobian_coordinated_rel=state.jacobian_coordinated_rel))
            # (4.3) absolute motion
            elif "absolute_velocity" in action:
                SoT.append(self._tasks["absolute_control"].compute(
                    control_vel_abs=action["absolute_velocity"],
                    jacobian_coordinated_abs=state.jacobian_coordinated_abs))

            else:
                print(f"When using individual control mode, \"absolute_velocity\" and/or \"relative_velocity\" must be specified")
                return np.zeros(Parameters.dof)
        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(Parameters.dof)

        # (5) joint potential task (tries to keep the robot in a natural configuration)
        if action["joint_potential"]:
            SoT.append(self._tasks["joint_position_potential"].compute(joint_position=state.joint_pos, timestep=dt))

        # solve HQP problem
        try:
            vel = self._hqp_solver.solve(SoT=SoT)
        except HQPTaskError as ex:
            print(f"Error in the HQP solver: {ex}")
            # TODO what to do with this?
            # print("Stopping EGM for safety")
            # try:
            #     self._stop_egm.call()
            # except Exception:
            #     print("Failed to stop EGM (ignore if simulation does not support EGM)")
            # stop everything (extra fail-safe step)
            print("Joint velocity set to 0")
            vel = np.zeros(Parameters.dof)

        return vel

    def stop(self):
        pass


class PINVIKAlgorithm(IKAlgorithm):

    def __init__(self):
        super().__init__(name="pinv", can_init_late=True)

    def init(self):
        """ Sets up the pseudo-inverse solver
        """
        pass

    def solve(self, action: dict, state: YumiCoordinatedRobotState):

        jacobian = None
        xdot = np.zeros(12)

        if action["control_space"] == "individual":
            xdot[0:6] = action.get("right_velocity", np.zeros(6))
            xdot[6:12] = action.get("left_velocity", np.zeros(6))
            jacobian = state.jacobian_grippers

        elif action["control_space"] == "coordinated":
            xdot[0:6] = action.get("absolute_velocity", np.zeros(6))
            xdot[6:12] = action.get("relative_velocity", np.zeros(6))
            jacobian = state.jacobian_coordinated

        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(Parameters.dof)

        # TODO `state.joint_pos` is a `np.concat`, veeeeery slow
        joint_pos = np.zeros(14)
        for i in range(7):
            joint_pos[i] = state.joint_pos_r[i]
            joint_pos[i+7] = state.joint_pos_l[i]
        jacobian_pinv = np.linalg.pinv(jacobian)
        vel = jacobian_pinv @ xdot \
            + (np.eye(Parameters.dof) - jacobian_pinv @ jacobian) @ Parameters.secondary_neutral(joint_pos, None)  # `state.joint_vel` not needed

        return vel

    def stop(self):
        pass