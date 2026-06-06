from typing import Dict, Optional, Callable

import numpy as np
from numpy.typing import ArrayLike

from .pinv_tasks import secondary_neutral, secondary_center, secondary_nothing, YumiSizedArray
from .hqp_tasks import *
from .hqp_parameters import HQPParameters
from .solver import IKAlgorithm
from ..common.device import YumiDualDeviceState
from ..common.controller_base import MixedVelocityYumiAction

from yumift_common.constants import YumiRobotConstants

from dynamicals.solvers.hqp import HQPSolver, HQPTaskError, Task


class HQPIKAlgorithm(IKAlgorithm):

    def __init__(self):
        super().__init__(name="hqp", can_init_late=True)

    # TODO handle this with the usual  .register_task("name", obj_task)  APIs
    #      though this time an ordered dict (or dict->array) is needed
    def init(self):
        """ Sets up the HQP solver and the desired tasks
        """
        self._hqp_solver = HQPSolver()
        self._tasks: Dict[str, Task] = {}

        # joint position limit
        self._tasks["joint_position_bound"] = JointPositionBoundsTask(
            dof=YumiRobotConstants.DOF,
            bounds_lower=np.hstack([YumiRobotConstants.JOINT_POS_UB, YumiRobotConstants.JOINT_POS_LB]),
            bounds_upper=np.hstack([YumiRobotConstants.JOINT_POS_UB, YumiRobotConstants.JOINT_POS_LB]))

        # joint velocity limit
        self._tasks["joint_velocity_bound"] = JointVelocityBoundsTask(
            dof=YumiRobotConstants.DOF,
            bounds_lower=-np.hstack([YumiRobotConstants.JOINT_VEL_AB, YumiRobotConstants.JOINT_VEL_AB]),
            bounds_upper=np.hstack([YumiRobotConstants.JOINT_VEL_AB, YumiRobotConstants.JOINT_VEL_AB]))

        # control objective
        self._tasks["individual_control"] = IndividualControl(dof=YumiRobotConstants.DOF)
        self._tasks["right_control"] = RightControl(dof=YumiRobotConstants.DOF)
        self._tasks["left_control"] = LeftControl(dof=YumiRobotConstants.DOF)
        self._tasks["coordinated_control"] = CoordinatedControl(dof=YumiRobotConstants.DOF)
        self._tasks["absolute_control"] = AbsoluteControl(dof=YumiRobotConstants.DOF)
        self._tasks["relative_control"] = RelativeControl(dof=YumiRobotConstants.DOF)

        # elbow collision avoidance
        self._tasks["self_collision_elbow"] = ElbowProximity(
            dof=YumiRobotConstants.DOF,
            min_dist=HQPParameters.elbows_min_distance)

        # end effector collision avoidance
        self._tasks["end_effector_collision"] = EndEffectorProximity(
            dof=YumiRobotConstants.DOF,
            min_dist=HQPParameters.grippers_min_distance)

        # joint potential 
        self._tasks["joint_position_potential"] = JointPositionPotential(
            dof=YumiRobotConstants.DOF,
            default_pos=YumiRobotConstants.JOINT_POS_NEUTRAL,
            weights=HQPParameters.potential_weight)

        # TODO remove me
        # cache
        self._cache_joint_state = np.zeros((YumiRobotConstants.DOF,))
        self._cache_velocities = np.zeros(12)

    def solve(self, action: MixedVelocityYumiAction, state: YumiDualDeviceState):
        """ Sets up stack of tasks and solves the inverse kinematics problem for
            individual or coordinated manipulation
        """
        # add extra feasibility tasks (if not already included)
        for key, value in HQPParameters.safety_objectives.items():
            if key not in action:
                action[key] = value

        self._cache_joint_state[0:7] = state.joint_pos_r
        self._cache_joint_state[7:14] = state.joint_pos_l

        dt = action["timestep"]
        
        # stack of tasks, in descending hierarchy
        SoT = []
        # (1) velocity bound
        if action["joint_velocity_bound"]:
            SoT.append(self._tasks["joint_velocity_bound"].compute())  # constant

        # (2) position bound
        if action["joint_position_bound"]:
            SoT.append(self._tasks["joint_position_bound"].compute(
                joint_position=self._cache_joint_state, 
                timestep=dt))

        # (3) elbow proximity limit task
        if action["elbow_collision"]:
            SoT.append(self._tasks["self_collision_elbow"].compute(
                jacobian_elbows=state.jacobian_elbows,
                pose_elbow_r=state.pose_elbow_r,
                pose_elbow_l=state.pose_elbow_l,
                timestep=dt))

        # (4) velocity command task
        if action["control_space"] == MixedVelocityYumiAction.ControlSpace.INDIVIDUAL:
            # (4.0) gripper collision avoidance
            if action["gripper_collision"]:
                SoT.append(self._tasks["end_effector_collision"].compute(
                    jacobian_grippers=state.jacobian_grippers,
                    pose_gripper_r=state.pose_gripper_r,
                    pose_gripper_l=state.pose_gripper_l,
                    timestep=dt))

            # (4.1) individual control
            if "velocity_right" in action and "velocity_left" in action:
                self._cache_velocities[:6] = action["velocity_right"]
                self._cache_velocities[6:] = action["velocity_left"]
                SoT.append(self._tasks["individual_control"].compute(
                    control_vel=self._cache_velocities,
                    jacobian_grippers=state.jacobian_grippers))
            # (4.2) right motion
            elif "velocity_right" in action:
                SoT.append(self._tasks["right_control"].compute(
                    control_vel_right=action["velocity_right"],
                    jacobian_grippers_right=state.jacobian_gripper_r))
            # (4.3) left motion
            elif "velocity_left" in action:
                SoT.append(self._tasks["left_control"].compute(
                    control_vel_left=action["velocity_left"],
                    jacobian_grippers_left=state.jacobian_gripper_l))

            else:
                print(f"When using individual control mode, \"velocity_right\" and/or \"velocity_left\" must be specified")
                return np.zeros(YumiRobotConstants.DOF)

        elif action["control_space"] == MixedVelocityYumiAction.ControlSpace.COORDINATED:
            # (4.1) coordinated motion
            if "velocity_relative" in action and "velocity_absolute" in action:
                self._cache_velocities[:6] = action["velocity_absolute"]
                self._cache_velocities[6:] = action["velocity_relative"]
                SoT.append(self._tasks["coordinated_control"].compute(
                    control_vel=self._cache_velocities,
                    jacobian_coordinated=state.jacobian_coordinated))
            # (4.2) relative motion
            elif "velocity_relative" in action:
                SoT.append(self._tasks["relative_control"].compute(
                    control_vel_rel=action["velocity_relative"],
                    jacobian_coordinated_rel=state.jacobian_coordinated_rel))
            # (4.3) absolute motion
            elif "velocity_absolute" in action:
                SoT.append(self._tasks["absolute_control"].compute(
                    control_vel_abs=action["velocity_absolute"],
                    jacobian_coordinated_abs=state.jacobian_coordinated_abs))

            else:
                print(f"When using individual control mode, \"velocity_absolute\" and/or \"velocity_relative\" must be specified")
                return np.zeros(YumiRobotConstants.DOF)
        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(YumiRobotConstants.DOF)

        # (5) joint potential task (tries to keep the robot in a natural configuration)
        if action["joint_potential"]:
            SoT.append(self._tasks["joint_position_potential"].compute(
                joint_position=self._cache_joint_state, 
                timestep=dt))

        # solve HQP problem
        try:
            vel = self._hqp_solver.solve(SoT=SoT)
        except HQPTaskError as ex:
            print(f"Stopping. Error in the HQP solver: {ex}")
            vel = np.zeros(YumiRobotConstants.DOF)

        return vel

    def stop(self):
        pass


class PINVIKAlgorithm(IKAlgorithm):

    def __init__(self, 
        weights : Optional[ArrayLike] = [50., 50., 50., 50., 1., 1., 1.], 
        damping : Optional[float] = 0.01, 
        secondary_obj : Optional[Callable[[YumiSizedArray, YumiSizedArray], YumiSizedArray]] = secondary_neutral
    ):
        """ :param weights: cost for each joint
        """
        super().__init__(name="pinv", can_init_late=True)
        
        self.secondary_obj = secondary_obj if secondary_obj is not None else secondary_nothing
        
        # weighted jacobian (W diagonal matrix)
        #   J+ = W^-1 J' (J W^-1 J')^-1
        if weights is None:
            self.pinv_funct = lambda J : np.linalg.pinv(J)
            self.W = np.ones(YumiRobotConstants.DOF)
        else:
            self.pinv_funct = lambda J : self.L[:,None] * np.linalg.pinv(J * self.L)  # use * instead of @ for performance
            weights = np.asarray(weights)
            if weights.shape == (YumiRobotConstants.DOF,):
                self.W = weights
            elif weights.shape == (YumiRobotConstants.DOF//2,):
                self.W = np.concatenate([weights, weights])
            else:
                raise AttributeError("Weights must be None, 7-DOF or 14-DOF")
        
        # performance can improved for static W using its Cholesky decomposition
        #   W^-1 = L L' = L^2  (L is diagonal as well)
        #   J+ = W^-1 J' (J W^-1 J')^-1
        #      = L (J L)' ((J L) (J L)')^-1
        self.L = np.sqrt(1/self.W)
        
        if damping is not None:
            # Tikhonov regularization
            # J+ = J' (J J' + G G')^-1
            # G = d I  (usually, becoming L2 regularization)
            self.G = damping * np.eye(YumiRobotConstants.EE)
            self.GGT = self.G @ self.G.T
            
            def damped_weighted_pinv(J : np.ndarray):
                JL = J * self.L
                return self.L[:,None] * JL.T @ np.linalg.inv(JL @ JL.T + self.GGT)
            
            self.pinv_funct = damped_weighted_pinv

        self._cached_eye_DOF = np.eye(YumiRobotConstants.DOF)
        
    def init(self):
        """ Sets up the pseudo-inverse solver
        """
        pass
    
    # TODO not all actions are `dict`
    def solve(self, action: dict, state: YumiDualDeviceState):

        jacobian = None
        xdot = np.zeros(12)

        if action["control_space"] == MixedVelocityYumiAction.ControlSpace.INDIVIDUAL:
            xdot[0:6] = action.get("velocity_right", np.zeros(6))
            xdot[6:12] = action.get("velocity_left", np.zeros(6))
            jacobian = state.jacobian_grippers

        elif action["control_space"] == MixedVelocityYumiAction.ControlSpace.COORDINATED:
            xdot[0:6] = action.get("velocity_absolute", np.zeros(6))
            xdot[6:12] = action.get("velocity_relative", np.zeros(6))
            jacobian = state.jacobian_coordinated

        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(YumiRobotConstants.DOF)
        
        # TODO `state.joint_pos` is a `np.concat`, veeeeery slow
        joint_pos = np.zeros(14)
        for i in range(7):
            joint_pos[i] = state.joint_pos_r[i]
            joint_pos[i+7] = state.joint_pos_l[i]
        
        jacobian_pinv = self.pinv_funct(jacobian)
        ortho_proj = self._cached_eye_DOF - jacobian_pinv @ jacobian
        vel = jacobian_pinv @ xdot + ortho_proj @ self.secondary_obj(joint_pos, None)  # `state.joint_vel` not needed

        return vel

    def stop(self):
        pass
