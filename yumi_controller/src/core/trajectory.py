from typing import List

import numpy as np
import quaternion as quat

from trajectory.base import Param, MultiParam, MultiTrajectory, FakeTrajectory
from trajectory.base_impl import PoseParam
from trajectory.polynomial import CubicPosePath

from dynamics.quat_utils import quat_diff


class YumiParam(Param):
    """ Class for storing Yumi trajectory parameters.
    """
    def __init__(self,
            pos_r: np.ndarray = np.array([0.4, -0.2, 0.2]), 
            rot_r: np.quaternion = quat.x, 
            vel_r: np.ndarray = None, 
            grip_r: float = 0,
            pos_l: np.ndarray = np.array([0.4,  0.2, 0.2]), 
            rot_l: np.quaternion = quat.x, 
            vel_l: np.ndarray = None, 
            grip_l: float = 0
        ) -> None:
        vel = None
        # TODO this is expensive
        # if vel_r is None or vel_l is None:
        #     vel = None
        # else:
        #     vel = np.concatenate([vel_r, vel_l])
        super().__init__([pos_r, rot_r, pos_l, rot_l], vel)
        self.pose_right = PoseParam(pos_r, rot_r, vel_r)
        self.pose_left = PoseParam(pos_l, rot_l, vel_l)
        self.grip_right = grip_r
        self.grip_left = grip_l
    
    @property
    def position(self):
        return np.concatenate([self.pose_right.pos, self.pose_left.pos])

    @property
    def rotation(self):
        return np.stack([self.pose_right.rot, self.pose_left.rot])
    
    @property
    def velocity(self):
        return np.concatenate([self.pose_right.vel, self.pose_left.vel])


# just for a cleaner interface
class YumiTrajectoryParam(MultiParam[YumiParam]):
    def __init__(self, param: YumiParam, duration: int)-> None: 
        super().__init__(param, duration)
    

# @sampled
class YumiTrajectory(MultiTrajectory[YumiParam]):
    """ Generates a trajectory from trajectory parameters, the process is identical for individual and coordinated manipulation.
        The variable names follows individual motion with left and right. This means when coordinate manipulation is used, right
        is absolute motion and left becomes relative motion.
    """
    def __init__(self) -> None:
        super().__init__(FakeTrajectory())
        self.traj_right = CubicPosePath()
        self.traj_left = CubicPosePath()
    
    @staticmethod
    def _calculate_intermediate_velocity_linear(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
        t2: float, t3: float,
        eps: float = 1e-3
    ) -> np.ndarray:
        """ Calculates transitional velocities between trajectory parameters
            :param p1: position at point before
            :param p2: position at current point
            :param p3: position at point after
            :param t2: time between p1 and p2
            :param t3: time between p2 and p3
            :Returns a velocity for p2
        """
        vel = np.zeros(3)
        p12 = p2 - p1
        p23 = p3 - p2
        vm1 = p12/t2  # avg velocity for previous segment
        vm2 = p23/t3  # avg velocity for next segment
        # if velocities are close to zero or change direction then the transitional velocity 
        # is set to zero otherwise, it is the average of the velocities of two segments
        vel = ( 0.5 * (vm1 + vm2) ) * ( (np.abs(vm1) >= eps) * (np.abs(vm2) >= eps) * (np.sign(vm1) == np.sign(vm2)) )
        return vel
    
    @staticmethod
    def _calculate_intermediate_velocity_angular(
        q1: np.quaternion, q2: np.quaternion, q3: np.quaternion, 
        t2: float, t3: float,
        eps: float = 1e-3
    ) -> np.ndarray:
        """ Calculates transitional velocities between trajectory parameters
            :param q1: quaternion at point before
            :param q2: quaternion at current point
            :param q3: quaternion at point after
            :param t2: time between q1 and q2
            :param t3: time between q2 and q3
            :Returns a velocity for q2
        """
        vel = np.zeros(3)
        q12 = quat_diff(q1, q2)
        q23 = quat_diff(q2, q3)
        # `quat.as_rotation_vector()` === `2*np.log().vec` but batched (so slower)
        vm1 = 2*np.log(q12).vec/t2  # avg velocity for previous segment
        vm2 = 2*np.log(q23).vec/t3  # avg velocity for next segment
        # if velocities are close to zero or change direction then the transitional velocity 
        # is set to zero otherwise, it is the average of the velocities of two segments.
        # we use `*` here instead of `and` because all there are vectors, and 
        # `and` only works on scalars
        vel = ( 0.5 * (vm1 + vm2) ) * ( (np.abs(vm1) >= eps) * (np.abs(vm2) >= eps) * (np.sign(vm1) == np.sign(vm2)) )
        return vel
    
    def update(self, path_parameters: List[MultiParam[YumiParam]]) -> None:
        """ Updates the inner trajectory when new path parameterns are been received
            :param path_parameters: list of path parameters
        """
        # calculate the transition velocities between the path parameters if necessary
        params_right = [pp.param.pose_right for pp in path_parameters]
        if params_right[0].vel is None:
            params_right[0].vel = np.zeros(6)
        for i in range(1, len(params_right)-1):
            if params_right[i].vel is None:
                params_right[i].vel = np.zeros(6)
                params_right[i].vel[0:3] = YumiTrajectory._calculate_intermediate_velocity_linear(
                    params_right[i-1].pos, params_right[i].pos, params_right[i+1].pos,
                    path_parameters[i].duration, path_parameters[i+1].duration)
                params_right[i].vel[3:6] = YumiTrajectory._calculate_intermediate_velocity_angular(
                    params_right[i-1].rot, params_right[i].rot, params_right[i+1].rot,
                    path_parameters[i].duration, path_parameters[i+1].duration)
        if params_right[-1].vel is None:
            params_right[-1].vel = np.zeros(6)
        
        params_left = [pp.param.pose_left for pp in path_parameters]
        if params_left[0].vel is None:
            params_left[0].vel = np.zeros(6)
        for i in range(1, len(params_left)-1):
            if params_left[i].vel is None:
                params_left[i].vel = np.zeros(6)
                params_left[i].vel[0:3] = YumiTrajectory._calculate_intermediate_velocity_linear(
                    params_left[i-1].pos, params_left[i].pos, params_left[i+1].pos,
                    path_parameters[i].duration, path_parameters[i+1].duration)
                params_left[i].vel[3:6] = YumiTrajectory._calculate_intermediate_velocity_angular(
                    params_left[i-1].rot, params_left[i].rot, params_left[i+1].rot,
                    path_parameters[i].duration, path_parameters[i+1].duration)
        if params_left[-1].vel is None:
            params_left[-1].vel = np.zeros(6)
        
        super().update(path_parameters)
        self.traj_right.update( [MultiParam[PoseParam](pp.param.pose_right, pp.duration) for pp in path_parameters] )
        self.traj_left.update( [MultiParam[PoseParam](pp.param.pose_left, pp.duration) for pp in path_parameters] )
    
    def is_new_segment(self):
        """ Returns True if a new segment has been entered, only shows true
            once for each segment. Used to check if new gripper commands should be sent.
        """
        return self.traj_right.is_new_segment() or self.traj_left.is_new_segment()

    def compute(self, t) -> YumiParam:
        """ Updates the desired velocities and target position based on the current 
            trajectory segment, moving to the next one if necessary.
        """
        # since we use a FakeTrajectory and rely on two other inner trajectories,
        # there should be no need to call super().compute(t), but the grippers 
        # are not handled by the two trajectories, so calling the method is needed
        # in order to update the state of the grippers via ._update_segment()
        super().compute(t)
                
        # we must be override this method since we provided a FakeTrajectory object 
        pose_r = self.traj_right.compute(t)
        pose_l = self.traj_left.compute(t)
        
        # get desired gripper position (gripper must be updated once per segment)
        # TODO allow for intra-segment update of gripper
        desired_gripper_right = self._path_params[self._segment_idx_curr].param.grip_right
        desired_gripper_left = self._path_params[self._segment_idx_curr].param.grip_left

        return YumiParam(pose_r.pos, pose_r.rot, pose_r.vel, desired_gripper_right, pose_l.pos, pose_l.rot, pose_l.vel, desired_gripper_left)
