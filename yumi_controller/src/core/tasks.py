import numpy as np

from .hqp import Task
from dynamics import utils


class JacobianControlTask(Task):
    """ Task for controlling a generic jacobian
    """
    def __init__(self, dof: int):
        super().__init__(dof)
        self.constr_type = Task.ConstraintType.EQUAL

    def compute(self, control_vel: np.ndarray, jacobian: np.ndarray):
        """ Updates jacobian and the target velocities
            :param control_vel: shape(n)
            :param jacobian_grippers: shape(n,dof)
        """
        self.constr_mat = jacobian
        self.constr_vec = control_vel
        
        return self


#
# Joint bounds
#

class JointPositionBoundsTask(Task):
    """ Task for keeping joint positions from saturating
    """
    def __init__(self, dof: int, bounds_lower: np.ndarray, bounds_upper: np.ndarray, timestep: float):
        super().__init__(dof, 1e3)
        self.constr_type = Task.ConstraintType.UPPER
        self.bounds_lower = bounds_lower
        self.bounds_upper = bounds_upper
        self.timestep = timestep

    def compute(self, joint_position: np.ndarray):
        constr_mat_lower = -self.timestep * np.eye(self.ndim)
        constr_mat_upper =  self.timestep * np.eye(self.ndim)
        constr_vec_lower = -self.bounds_lower + joint_position
        constr_vec_upper =  self.bounds_upper - joint_position

        self.constr_mat = np.vstack((constr_mat_upper, constr_mat_lower))
        self.constr_vec = np.hstack((constr_vec_upper, constr_vec_lower))
        
        return self


class JointVelocityBoundsTask(Task):
    """ Task for keeping joint velocity within limits
    """
    def __init__(self, dof: int, bounds_lower: np.ndarray, bounds_upper: np.ndarray):
        super().__init__(dof, 1e3)
        self.constr_type = Task.ConstraintType.UPPER
        self.bounds_lower = bounds_lower
        self.bounds_upper = bounds_upper

    def compute(self):
        constr_mat_lower = -np.eye(self.ndim)
        constr_mat_upper =  np.eye(self.ndim)
        constr_vec_lower = -self.bounds_lower
        constr_vec_upper =  self.bounds_upper

        self.constr_mat = np.vstack((constr_mat_upper, constr_mat_lower))
        self.constr_vec = np.hstack((constr_vec_upper, constr_vec_lower))
        
        return self


#
# Potential-based control
#

class JointPositionPotential(Task):
    """ Task for keeping a good joint configuration. 
    """
    def __init__(self, dof: int, default_pos: np.ndarray, weights: np.ndarray, timestep: float):
        super().__init__(dof, 2e2)
        self.constr_type = Task.ConstraintType.EQUAL
        self.timestep = timestep
        self.default_pos = default_pos
        self.weights = weights

    def compute(self, joint_position: np.ndarray):
        """ Sets up constraints for joint potential,
            :param joint_position: current joint state
        """
        
        vec = (self.default_pos - joint_position) * 0.5 * self.weights

        self.constr_mat = 100 * self.timestep * np.eye(self.ndim)
        self.constr_vec = vec
        
        return self


#
# Mode control (jacobian-based)
# TODO these are yumi specific, move to a different file
#

class IndividualControl(JacobianControlTask):
    """ Task for controlling each gripper separately in a combined jacobian.
    """
    def __init__(self, dof: int):
        super().__init__(dof)

    def compute(self, control_vel: np.ndarray, jacobian_grippers: np.ndarray):
        """ Sets up the constraints for individual (right, left) velocity control.
            :param control_vel: np.array([pos_r, rot_r, pos_l, rot_l]) shape(12)
            :param jacobian_grippers: shape(12,dof)
        """
        return super().compute(control_vel, jacobian_grippers)


class RightControl(JacobianControlTask):
    """ Task for controlling the right gripper.
    """
    def __init__(self, dof: int):
        super().__init__(dof)
    
    def compute(self, control_vel_right: np.ndarray, jacobian_grippers_right: np.ndarray):
        """ Sets up the constraints for right velocity control.
            :param control_vel_right: np.array([pos_right, rot_right]) shape(6)
            :param jacobian_grippers_right: shape(6,dof)
        """
        return super().compute(control_vel_right, jacobian_grippers_right)


class LeftControl(JacobianControlTask):
    """ Task for controlling the left gripper.
    """
    def __init__(self, dof: int):
        super().__init__(dof)
    
    def compute(self, control_vel_left: np.ndarray, jacobian_grippers_left: np.ndarray):
        """ Sets up the constraints for left velocity control.
            :param control_vel_left: np.array([pos_left, rot_left]) shape(6)
            :param jacobian_grippers_left: shape(6,dof)
        """
        return super().compute(control_vel_left, jacobian_grippers_left)


class CoordinatedControl(JacobianControlTask):
    """ Task for controlling each coordinated frame separately in a combined jacobian.
    """
    def __init__(self, dof: int):
        super().__init__(dof)

    def compute(self, control_vel: np.ndarray, jacobian_coordinated: np.ndarray):
        """ Sets up the constraints for coordinated (abs, rel) velocity control.
            :param control_vel: np.array([pos_abs, rot_abs, pos_rel, rot_rel]) shape(12)
            :param jacobian_coordinated: shape(12,dof)
        """
        return super().compute(control_vel, jacobian_coordinated)


class AbsoluteControl(JacobianControlTask):
    """ Task for controlling the average of the grippers.
    """
    def __init__(self, dof: int):
        super().__init__(dof)
    
    def compute(self, control_vel_abs: np.ndarray, jacobian_coordinated_abs: np.ndarray):
        """ Sets up the constraints for absolute velocity control.
            :param control_vel_abs: np.array([pos_abs, rot_abs]) shape(6)
            :param jacobian_coordinated_abs: shape(6,dof)
        """
        return super().compute(control_vel_abs, jacobian_coordinated_abs)


class RelativeControl(JacobianControlTask):
    """ Task for controlling the grippers relative to each other.
    """
    def __init__(self, dof: int):
        super().__init__(dof)
    
    def compute(self, control_vel_rel: np.ndarray, jacobian_coordinated_rel: np.ndarray):
        """ Sets up the constraints for relative velocity control.
            :param control_vel_rel: np.array([pos_rel, rot_rel]) shape(6)
            :param jacobian_coordinated_rel: shape(6,dof)
        """
        return super().compute(control_vel_rel, jacobian_coordinated_rel)


class ElbowProximity(Task):
    """ Task for keeping a proximity between the elbows of the robot
    """
    def __init__(self, dof: int, min_dist: float, timestep: float):
        super().__init__(dof, 1e3)
        self.constr_type = Task.ConstraintType.UPPER
        self.timestep = timestep
        self.min_dist = min_dist

    def compute(self, jacobian_elbows: np.ndarray, pose_elbow_r: utils.Frame, pose_elbow_l: utils.Frame):
        """ Sets up the constraint for elbow proximity
            :param jacobian_elbows: combined jacobian of the grippers
            :param pose_elbow_r: pose of the right elbow
            :param pose_elbow_l: pose of the left elbow
        """
        jacobian = np.zeros((2, self.dof))
        jacobian[0, 0:4] = jacobian_elbows[1, 0:4]  # select y-axis
        jacobian[1, 7:11] = jacobian_elbows[7, 4:8]  # select y-axis
        link_mat = np.array([1, -1])

        pos_r_y = pose_elbow_r.pos[1]  # select y-axis
        pos_l_y = pose_elbow_l.pos[1]
        diff_dir, diff_norm = utils.normalize(pos_r_y - pos_l_y, return_norm=True)

        jacobian_proximity = -self.timestep * 10 * diff_dir * link_mat @ jacobian
        
        self.constr_mat = np.expand_dims(jacobian_proximity, axis=0)
        self.constr_vec = -np.array([self.min_dist - diff_norm])
        
        return self


class EndEffectorProximity(Task):
    """ Task for keeping minimum proximity between the grippers 
    """
    def __init__(self, dof: int, min_dist: float, timestep: float):
        super().__init__(dof, 1e3)
        self.constr_type = Task.ConstraintType.UPPER
        self.timestep = timestep
        self.min_dist = min_dist

    def compute(self, jacobian_grippers: np.ndarray, pose_gripper_r: utils.Frame, pose_gripper_l: utils.Frame):
        """ Sets up the constraints collision avoidance, i.e. the grippers will deviate 
            from control command in order to not collide.
            :param jacobian_grippers: shape(12,14) combined jacobian of the grippers
            :param pose_gripper_r: pose of the right gripper 
            :param pose_gripper_l: pose of the left gripper
        """
        jacobian = np.zeros((4, self.dof))
        jacobian[0:2, 0:7] = jacobian_grippers[0:2, 0:7]  # select xy-plane
        jacobian[2:4, 7:14] = jacobian_grippers[6:8, 7:14]  # select xy-plane
        link_mat = np.block([ np.eye(2), -np.eye(2) ])

        pos_r_xy = pose_gripper_r.pos[0:2]  # select xy-plane
        pos_l_xy = pose_gripper_l.pos[0:2]
        diff_dir, diff_norm = utils.normalize(pos_r_xy - pos_l_xy, return_norm=True)

        jacobian_proximity = -self.timestep * 10 * diff_dir @ link_mat @ jacobian
        
        self.constr_mat = np.expand_dims(jacobian_proximity, axis=0)
        self.constr_vec = -np.array([self.min_dist - diff_norm])
        
        return self
