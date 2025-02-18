import numpy as np
import quaternion as quat

from .parameters import Parameters
from .robot_state import YumiCoordinatedRobotState

from dynamics.control_laws import ControlLawError, AbstractControlLaw, CartesianVelocityControlLaw
from dynamics.systems import AdmittanceWrench
from dynamics.utils import Frame


class YumiIndividualCartesianVelocityControlLaw(AbstractControlLaw):
    """ Generates velocity commands in cartesian space with the law
                dx := dx_tgt + k * (x_tgt - x)
        where
            x, dx           state and speed of the YuMi (either linear or angular)
            x_tgt, dx_tgt   target state and speed for the YuMi (either linear or angular)
        in individual motion only.
    """
    def __init__(self, gains):
        super().__init__(initial_timestep=Parameters.dt)
        self.mode = "individual"
        self.control_right = CartesianVelocityControlLaw(gains["individual"]["right"]["clik"]["position"], gains["individual"]["right"]["clik"]["rotation"], gains["individual"]["right"]["clik"]["max_deviation"])
        self.control_left  = CartesianVelocityControlLaw(gains["individual"]["left"]["clik"]["position"], gains["individual"]["left"]["clik"]["rotation"], gains["individual"]["left"]["clik"]["max_deviation"])
        
    @property
    def current_pose(self):
        """ Return current pose as tuple (pos_right, rot_right, pos_left, rot_left)
        """
        return self.current_pose_individual
    
    @property
    def current_pose_individual(self):
        """
        Return current individual pose as a tuple of (pos_r, rot_r, pos_l, rot_l)
        """
        current_pos_r = np.copy(self.control_right.current_position)
        current_rot_r = np.copy(self.control_right.current_rotation)
        current_pos_l = np.copy(self.control_left.current_position)
        current_rot_l = np.copy(self.control_left.current_rotation)
        return current_pos_r, current_rot_r, current_pos_l, current_rot_l
    
    @property
    def last_target_velocity(self):
        """
        Return last target velocity as a tuple of (vel_1, vel_2) where _1
        is either _right or _absolute and _2 is either _left or _relative 
        """
        return self.last_target_velocity_individual
    
    @property
    def last_target_velocity_individual(self):
        """
        Return last individual target velocity as a tuple of (vel_r, vel_l)
        """
        vel_r_init = np.copy(self.control_right.target_velocity)
        vel_l_init = np.copy(self.control_left.target_velocity)
        return vel_r_init, vel_l_init
    
    def clear(self):
        self.control_right.clear()
        self.control_left.clear()

    def update_current_state(self, yumi_state: YumiCoordinatedRobotState):
        """ 
        Updates individual poses
        """
        self.control_right.update_current_state(yumi_state.pose_gripper_r)
        self.control_left.update_current_state(yumi_state.pose_gripper_l)
    
    def update_desired_state(self, target_state: YumiCoordinatedRobotState):
        """ Updates the desired velocities and target position. 
            ATTENTION: this function uses `pose_gripper_r` and `pose_gripper_l` as
            desired target for both the individual (right and left) and the coordinated 
            (absolute and relative) modes. This behaviour is desired since in this way 
            we avoid calculating the coordinated poses from the individuals ones and 
            vice versa, which is unnecessary since this is a velocity controller.
            This means that `target_state` will be used as individual or coordinated 
            based on the current value of `self.mode`.
        """
        self.control_right.update_desired_state(target_state.pose_gripper_r)
        self.control_left.update_desired_state(target_state.pose_gripper_l)

    def compute_individual_right_target_velocity(self):
        """ Calculates the target velocities for individual right arm control.
        """
        try:
            self.control_right.compute_target_state()
        except ControlLawError as ex:
            # turn off deviation error if gripper collision constraint is active for individual mode
            if not Parameters.safety_objectives["gripper_collision"]:
                raise ex
        return self.control_right.target_velocity

    def compute_individual_left_target_velocity(self):
        """ Calculates the target velocities for individual left arm control.
        """
        try:
            self.control_left.compute_target_state()
        except Exception as ex:
            if not Parameters.safety_objectives["gripper_collision"]:
                raise ex
        return self.control_left.target_velocity

    def compute_target_state(self):
        return self.compute_individual_right_target_velocity(), self.compute_individual_left_target_velocity()


# TODO make this a CartesianVelocityControlLaw
class YumiDualCartesianVelocityControlLaw(AbstractControlLaw):
    """ Generates velocity commands in cartesian space with the law
                dx := dx_tgt + k * (x_tgt - x)
        where
            x, dx           state and speed of the YuMi (either linear or angular)
            x_tgt, dx_tgt   target state and speed for the YuMi (either linear or angular)
        in either individual or coordinated motion.
    """
    def __init__(self, gains):
        super().__init__(initial_timestep=Parameters.dt)
        self.mode = None  # can be either "individual" or "coordinated"
        self.control_right = CartesianVelocityControlLaw(gains["individual"]["right"]["clik"]["position"], gains["individual"]["right"]["clik"]["rotation"], gains["individual"]["right"]["clik"]["max_deviation"])
        self.control_left  = CartesianVelocityControlLaw(gains["individual"]["left"]["clik"]["position"], gains["individual"]["left"]["clik"]["rotation"], gains["individual"]["left"]["clik"]["max_deviation"])
        self.control_abs = CartesianVelocityControlLaw(gains["coordinated"]["absolute"]["clik"]["position"], gains["coordinated"]["absolute"]["clik"]["rotation"], gains["coordinated"]["absolute"]["clik"]["max_deviation"])
        self.control_rel = CartesianVelocityControlLaw(gains["coordinated"]["relative"]["clik"]["position"], gains["coordinated"]["relative"]["clik"]["rotation"], gains["coordinated"]["relative"]["clik"]["max_deviation"])
        
    @property
    def current_pose(self):
        """
        Return current pose as a tuple of (pos_1, rot_1, pos_2, rot_2) where _1
        is either _right or _absolute and _2 is either _left or _relative 
        """
        if self.mode == "individual":
            return self.current_pose_individual
        else:
            return self.current_pose_coordinated
    
    @property
    def current_pose_individual(self):
        """
        Return current individual pose as a tuple of (pos_r, rot_r, pos_l, rot_l)
        """
        current_pos_r = np.copy(self.control_right.current_position)
        current_rot_r = np.copy(self.control_right.current_rotation)
        current_pos_l = np.copy(self.control_left.current_position)
        current_rot_l = np.copy(self.control_left.current_rotation)
        return current_pos_r, current_rot_r, current_pos_l, current_rot_l
    
    @property
    def current_pose_coordinated(self):
        """
        Return current coordinated pose as a tuple of (pos_abs, rot_abs, pos_rel, rot_rel)
        """
        current_pos_abs = np.copy(self.control_abs.current_position)
        current_rot_abs = np.copy(self.control_abs.current_rotation)
        current_pos_rel = np.copy(self.control_rel.current_position)
        current_rot_rel = np.copy(self.control_rel.current_rotation)
        return current_pos_abs, current_rot_abs, current_pos_rel, current_rot_rel
    
    @property
    def last_target_velocity(self):
        """
        Return last target velocity as a tuple of (vel_1, vel_2) where _1
        is either _right or _absolute and _2 is either _left or _relative 
        """
        if self.mode == "individual":
            return self.last_target_velocity_individual
        else:
            return self.last_target_velocity_coordinated
    
    @property
    def last_target_velocity_individual(self):
        """
        Return last individual target velocity as a tuple of (vel_r, vel_l)
        """
        vel_r_init = np.copy(self.control_right.target_velocity)
        vel_l_init = np.copy(self.control_left.target_velocity)
        return vel_r_init, vel_l_init
    
    @property
    def last_target_velocity_coordinated(self):
        """
        Return last coordinated target velocity as a tuple of (vel_abs, vel_rel)
        """
        vel_abs_init = np.copy(self.control_abs.target_velocity)
        vel_rel_init = np.copy(self.control_rel.target_velocity)
        return vel_abs_init, vel_rel_init
    
    def clear(self):
        self.mode = None
        self.control_right.clear()
        self.control_left.clear()
        self.control_abs.clear()
        self.control_rel.clear()

    def update_current_state(self, yumi_state: YumiCoordinatedRobotState):
        """ 
        Updates individual and coordinated poses
        """
        self.control_right.update_current_state(yumi_state.pose_gripper_r)
        self.control_left.update_current_state(yumi_state.pose_gripper_l)
        self.control_abs.update_current_state(yumi_state.pose_abs)
        self.control_rel.update_current_state(yumi_state.pose_rel)
    
    def update_desired_state(self, desired_state: YumiCoordinatedRobotState):
        """ Updates the desired velocities and target position. 
            ATTENTION: this function uses `pose_gripper_r` and `pose_gripper_l` as
            desired target for both the individual (right and left) and the coordinated 
            (absolute and relative) modes. This behaviour is desired since in this way 
            we avoid calculating the coordinated poses from the individuals ones and 
            vice versa, which is unnecessary since this is a velocity controller.
            This means that `target_state` will be used as individual or coordinated 
            based on the current value of `self.mode`.
        """
        self.control_right.update_desired_state(desired_state.pose_gripper_r)
        self.control_left.update_desired_state(desired_state.pose_gripper_l)
        self.control_abs.update_desired_state(desired_state.pose_gripper_r)
        self.control_rel.update_desired_state(desired_state.pose_gripper_l)

    def compute_individual_right_target_velocity(self):
        """ Calculates the target velocities for individual right arm control.
        """
        try:
            self.control_right.compute_target_state()
        except ControlLawError as ex:
            # turn off deviation error if gripper collision constraint is active for individual mode
            if not Parameters.safety_objectives["gripper_collision"]:
                raise ex
        return self.control_right.target_velocity

    def compute_individual_left_target_velocity(self):
        """ Calculates the target velocities for individual left arm control.
        """
        try:
            self.control_left.compute_target_state()
        except Exception as ex:
            if not Parameters.safety_objectives["gripper_collision"]:
                raise ex
        return self.control_left.target_velocity
    
    def compute_coordinated_absolute_target_velocity(self):
        """ Calculates the target velocities for absolute motion i.e. controlling
            the average of the grippers.
        """
        return self.control_abs.compute_target_state()

    def compute_coordinated_relative_target_velocity(self):
        """ Calculates the target velocities for relative motion i.e. controlling
            the grippers relative to each other in absolute frame.
        """
        return self.control_rel.compute_target_state()

    def compute_target_state(self):
        if self.mode == "individual":
            return self.compute_individual_right_target_velocity(), self.compute_individual_left_target_velocity()
        else:
            return self.compute_coordinated_absolute_target_velocity(), self.compute_coordinated_relative_target_velocity()


class YumiDualWrenchFeedbackControlLaw(YumiDualCartesianVelocityControlLaw):
    """
    Generates velocity commands in cartesian space with the law
            dx := dx_tgt + k * (x_tgt - x) + l * f
    where
        x, dx           state and speed of the YuMi (either linear or angular)
        x_tgt, dx_tgt   target state and speed for the YuMi (either linear or angular)
        f               external forces at the end effector
    in either individual or coordinated motion.
    """
    def __init__(self, gains):
        super().__init__(gains)
        self._gains_right = np.array([gains["individual"]["right"]["direct_force"]["force"]]*3 + [gains["individual"]["right"]["direct_force"]["torque"]]*3)
        self._gains_left = np.array([gains["individual"]["left"]["direct_force"]["force"]]*3 + [gains["individual"]["left"]["direct_force"]["torque"]]*3)
        self._gains_abs = np.array([gains["coordinated"]["absolute"]["direct_force"]["force"]]*3 + [gains["coordinated"]["absolute"]["direct_force"]["torque"]]*3)
        self._gains_rel = np.array([gains["coordinated"]["relative"]["direct_force"]["force"]]*3 + [gains["coordinated"]["relative"]["direct_force"]["torque"]]*3)
        self.wrench_right = np.zeros((6,))
        self.wrench_left = np.zeros((6,))
        self.wrench_abs = np.zeros((6,))
        self.wrench_rel = np.zeros((6,))
    
    def update_current_state(self, yumi_state: YumiCoordinatedRobotState):
        super().update_current_state(yumi_state)
        self.wrench_right = yumi_state.pose_wrench_r
        self.wrench_left = yumi_state.pose_wrench_l
        self.wrench_abs = yumi_state.pose_wrench_abs
        self.wrench_rel = yumi_state.pose_wrench_rel
    
    def compute_individual_right_target_velocity(self):
        return super().compute_individual_right_target_velocity() + self._gains_right * self.wrench_right
    
    def compute_individual_left_target_velocity(self):
        return super().compute_individual_left_target_velocity() + self._gains_left * self.wrench_left
    
    def compute_coordinated_absolute_target_velocity(self):
        return super().compute_coordinated_absolute_target_velocity() + self._gains_abs * self.wrench_abs
    
    def compute_coordinated_relative_target_velocity(self):
        return super().compute_coordinated_relative_target_velocity() + self._gains_rel * self.wrench_rel


class YumiDualAdmittanceControlLaw(YumiDualCartesianVelocityControlLaw):
    """
    Generates velocity commands in cartesian space with the law
            dx := dx_tgt + k * (x_tgt - x)
    and
             x_tgt :=  x_des +  e
            dx_tgt := dx_des + de
            M * dde + D * de + K * e = f
    where
        x, dx           state and speed of the YuMi (either linear or angular)
        x_des, dx_des   desired (ex target) state and speed for the YuMi (either linear or angular)
        f               external forces at the end effector
        M, D, K         admittance coefficients
        e, de, dde      error (and derivatives) on the desired point due to external forces
    in either individual or coordinated motion.
    """
    def __init__(self, gains, discretization="forward"):
        super().__init__(gains)
        def weights(side: str, what: str):
            if side == "right":
                adm_f = gains["individual"]["right"]
                adm_t = gains["individual"]["right"]
            elif side == "left":
                adm_f = gains["individual"]["left"]
                adm_t = gains["individual"]["left"]
            elif side == "abs":
                adm_f = gains["coordinated"]["absolute"]
                adm_t = gains["coordinated"]["absolute"]
            else:
                adm_f = gains["coordinated"]["relative"]
                adm_t = gains["coordinated"]["relative"]
            weight_f = adm_f["admittance"]["force"][what]
            weight_t = adm_t["admittance"]["torque"][what]
            if weight_f is None or weight_t is None:
                return None
            return np.diag(weight_f).tolist() + np.diag(weight_t).tolist()
        
        self.admittance_right = AdmittanceWrench(weights("right", "m"), weights("right", "k"), weights("right", "d"), Parameters.dt, discretization)
        self.admittance_left = AdmittanceWrench(weights("left", "m"), weights("left", "k"), weights("left", "d"), Parameters.dt, discretization)
        self.admittance_abs = AdmittanceWrench(weights("abs", "m"), weights("abs", "k"), weights("abs", "d"), Parameters.dt, discretization)
        self.admittance_rel = AdmittanceWrench(weights("rel", "m"), weights("rel", "k"), weights("rel", "d"), Parameters.dt, discretization)
        self.wrench_right = np.zeros((6,))
        self.wrench_left = np.zeros((6,))
        self.wrench_abs = np.zeros((6,))
        self.wrench_rel = np.zeros((6,))
    
    def clear(self):
        super().clear()
        self.admittance_right.reset()
        self.admittance_left.reset()
        self.admittance_abs.reset()
        self.admittance_left.reset()
    
    def update_current_state(self, yumi_state: YumiCoordinatedRobotState):
        super().update_current_state(yumi_state)
        self.wrench_right = yumi_state.pose_wrench_r
        self.wrench_left = yumi_state.pose_wrench_l
        self.wrench_abs = yumi_state.pose_wrench_abs
        self.wrench_rel = yumi_state.pose_wrench_rel
    
    def update_desired_state(self, target_state: YumiCoordinatedRobotState):
        # here we inject the wrenches into the trajectory
        # "target" is renamed to "desired", so that "target" can now be repurposed 
        # include both the "desired" trajectory and the effect of external forces 
        # at the individual/coordinated frames (obtained by admittances)
        target_vel = np.zeros(12)
        # des_pos = [pR, pL] or [pA, pR]
        # des_rot = [oR, oL] or [oA, oR]
        # des_vel = [vR, wR, vL, wL] or [vA, wA, vR, wR]
        # wrenches = [fR, mR, fL, mL] or [fA, mA, fR, mR]
        
        # compensate for external wrenches
        (error_pos_r, error_rot_r), (error_vel_r, error_wel_r) = self.admittance_right.compute(self.wrench_right, self.dt)
        (error_pos_l, error_rot_l), (error_vel_l, error_wel_l) = self.admittance_left.compute(self.wrench_left, self.dt)
        (error_pos_abs, error_rot_abs), (error_vel_abs, error_wel_abs) = self.admittance_abs.compute(self.wrench_abs, self.dt)
        (error_pos_rel, error_rot_rel), (error_vel_rel, error_wel_rel) = self.admittance_rel.compute(self.wrench_rel, self.dt)
        
        if self.mode == "individual":
            error_pos_1, error_rot_1, error_vel_1, error_wel_1 = error_pos_r, error_rot_r, error_vel_r, error_wel_r
            error_pos_2, error_rot_2, error_vel_2, error_wel_2 = error_pos_l, error_rot_l, error_vel_l, error_wel_l
        else:
            error_pos_1, error_rot_1, error_vel_1, error_wel_1 = error_pos_abs, error_rot_abs, error_vel_abs, error_wel_abs
            error_pos_2, error_rot_2, error_vel_2, error_wel_2 = error_pos_rel, error_rot_rel, error_vel_rel, error_wel_rel
        
        #### FORCES ####
        target_pos_1 = target_state.pose_gripper_r.pos + error_pos_1
        target_pos_2 = target_state.pose_gripper_l.pos + error_pos_2
        
        target_vel[0] = target_state.pose_gripper_r.vel[0] + error_vel_1[0]
        target_vel[1] = target_state.pose_gripper_r.vel[1] + error_vel_1[1]
        target_vel[2] = target_state.pose_gripper_r.vel[2] + error_vel_1[2]
        target_vel[6] = target_state.pose_gripper_l.vel[0] + error_vel_2[0]
        target_vel[7] = target_state.pose_gripper_l.vel[1] + error_vel_2[1]
        target_vel[8] = target_state.pose_gripper_l.vel[2] + error_vel_2[2]
        
        #### TORQUES ####
        target_rot_1 = error_rot_1 * target_state.pose_gripper_r.rot
        target_rot_2 = error_rot_2 * target_state.pose_gripper_l.rot
        
        target_vel[3] = target_state.pose_gripper_r.vel[3] + error_wel_1[0]
        target_vel[4] = target_state.pose_gripper_r.vel[4] + error_wel_1[1]
        target_vel[5] = target_state.pose_gripper_r.vel[5] + error_wel_1[2]
        target_vel[9] = target_state.pose_gripper_l.vel[3] + error_wel_2[0]
        target_vel[10] = target_state.pose_gripper_l.vel[4] + error_wel_2[1]
        target_vel[11] = target_state.pose_gripper_l.vel[5] + error_wel_2[2]
        
        # recompose target (fill old object with new data)
        target_state.pose_gripper_r = Frame(target_pos_1, target_rot_1, target_vel[0:6])
        target_state.pose_gripper_l = Frame(target_pos_2, target_rot_2, target_vel[6:12])
        return super().update_desired_state(target_state)
