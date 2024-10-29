import numpy as np
import quaternion as quat

from .parameters import Parameters
from .robot_state import YumiCoordinatedRobotState

from dynamics.control_laws import ControlLawError, CartesianVelocityControlLaw
from dynamics.filters import AdmittanceForce, AdmittanceTorque
from dynamics.utils import Frame


# TODO make this a CartesianVelocityControlLaw
class YumiDualCartesianVelocityControlLaw(object):
    """
    Generates velocity commands in cartesian space with the law
            dx := dx_tgt + k * (x_tgt - x)
    where
        x, dx           state and speed of the YuMi (either linear or angular)
        x_tgt, dx_tgt   target state and speed for the YuMi (either linear or angular)
    in either individual or coordinated motion.
    """
    def __init__(self, gains):
        self.mode = None  # can be either "individual" or "coordinated"
        self.control_right = CartesianVelocityControlLaw(gains["individual"]["right"]["position"], gains["individual"]["right"]["rotation"], gains["individual"]["right"]["max_deviation"])
        self.control_left  = CartesianVelocityControlLaw(gains["individual"]["left"]["position"], gains["individual"]["left"]["rotation"], gains["individual"]["left"]["max_deviation"])
        self.control_abs = CartesianVelocityControlLaw(gains["coordinated"]["absolute"]["position"], gains["coordinated"]["absolute"]["rotation"], gains["coordinated"]["absolute"]["max_deviation"])
        self.control_rel = CartesianVelocityControlLaw(gains["coordinated"]["relative"]["position"], gains["coordinated"]["relative"]["rotation"], gains["coordinated"]["relative"]["max_deviation"])
        self.grip_r = None
        self.grip_l = None
        self.dt = Parameters.dt
        
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
        self.grip_r = None
        self.grip_l = None

    def update_current_dt(self, dt: float):
        self.dt = dt

    def update_current_pose(self, yumi_state: YumiCoordinatedRobotState):
        """ 
        Updates individual and coordinated poses
        """
        self.control_right.update_current_pose(yumi_state.pose_gripper_r)
        self.control_left.update_current_pose(yumi_state.pose_gripper_l)
        self.control_abs.update_current_pose(yumi_state.pose_abs)
        self.control_rel.update_current_pose(yumi_state.pose_rel)
    
    def update_target_pose(self, target_state: YumiCoordinatedRobotState):
        """ Updates the desired velocities and target position. 
            ATTENTION: this function uses `pose_gripper_r` and `pose_gripper_l` as
            desired target for both the individual (right and left) and the coordinated 
            (absolute and relative) modes. This behaviour is desired since in this way 
            we avoid calculating the coordinated poses from the individuals ones and 
            vice versa, which is unnecessary since this is a velocity controller.
            This means that `target_state` will be used as individual or coordinated 
            based on the current value of `self.mode`.
        """
        self.control_right.update_desired_pose(target_state.pose_gripper_r)
        self.control_left.update_desired_pose(target_state.pose_gripper_l)
        self.control_abs.update_desired_pose(target_state.pose_gripper_r)
        self.control_rel.update_desired_pose(target_state.pose_gripper_l)
        self.grip_r, self.grip_l = target_state.grip_r, target_state.grip_l

    def compute_individual_right_target_velocity(self):
        """ Calculates the target velocities for individual right arm control.
        """
        try:
            self.control_right.compute_target_velocity()
        except ControlLawError as ex:
            # turn off deviation error if gripper collision constraint is active for individual mode
            if not Parameters.feasibility_objectives["gripper_collision"]:
                raise ex
        return self.control_right.target_velocity

    def compute_individual_left_target_velocity(self):
        """ Calculates the target velocities for individual left arm control.
        """
        try:
            self.control_left.compute_target_velocity()
        except Exception as ex:
            if not Parameters.feasibility_objectives["gripper_collision"]:
                raise ex
        return self.control_left.target_velocity
    
    def compute_coordinated_absolute_target_velocity(self):
        """ Calculates the target velocities for absolute motion i.e. controlling
            the average of the grippers.
        """
        return self.control_abs.compute_target_velocity()

    def compute_coordinated_relative_target_velocity(self):
        """ Calculates the target velocities for relative motion i.e. controlling
            the grippers relative to each other in absolute frame.
        """
        return self.control_rel.compute_target_velocity()

    def compute_target_velocity(self):
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
        self._gains_right = np.array([gains["individual"]["right"]["force"]]*3 + [gains["individual"]["right"]["torque"]]*3)
        self._gains_left = np.array([gains["individual"]["left"]["force"]]*3 + [gains["individual"]["left"]["torque"]]*3)
        self._gains_abs = np.array([gains["coordinated"]["absolute"]["force"]]*3 + [gains["coordinated"]["absolute"]["torque"]]*3)
        self._gains_rel = np.array([gains["coordinated"]["relative"]["force"]]*3 + [gains["coordinated"]["relative"]["torque"]]*3)
        self.wrench_right = np.zeros((6,))
        self.wrench_left = np.zeros((6,))
        self.wrench_abs = np.zeros((6,))
        self.wrench_rel = np.zeros((6,))
    
    def update_current_pose(self, yumi_state: YumiCoordinatedRobotState):
        super().update_current_pose(yumi_state)
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
        adm_f_r = gains["individual"]["right"]["admittance"]["force"]
        adm_t_r = gains["individual"]["right"]["admittance"]["torque"]
        adm_f_l = gains["individual"]["left"]["admittance"]["force"]
        adm_t_l = gains["individual"]["left"]["admittance"]["torque"]
        adm_f_abs = gains["coordinated"]["absolute"]["admittance"]["force"]
        adm_t_abs = gains["coordinated"]["absolute"]["admittance"]["torque"]
        adm_f_rel = gains["coordinated"]["relative"]["admittance"]["force"]
        adm_t_rel = gains["coordinated"]["relative"]["admittance"]["torque"]
        self.admittance_force_r = AdmittanceForce(adm_f_r["m"], adm_f_r["k"], adm_f_r["d"], Parameters.dt, discretization)
        self.admittance_torque_r = AdmittanceTorque(adm_t_r["m"], adm_t_r["k"], adm_t_r["d"], Parameters.dt, discretization)
        self.admittance_force_l = AdmittanceForce(adm_f_l["m"], adm_f_l["k"], adm_f_l["d"], Parameters.dt, discretization)
        self.admittance_torque_l = AdmittanceTorque(adm_t_l["m"], adm_t_l["k"], adm_t_l["d"], Parameters.dt, discretization)
        self.admittance_force_abs = AdmittanceForce(adm_f_abs["m"], adm_f_abs["k"], adm_f_abs["d"], Parameters.dt, discretization)
        self.admittance_torque_abs = AdmittanceTorque(adm_t_abs["m"], adm_t_abs["k"], adm_t_abs["d"], Parameters.dt, discretization)
        self.admittance_force_rel = AdmittanceForce(adm_f_rel["m"], adm_f_rel["k"], adm_f_rel["d"], Parameters.dt, discretization)
        self.admittance_torque_rel = AdmittanceTorque(adm_t_rel["m"], adm_t_rel["k"], adm_t_rel["d"], Parameters.dt, discretization)
        self.wrench_right = np.zeros((6,))
        self.wrench_left = np.zeros((6,))
        self.wrench_abs = np.zeros((6,))
        self.wrench_rel = np.zeros((6,))
    
    def clear(self):
        super().clear()
        self.admittance_force_r.clear()
        self.admittance_torque_r.clear()
        self.admittance_force_l.clear()
        self.admittance_torque_l.clear()
        self.admittance_force_abs.clear()
        self.admittance_torque_abs.clear()
        self.admittance_force_rel.clear()
        self.admittance_torque_rel.clear()
    
    def update_current_pose(self, yumi_state: YumiCoordinatedRobotState):
        super().update_current_pose(yumi_state)
        self.wrench_right = yumi_state.pose_wrench_r
        self.wrench_left = yumi_state.pose_wrench_l
        self.wrench_abs = yumi_state.pose_wrench_abs
        self.wrench_rel = yumi_state.pose_wrench_rel
    
    def update_target_pose(self, target_state: YumiCoordinatedRobotState):
        # here we inject the wrenches into the trajectory
        # "target" is renamed to "desired", so that "target" can now be repurposed 
        # include both the "desired" trajectory and the effect of external forces 
        # at the individual/coordinated frames (obtained by admittances)
        des_pos = np.concatenate([target_state.pose_gripper_r.pos, target_state.pose_gripper_l.pos])
        des_rot = np.stack([target_state.pose_gripper_r.rot, target_state.pose_gripper_l.rot])
        des_vel = np.concatenate([target_state.pose_gripper_r.vel, target_state.pose_gripper_l.vel])
        des_grip_r = target_state.grip_r
        des_grip_l = target_state.grip_l
        # des_pos = [pR, pL] or [pA, pR]
        # des_rot = [oR, oL] or [oA, oR]
        # des_vel = [vR, wR, vL, wL] or [vA, wA, vR, wR]
        # wrenches = [fR, mR, fL, mL] or [fA, mA, fR, mR]
        target_pos = des_pos.copy()
        target_rot = des_rot.copy()
        target_vel = des_vel.copy()
        
        # compensate for external wrenches
        if self.mode == "individual":
            forces_1, forces_2 = self.wrench_right[0:3], self.wrench_left[0:3]
            torques_1, torques_2 = self.wrench_right[3:6], self.wrench_left[3:6]
            adm_force_1 = self.admittance_force_r
            adm_force_2 = self.admittance_force_l
            adm_torque_1 = self.admittance_torque_r
            adm_torque_2 = self.admittance_torque_l
        else:
            forces_1, forces_2 = self.wrench_abs[0:3], self.wrench_rel[0:3]
            torques_1, torques_2 = self.wrench_abs[3:6], self.wrench_rel[3:6]
            adm_force_1 = self.admittance_force_abs
            adm_force_2 = self.admittance_force_rel
            adm_torque_1 = self.admittance_torque_abs
            adm_torque_2 = self.admittance_torque_rel
        
        #### FORCES ####
        error_pos_1, error_vel_1 = adm_force_1.compute(forces_1, self.dt)
        error_pos_2, error_vel_2 = adm_force_2.compute(forces_2, self.dt)
        
        error_pos = np.concatenate([error_pos_1, error_pos_2])
        target_pos = des_pos + error_pos
        
        error_vel = np.concatenate([error_vel_1, error_vel_2])
        target_vel[[0,1,2,6,7,8]] = des_vel[[0,1,2,6,7,8]] + error_vel
        
        #### TORQUES ####
        error_rot_1, error_wel_1 = adm_torque_1.compute(torques_1, self.dt)
        error_rot_2, error_wel_2 = adm_torque_2.compute(torques_2, self.dt)
        
        target_rot_r = quat.from_float_array(error_rot_1) * des_rot[0]
        target_rot_l = quat.from_float_array(error_rot_2) * des_rot[1]
        target_rot = np.stack([target_rot_r, target_rot_l])
        
        error_wel = np.concatenate([error_wel_1, error_wel_2])
        target_vel[[3,4,5,9,10,11]] = des_vel[[3,4,5,9,10,11]] + error_wel
        
        # recompose target
        target_state = YumiCoordinatedRobotState(
            grip_r=des_grip_r,
            grip_l=des_grip_l)
        target_state.pose_gripper_r = Frame(target_pos[0:3], target_rot[0], target_vel[0:6])
        target_state.pose_gripper_l = Frame(target_pos[3:6], target_rot[1], target_vel[6:12])
        return super().update_target_pose(target_state)
