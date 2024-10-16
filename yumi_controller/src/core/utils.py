import numpy as np
import quaternion as quat

import rospy, tf

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, WrenchStamped
from abb_rapid_sm_addin_msgs.srv import SetSGCommand
from abb_robot_msgs.srv import TriggerWithResultCode

from yumi_controller.msg import Kinematics_msg

from dynamics import utils as utils_dyn
from core.trajectory import YumiParam


class YumiRobotState(utils_dyn.RobotState):
    """ State of the robot [right, left]
    """
    def __init__(
        self,
        joint_pos: np.ndarray = np.zeros(14),
        joint_vel: np.ndarray = np.zeros(14),
        joint_torque: np.ndarray = np.zeros(14),
        pose_pos: np.ndarray = np.zeros(6),
        pose_rot: np.ndarray = np.stack([quat.one, quat.one]),
        pose_vel: np.ndarray = np.zeros(12),
        pose_wrench: np.ndarray = np.zeros(12),
        jacobian: np.ndarray = np.zeros((12, 14)),
        grip_r: float = 0.,
        grip_l: float = 0.
    ):
        super().__init__(14)  # ignore all variables, instead wrap _right and _left using properties
        self._right = utils_dyn.RobotState(7, joint_pos[:7], joint_vel[:7], None, joint_torque[:7], 
                                          pose_pos[:3], pose_rot[0], pose_vel[:6], None, pose_wrench[:6], 
                                          jacobian[0:6, 0:7])
        self._left = utils_dyn.RobotState(7, joint_pos[7:], joint_vel[7:], None, joint_torque[7:], 
                                         pose_pos[3:], pose_rot[1], pose_vel[6:], None, pose_wrench[6:], 
                                         jacobian[6:12, 7:14])
        # gripping
        self.grip_r = grip_r
        self.grip_l = grip_l

    ####### RIGHT #######
    @property
    def joint_pos_r(self):
        return self._right.joint_pos
    
    @property
    def joint_vel_r(self):
        return self._right.joint_vel
    
    @property
    def joint_acc_r(self):
        return self._right.joint_acc
    
    @property
    def joint_torque_r(self):
        return self._right.joint_torque

    @property
    def pose_pos_r(self):
        return self._right.pose_pos

    @property
    def pose_rot_r(self):
        return self._right.pose_rot
    
    @property
    def pose_vel_r(self):
        return self._right.pose_vel
    
    @property
    def pose_vel_lin_r(self):
        return self._right.pose_vel_lin

    @property
    def pose_vel_ang_r(self):
        return self._right.pose_vel_ang
    
    @property
    def pose_acc_r(self):
        return self._right.pose_acc
    
    @property
    def pose_acc_lin_r(self):
        return self._right.pose_acc_lin

    @property
    def pose_acc_ang_r(self):
        return self._right.pose_acc_ang
    
    @property
    def pose_wrench_r(self):
        return self._right.pose_wrench
    
    @property
    def pose_force_r(self):
        return self._right.pose_force
    
    @property
    def pose_torque_r(self):
        return self._right.pose_torque
    
    @property
    def jacobian_r(self):
        return self._right.jacobian
    
    ####### LEFT #######
    @property
    def joint_pos_l(self):
        return self._left.joint_pos
    
    @property
    def joint_vel_l(self):
        return self._left.joint_vel
    
    @property
    def joint_acc_l(self):
        return self._left.joint_acc
    
    @property
    def joint_torque_l(self):
        return self._left.joint_torque

    @property
    def pose_pos_l(self):
        return self._left.pose_pos

    @property
    def pose_rot_l(self):
        return self._left.pose_rot
    
    @property
    def pose_vel_l(self):
        return self._left.pose_vel
    
    @property
    def pose_vel_lin_l(self):
        return self._left.pose_vel_lin

    @property
    def pose_vel_ang_l(self):
        return self._left.pose_vel_ang
    
    @property
    def pose_acc_l(self):
        return self._left.pose_acc
    
    @property
    def pose_acc_lin_l(self):
        return self._left.pose_acc_lin

    @property
    def pose_acc_ang_l(self):
        return self._left.pose_acc_ang
    
    @property
    def pose_wrench_l(self):
        return self._left.pose_wrench
    
    @property
    def pose_force_l(self):
        return self._left.pose_force
    
    @property
    def pose_torque_l(self):
        return self._left.pose_torque
    
    @property
    def jacobian_l(self):
        return self._left.jacobian

    ####### OVERALL #######
    @property
    def joint_pos(self):
        return np.concatenate([self.joint_pos_r, self.joint_pos_l])
    
    # TODO refactor this
    @joint_pos.setter
    def joint_pos(self, value):
        self._right._joint_pos = value[:7]
        self._left._joint_pos = value[7:]
    
    @property
    def joint_vel(self):
        return np.concatenate([self.joint_vel_r, self.joint_vel_l])
    
    # TODO refactor this
    @joint_vel.setter
    def joint_vel(self, value):
        self._right._joint_vel = value[:7]
        self._left._joint_vel = value[7:]
        
    @property
    def joint_acc(self):
        return np.concatenate([self.joint_acc_r, self.joint_acc_l])
    
    @property
    def joint_torque(self):
        return np.concatenate([self.joint_torque_r, self.joint_torque_l])

    # TODO refactor this
    @joint_torque.setter
    def joint_torque(self, value):
        self._right._joint_torque = value[:7]
        self._left._joint_torque = value[7:]

    @property
    def pose_pos(self):
        return np.concatenate([self.pose_pos_r, self.pose_pos_l])

    @property
    def pose_rot(self):
        return np.stack([self.pose_rot_r, self.pose_rot_l])
    
    @property
    def pose_vel(self):
        return np.concatenate([self.pose_vel_r, self.pose_vel_l])
        
    @property
    def pose_vel_lin(self):
        return np.concatenate([self.pose_vel_lin_r, self.pose_vel_lin_l])

    @property
    def pose_vel_ang(self):
        return np.concatenate([self.pose_vel_ang_r, self.pose_vel_ang_l])
    
    @property
    def pose_acc(self):
        return np.concatenate([self.pose_acc_r, self.pose_acc_l])
    
    @property
    def pose_acc_lin(self):
        return np.concatenate([self.pose_acc_lin_r, self.pose_acc_lin_l])

    @property
    def pose_acc_ang(self):
        return np.concatenate([self.pose_acc_ang_r, self.pose_acc_ang_l])
    
    @property
    def pose_wrench(self):
        return np.concatenate([self.pose_wrench_r, self.pose_wrench_l])
    
    # TODO refactor this
    @pose_wrench.setter
    def pose_wrench(self, value):
        self._right._pose_wrench = value[:6]
        self._left._pose_wrench = value[6:]
    
    @property
    def pose_force(self):
        return np.concatenate([self.pose_force_r, self.pose_force_l])
    
    @property
    def pose_torque(self):
        return np.concatenate([self.pose_torque_r, self.pose_torque_l])
    
    @property
    def jacobian(self):
        return utils_dyn.jacobian_combine(self.jacobian_r, self.jacobian_l)
    
    # TODO refactor this
    @jacobian.setter
    def jacobian(self, value):
        self._right._jacobian = value[:6, :7]
        self._left._jacobian = value[6:, 7:]


class YumiCoordinatedRobotState(YumiRobotState):
    """ State of the robot [right, left].
        Symmetry [-1,+1] indicates the balance between right (+1) and left (-1).
    """
    def __init__(
        self,
        joint_pos: np.ndarray = np.zeros(14),
        joint_vel: np.ndarray = np.zeros(14),
        joint_torque: np.ndarray = np.zeros(14),
        pose_pos: np.ndarray = np.zeros(6),
        pose_rot: np.ndarray = np.stack([quat.one, quat.one]),
        pose_vel: np.ndarray = np.zeros(12),
        pose_wrench: np.ndarray = np.zeros(12),
        jacobian: np.ndarray = np.zeros((12, 14)),
        grip_r: float = 0.,
        grip_l: float = 0.,
        symmetry: float = 0.
    ):
        super().__init__(joint_pos, joint_vel, joint_torque, pose_pos, pose_rot, pose_vel, pose_wrench, jacobian, grip_r, grip_l)
        # grippers
        self.pose_gripper_r = utils_dyn.Frame()
        self.pose_gripper_l = utils_dyn.Frame()
        self.jacobian_grippers = np.zeros((12, 14))
        # elbows
        self.pose_elbow_r = utils_dyn.Frame()
        self.pose_elbow_l = utils_dyn.Frame()
        self.jacobian_elbows = np.zeros((12, 8))
        # coordinated
        assert -1 < symmetry and symmetry < +1, "Symmetry value must be in range [-1,+1]"
        self.alpha = (symmetry + 1) * 0.5
        self.pose_abs = utils_dyn.Frame()
        self.pose_rel = utils_dyn.Frame()
        self.pose_wrench_abs = np.zeros(6)
        self.pose_wrench_rel = np.zeros(6)
        self.jacobian_coordinated = np.zeros((12, 14))
    
    @property
    def poses_individual(self):
        return self.pose_gripper_r, self.pose_gripper_l
    
    @property
    def poses_coordinated(self):
        return self.pose_abs, self.pose_rel
        
    @property
    def jacobian_gripper_r(self):
        return self.jacobian_grippers[:6, :7]

    @property
    def jacobian_gripper_l(self):
        return self.jacobian_grippers[6:, 7:]
    
    @property
    def jacobian_elbow_r(self):
        return self.jacobian_elbows[:6, :4]

    @property
    def jacobian_elbow_l(self):
        return self.jacobian_elbows[6:, 4:]
    
    @property
    def jacobian_coordinated_abs(self):
        return self.jacobian_coordinated[:6, :]

    @property
    def jacobian_coordinated_rel(self):
        return self.jacobian_coordinated[6:, :]


# TODO polish this abomination
def YumiParam_to_YumiCoordinatedRobotState(yumi_param: YumiParam):
    yumi_state = YumiCoordinatedRobotState(
        grip_r=yumi_param.grip_right, 
        grip_l=yumi_param.grip_left)
    yumi_state.pose_gripper_r = utils_dyn.Frame(
        yumi_param.position[:3],
        yumi_param.rotation[0],
        yumi_param.velocity[:6])
    yumi_state.pose_gripper_l = utils_dyn.Frame(
        yumi_param.position[3:],
        yumi_param.rotation[1],
        yumi_param.velocity[6:])
    return yumi_state


# TODO maybe create a YumiWrenchStateUpdater?
class YumiDualStateUpdater(YumiCoordinatedRobotState):

    def __init__(
        self,
        arm_to_gripper_r: utils_dyn.Frame,
        arm_to_gripper_l: utils_dyn.Frame,
        symmetry: float = 0.
    ) -> None:
        super().__init__(symmetry=symmetry)
        self.arm_to_gripper_r = arm_to_gripper_r
        self.arm_to_gripper_l = arm_to_gripper_l
        
        # read force sensors
        self._wrenches = np.zeros(12)  # [fR, mR, fL, mL]
        rospy.Subscriber("/ftsensor_r/world_tip", WrenchStamped, self._callback_ext_force, callback_args="right", tcp_nodelay=True, queue_size=3)
        rospy.Subscriber("/ftsensor_l/world_tip", WrenchStamped, self._callback_ext_force, callback_args="left", tcp_nodelay=True, queue_size=3)
        
        # TODO need a mutex here for data access
        rospy.Subscriber("/jacobian_R_L", Kinematics_msg, self._callback, queue_size=3, tcp_nodelay=True)
        rospy.wait_for_message("/jacobian_R_L", Kinematics_msg)
    
    @staticmethod
    def _WrenchStampedMsg_to_ndarray(wrench_stamped: WrenchStamped):
        fx = wrench_stamped.wrench.force.x
        fy = wrench_stamped.wrench.force.y
        fz = wrench_stamped.wrench.force.z
        mx = wrench_stamped.wrench.torque.x
        my = wrench_stamped.wrench.torque.y
        mz = wrench_stamped.wrench.torque.z
        return np.array([fx, fy, fz, mx, my, mz])
    
    @staticmethod
    def _PoseMsg_to_frame(pose_msg: Pose):
        return utils_dyn.Frame(
            position=np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]),
            quaternion=np.quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z))
        
    def _callback_ext_force(self, data, arm):
        if arm == "right":
            self._wrenches[0:6] = self._WrenchStampedMsg_to_ndarray(data)
        elif arm == "left":
            self._wrenches[6:12] = self._WrenchStampedMsg_to_ndarray(data)
    
    def _callback(self, data: Kinematics_msg) -> None:
        """ Updates forward kinematics using KDL instead of TF tree
        """
        #############################      individual motion      #############################
        # update joint position, velocity ... 
        self.joint_pos = np.asarray(data.jointPosition)[:14]  # simulation adds gripping position
        self.joint_vel = np.asarray(data.jointVelocity)
        # ... and jacobian
        jacobians_arms = np.asarray(data.jacobian[1].data).reshape((6,7,2))
        self.jacobian = utils_dyn.jacobian_combine(jacobians_arms[:,:,0], jacobians_arms[:,:,1])
        
        # update gripper poses ...
        pose_arm_r = self._PoseMsg_to_frame(data.forwardKinematics[2])
        pose_arm_l = self._PoseMsg_to_frame(data.forwardKinematics[3])
        self.pose_gripper_r = pose_arm_r @ self.arm_to_gripper_r  # T_BG = T_BA * T_AG
        self.pose_gripper_l = pose_arm_l @ self.arm_to_gripper_l
        
        # TODO
        # new_pose_gripper_r = self._PoseMsg_to_frame(data.forwardKinematics[0])
        # new_pose_gripper_l = self._PoseMsg_to_frame(data.forwardKinematics[1])
        # print(self.pose_gripper_r)
        # print(new_pose_gripper_r)
        # print(self.pose_gripper_l - new_pose_gripper_l)
        
        # ... and jacobian (from base frame to tip of gripper, for both arms) ...
        # dist_vec_r = self.pose_gripper_r.pos - pose_arm_r.pos
        # dist_vec_l = self.pose_gripper_l.pos - pose_arm_l.pos
        # jacobian_gripper_r = utils_dyn.jacobian_change_end_frame(dist_vec_r, self.jacobian_r)
        # jacobian_gripper_l = utils_dyn.jacobian_change_end_frame(dist_vec_l, self.jacobian_l)
        # self.jacobian_grippers = utils_dyn.jacobian_combine(jacobian_gripper_r, jacobian_gripper_l)
        jacobian_grippers = np.asarray(data.jacobian[0].data).reshape((6,7,2))
        jacobian_gripper_r = jacobian_grippers[:,:,0]
        jacobian_gripper_l = jacobian_grippers[:,:,1]
        self.jacobian_grippers = utils_dyn.jacobian_combine(jacobian_gripper_r, jacobian_gripper_l)
        
        # ... and velocity (now that the jacobian has been updated)
        pose_grippers_vel = self.jacobian_grippers @ self.joint_vel
        self.pose_gripper_r.vel = pose_grippers_vel[:6]
        self.pose_gripper_l.vel = pose_grippers_vel[6:]
        
        # Q: couldn't we just have used  self.jacobian_grippers  immediately for velocity update,
        #    just like with elbows below? 
        # A: nope, because  self.jacobian_grippers  depends self.pose_gripper_*, which must be 
        #    calculated first (to avoid using the old one), so velocity update must be deferred
        
        # update elbow jacobian ... 
        jacobians_elbows = np.asarray(data.jacobian[2].data).reshape((6,4,2))
        self.jacobian_elbows = utils_dyn.jacobian_combine(jacobians_elbows[:,:,0], jacobians_elbows[:,:,1])
        # ... and pose ...
        self.pose_elbow_r = self._PoseMsg_to_frame(data.forwardKinematics[4])
        self.pose_elbow_l = self._PoseMsg_to_frame(data.forwardKinematics[5])
        # ... and velocity
        pose_elbow_vel = self.jacobian_elbows @ self.joint_vel[[0,1,2,3,7,8,9,10]]
        self.pose_elbow_r.vel = pose_elbow_vel[:6]
        self.pose_elbow_l.vel =  pose_elbow_vel[6:]
         
        # force
        self.pose_wrench = self._wrenches
        self.joint_torque = self.jacobian_grippers.T @ self.pose_wrench
        #######################################################################################
        
        #############################      coordinated motion     #############################
        # absolute pose, avg of the grippers
        rot_diff = utils_dyn.quat_min_diff(self.pose_gripper_l.rot, self.pose_gripper_r.rot) * self.pose_gripper_l.rot.conjugate()
        pos_abs = (1-self.alpha)*self.pose_gripper_r.pos + self.alpha*self.pose_gripper_l.pos
        rot_abs = quat.from_rotation_vector((1-self.alpha) * quat.as_rotation_vector(rot_diff)) * self.pose_gripper_l.rot
        self.pose_abs = utils_dyn.Frame(pos_abs, rot_abs)
        
        # relative pose, difference of the grippers wrt absolute frame
        coeff_r = self.alpha / ((1-self.alpha)**2 + self.alpha**2)
        coeff_l = (1-self.alpha) / ((1-self.alpha)**2 + self.alpha**2)
        pose_abs_inv = self.pose_abs.inv()
        pose_r_wrt_abs = pose_abs_inv @ self.pose_gripper_r
        pose_l_wrt_abs = pose_abs_inv @ self.pose_gripper_l
        # ATTENTION here we CANNOT simplify 
        #               pose_abs_inv @ self.pose_gripper_r - pose_abs_inv @ self.pose_gripper_l
        #           to  
        #               pose_abs_inv @ (self.pose_gripper_r - self.pose_gripper_l)
        #           due to non-commutation of rotation in the "pose subtraction" operation
        pos_rel = coeff_r * pose_r_wrt_abs.pos - coeff_l * pose_l_wrt_abs.pos
        rot_r_rel = quat.from_rotation_vector(coeff_r * quat.as_rotation_vector(pose_r_wrt_abs.rot))
        rot_l_rel = quat.from_rotation_vector(coeff_l * quat.as_rotation_vector(pose_l_wrt_abs.rot))
        rot_rel = utils_dyn.quat_min_diff(rot_l_rel, rot_r_rel) * rot_l_rel.conjugate()
        self.pose_rel = utils_dyn.Frame(pos_rel, rot_rel)
        
        # relative linking matrix: maps gripper velocities to the velocity difference wrt absolute frame
        base_ee_to_abs_rel_trans = utils_dyn.jacobian_change_frames( 0.5 * (self.pose_gripper_r.pos - self.pose_gripper_l.pos), pose_abs_inv.rot )
        link_mat_rel = base_ee_to_abs_rel_trans @ np.block([ coeff_r*np.eye(6), -coeff_l*np.eye(6) ])
        
        # absolute linking matrix: maps gripper velocities to the velocity average
        link_mat_abs = np.block([ (1-self.alpha)*np.eye(6), self.alpha*np.eye(6) ])
        
        # coordinated jacobian
        link_mat = np.block([[ link_mat_abs ],
                             [ link_mat_rel ]])
        
        self.jacobian_coordinated = link_mat @ self.jacobian_grippers
        
        # set velocities
        pose_coordinated_vel = link_mat @ pose_grippers_vel
        self.pose_abs.vel = pose_coordinated_vel[:6]
        self.pose_rel.vel = pose_coordinated_vel[6:]
        
        # update wrenches
        # (using the kineto-statics duality,  self.pose_wrench = link_mat.T @ wrench_coordinated )
        wrench_coordinated = np.linalg.pinv(link_mat.T) @ self.pose_wrench
        self.pose_wrench_abs = wrench_coordinated[:6]
        self.pose_wrench_rel = wrench_coordinated[6:]
        #######################################################################################
        

class YumiVelocityCommand(object):
    """ Used for storing the velocity command for yumi
    """
    def __init__(self):
        self._prev_joint_vel = np.zeros(14)
        self._pub = rospy.Publisher("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArray, queue_size=1, tcp_nodelay=True)

    def send_velocity_cmd(self, joint_velocity: np.ndarray):
        """ Velocity should be an np.array() with 14 elements, [right arm, left arm]
        """
        # flip the arry to [left, right]
        self._prev_joint_vel = joint_velocity
        joint_velocity = np.hstack([joint_velocity[7:14], joint_velocity[0:7]]).tolist()
        msg = Float64MultiArray()
        msg.data = joint_velocity
        self._pub.publish(msg)


class YumiGrippersCommand(object):
    """ Class for controlling the grippers on YuMi, the grippers are controlled
        in [mm] and uses ros service
    """
    def __init__(self):
        # rosservice, for control over grippers
        self._service_SetSGCommand = rospy.ServiceProxy("/yumi/rws/sm_addin/set_sg_command", SetSGCommand)
        self._service_RunSGRoutine = rospy.ServiceProxy("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCode)
        self._prev_gripper_r = 0
        self._prev_gripper_l = 0

    def send_position_cmd(self, gripper_r=None, gripper_l=None):
        """ Set new gripping position
            :param gripperRight: float [mm]
            :param gripperLeft: float [mm]
        """
        tol = 1e-5
        try:
            # stacks/set the commands for the grippers 
            # do not send the same command twice as grippers will momentarily regrip

            # for right gripper
            if gripper_r is not None:
                if abs(self._prev_gripper_r - gripper_r) >= tol:
                    if gripper_r <= 0.1:
                        self._service_SetSGCommand.call(task="T_ROB_R", command=6)
                    else:
                        self._service_SetSGCommand.call(task="T_ROB_R", command=5, target_position=gripper_r)
                    self._prev_gripper_r = gripper_r

            # for left gripper
            if gripper_l is not None:
                if abs(self._prev_gripper_l - gripper_l) >= tol:
                    if gripper_l <= 0.1: # if gripper set close to zero then grip in 
                        self._service_SetSGCommand.call(task="T_ROB_L", command=6)
                    else: # otherwise move to position 
                        self._service_SetSGCommand.call(task="T_ROB_L", command=5, target_position=gripper_l)
                    self._prev_gripper_l = gripper_l

            # sends of the commands to the robot
            self._service_RunSGRoutine.call()

        except Exception as ex:
            print(f"SmartGripper error : {ex}")


class TfBroadcastControllerFrames(object):
    """ Class for adding new frames to the tf tree and used internally for control.
    """
    def __init__(self, arm_to_gripper_right : utils_dyn.Frame, arm_to_gripper_left : utils_dyn.Frame, yumi_to_world : utils_dyn.Frame):
        """ Set up the frames to broadcast.
            :param arm_to_gripper_right: class instance of FramePose describing the local transformation from yumi_link_7_r to gripper_r_tip
            :param arm_to_gripper_left: class instance of FramePose describing the local transformation from yumi_link_7_l to gripper_l_tip
            :param yumi_to_world: class instance of FramePose describing the local transformation from yumi_base_link to world
        """
        self._broadcaster = tf.TransformBroadcaster()
        self._arm_to_gripper_r = arm_to_gripper_right
        self._arm_to_gripper_l = arm_to_gripper_left
        self._yumi_to_world = yumi_to_world

    @property
    def arm_to_gripper_r(self):
        """ Returns the frame
        """
        return self._arm_to_gripper_r

    @arm_to_gripper_r.setter
    def arm_to_gripper_r(self, gripperRight):
        """ Update the frame
            :param gripperRight: class instance of FramePose describing the local transformation from yumi_link_7_r to gripper_r_tip
        """
        self._arm_to_gripper_r = gripperRight

    @property
    def arm_to_gripper_l(self):
        """ Returns the frame
        """
        return self._arm_to_gripper_l

    @arm_to_gripper_l.setter
    def arm_to_gripper_l(self, gripperLeft):
        """ Update the frame
            :param gripperLeft: class instance of FramePose describing the local transformation from yumi_link_7_l to gripper_l_tip
        """
        self._arm_to_gripper_l = gripperLeft

    @property
    def yumi_to_world(self):
        """ Returns the frame
        """
        return self._yumi_to_world

    @yumi_to_world.setter
    def yumi_to_world(self, yumiToWorld):
        """ Update the frame
            :param yumiToWorld: class instance of FramePose describing the local transformation from yumi_base_link to world
        """
        self._yumi_to_world = yumiToWorld


    def broadcast(self):
        """ Sends out to the tf tree
        """
        self._broadcaster.sendTransform(
            tuple(self._yumi_to_world.pos),
            tuple(np.roll(quat.as_float_array(self._yumi_to_world.rot), -1)),
            rospy.Time.now(), "world", "yumi_base_link")
        self._broadcaster.sendTransform(
            tuple(self._arm_to_gripper_r.pos),
            tuple(np.roll(quat.as_float_array(self._arm_to_gripper_r.rot), -1)),
            rospy.Time.now(), "gripper_r_tip", "yumi_link_7_r")
        self._broadcaster.sendTransform(
            tuple(self._arm_to_gripper_l.pos),
            tuple(np.roll(quat.as_float_array(self._arm_to_gripper_l.rot), -1)),
            rospy.Time.now(), "gripper_l_tip", "yumi_link_7_l")
