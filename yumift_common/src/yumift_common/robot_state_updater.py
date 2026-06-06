#!/usr/bin/env python3
import rospy
import numpy as np

from geometry_msgs.msg import WrenchStamped, Pose, Twist, Wrench
from sensor_msgs.msg import JointState
from yumift_msgs.msg import YumiKinematics, RobotState, Jacobian

from yumift_common.robot_state import YumiCoordinatedRobotState
import yumift_common.msg_utils as msg_utils

from dynamicals.utils import Frame, jacobian_change_frames, jacobian_combine, quat_diff


RIGHT        = 0b00000001
LEFT         = 0b00000010
INDIVIDUAL   = RIGHT | LEFT
ABSOLUTE     = 0b00000100
RELATIVE     = 0b00001000
COORDINATED  = ABSOLUTE | RELATIVE
ELBOW_RIGHT  = 0b00010000
ELBOW_LEFT   = 0b00100000
ELBOWS       = ELBOW_RIGHT | ELBOW_LEFT
ARM_RIGHT    = 0b01000000
ARM_LEFT     = 0b10000000
ARMS         = ARM_RIGHT | ARM_LEFT
EVERYTHING   = INDIVIDUAL | COORDINATED | ELBOWS | ARMS


class YumiStateUpdater(object):

    def __init__(
        self,
        state : YumiCoordinatedRobotState,
        options : int
    ) -> None:
        super().__init__()
        
        # values to be updated by reference
        self.state = state
        
        # coordinated jacobian
        self.link_mat = np.zeros((12,12))
        self._cache_eye = np.eye(6)
        
        # RobotState publisher
        self._robot_state_publisher = rospy.Publisher("/robot_state_coordinated", RobotState, queue_size=1)
        self.options = options
        if self.options & COORDINATED:
            self.options |= INDIVIDUAL
        
        # read force sensors
        self._wrenches = np.zeros(12)  # [fR, mR, fL, mL]
        rospy.Subscriber("/wrench_right", WrenchStamped, self._callback_ext_force, callback_args="right", queue_size=1)
        rospy.Subscriber("/wrench_left", WrenchStamped, self._callback_ext_force, callback_args="left", queue_size=1)
        
        rospy.Subscriber("/kinematics", YumiKinematics, self._callback_yumi_kinematics, queue_size=1)
        rospy.wait_for_message("/kinematics", YumiKinematics)
        
        
    def _callback_ext_force(self, data: WrenchStamped, arm: str):
        if arm == "right":
            self._wrenches[0:6] = msg_utils.WrenchMsg_to_ndarray(data.wrench)
        elif arm == "left":
            self._wrenches[6:12] = msg_utils.WrenchMsg_to_ndarray(data.wrench)
    
    def _callback_yumi_kinematics(self, data: YumiKinematics) -> None:
        """ Updates forward kinematics using KDL instead of TF tree
        """
        self._update_individual(data)
        if self.options & COORDINATED:
            self._update_coordinated()
        # publish state
        self._callback_robot_state(self.options)
        
    
    def _update_individual(self, data: YumiKinematics):
        state = self.state
        # update joint position, velocity ... 
        state.joint_pos = np.asarray(data.jointPosition)[:14]  # simulation adds gripping position
        state.joint_vel = np.asarray(data.jointVelocity)
        # ... and jacobian
        jacobians_arms = np.asarray(data.jacobian[1].data).reshape((6,7,2))
        state.jacobian = jacobian_combine(jacobians_arms[:,:,0], jacobians_arms[:,:,1])  # right, left
        
        # update gripper jacobian ...
        jacobian_grippers = np.asarray(data.jacobian[0].data).reshape((6,7,2))
        state.jacobian_grippers = jacobian_combine(jacobian_grippers[:,:,0], jacobian_grippers[:,:,1])  # right, left
        # ... and pose ...
        state.pose_gripper_r = msg_utils.PoseMsg_to_Frame(data.forwardKinematics[0])
        state.pose_gripper_l = msg_utils.PoseMsg_to_Frame(data.forwardKinematics[1])
        # ... and velocity
        pose_grippers_vel = state.jacobian_grippers @ state.joint_vel
        state.pose_gripper_r.vel = pose_grippers_vel[:6]
        state.pose_gripper_l.vel = pose_grippers_vel[6:]
        
        # update elbow jacobian ... 
        jacobians_elbows = np.asarray(data.jacobian[2].data).reshape((6,4,2))
        state.jacobian_elbows = jacobian_combine(jacobians_elbows[:,:,0], jacobians_elbows[:,:,1])  # right, left
        # ... and pose ...
        state.pose_elbow_r = msg_utils.PoseMsg_to_Frame(data.forwardKinematics[4])
        state.pose_elbow_l = msg_utils.PoseMsg_to_Frame(data.forwardKinematics[5])
        # ... and velocity
        pose_elbow_vel = state.jacobian_elbows @ state.joint_vel[[0,1,2,3,7,8,9,10]]
        state.pose_elbow_r.vel = pose_elbow_vel[:6]
        state.pose_elbow_l.vel =  pose_elbow_vel[6:]
         
        # force
        state.effector_wrc = self._wrenches
        state.joint_tau = state.jacobian_grippers.T @ state.effector_wrc
    
    def _update_coordinated(self):
        state = self.state
        link = self.link_mat
        
        alpha = state.alpha
        alpha_ = 1 - state.alpha
        beta = alpha / (alpha**2 + alpha_**2)
        beta_ = alpha_ / (alpha**2 + alpha_**2)
        
        # if np.isclose(quat_diff(rot_abs, state.pose_abs.rot).w, 0):
        #     rot_abs = -rot_abs
        
        # state.pose_abs = state.pose_gripper_r @ (state.pose_gripper_r.inv() @ state.pose_gripper_l).partial(alpha_)
        # state.pose_rel = state.pose_gripper_r.partial(beta_).inv() @ state.pose_gripper_l.partial(beta)
        
        # absolute pose, average of the grippers wrt base frame
        pos_abs = alpha_ * state.pose_gripper_r.pos + alpha * state.pose_gripper_l.pos
        rot_diff = quat_diff(state.pose_gripper_l.rot, state.pose_gripper_r.rot)
        rot_diff_asym = np.exp(alpha_ * np.log(rot_diff))
        rot_abs = rot_diff_asym * state.pose_gripper_l.rot
        state.pose_abs = Frame(pos_abs, rot_abs)
        # relative pose, difference of the grippers wrt absolute frame
        pose_abs_inv = state.pose_abs.inv()
        pose_r_wrt_abs = pose_abs_inv @ state.pose_gripper_r
        pose_l_wrt_abs = pose_abs_inv @ state.pose_gripper_l        
        pos_rel = beta * pose_r_wrt_abs.pos - beta_ * pose_l_wrt_abs.pos
        rot_r_rel = np.exp(beta * np.log(pose_r_wrt_abs.rot))  # see above the explanation
        rot_l_rel = np.exp(beta_ * np.log(pose_l_wrt_abs.rot))
        rot_rel = quat_diff(rot_l_rel, rot_r_rel)
        state.pose_rel = Frame(pos_rel, rot_rel)
        
        # absolute linking matrix: maps gripper velocities to the velocity average
        link[:6,:6] = alpha_ * self._cache_eye
        link[:6,6:] = alpha  * self._cache_eye
        # relative linking matrix: maps gripper velocities to the velocity difference wrt absolute frame
        base_ee_to_abs_rel_trans = jacobian_change_frames( alpha_ * state.pose_gripper_r.pos - alpha * state.pose_gripper_l.pos, state.pose_abs.rot.conjugate() )
        link[6:,:6] = base_ee_to_abs_rel_trans @ ( beta *self._cache_eye)
        link[6:,6:] = base_ee_to_abs_rel_trans @ (-beta_*self._cache_eye)
        
        state.jacobian_coordinated = link @ state.jacobian_grippers
        # set velocities (avoid np.concatenate as it is expensive, prefer slicing (still expensive))
        state.pose_abs.vel = link[:6,:6] @ state.pose_gripper_r.vel + link[:6,6:] @ state.pose_gripper_l.vel
        state.pose_rel.vel = link[6:,:6] @ state.pose_gripper_r.vel + link[6:,6:] @ state.pose_gripper_l.vel
        
        # update wrenches
        # (using the kineto-statics duality, i.e. pose_wrench = link.T @ wrench_coordinated )
        wrench_coordinated = np.linalg.inv(link.T) @ state.effector_wrc
        state.pose_wrench_abs = wrench_coordinated[:6]
        state.pose_wrench_rel = wrench_coordinated[6:]


    @staticmethod
    def _make_jointstate_msg(joint_pos : np.ndarray, joint_vel : np.ndarray, grip : float):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [f"j{i+1}" for i in range(len(joint_pos))] + ["grip"]
        msg.position = joint_pos.tolist() + [grip]
        msg.velocity = joint_vel.tolist() + [0]
        msg.effort = [0.] * (len(joint_pos) + 1)
        return msg

    @staticmethod
    def _make_jacobian_msg(jacobian : np.ndarray):
        msg = Jacobian()
        msg.header.stamp = rospy.Time.now()
        msg.dof = jacobian.shape[1]
        msg.vx = jacobian[0, :].tolist()
        msg.vy = jacobian[1, :].tolist()
        msg.vz = jacobian[2, :].tolist()
        msg.wx = jacobian[3, :].tolist()
        msg.wy = jacobian[4, :].tolist()
        msg.wz = jacobian[5, :].tolist()
        return msg

    @staticmethod
    def _make_pose_msg(pose : Frame):
        msg = Pose()
        msg.position.x = pose.pos[0]
        msg.position.y = pose.pos[1]
        msg.position.z = pose.pos[2]
        msg.orientation.w = pose.rot.w
        msg.orientation.x = pose.rot.x
        msg.orientation.y = pose.rot.y
        msg.orientation.z = pose.rot.z
        return msg
    
    @staticmethod
    def _make_twist_msg(twist : np.ndarray):
        msg = Twist()
        msg.linear.x = twist[0]
        msg.linear.y = twist[1]
        msg.linear.z = twist[2]
        msg.angular.x = twist[3]
        msg.angular.y = twist[4]
        msg.angular.z = twist[5]
        return msg
    
    @staticmethod
    def _make_wrench_msg(wrench : np.ndarray):
        msg = Wrench()
        msg.force.x = wrench[0]
        msg.force.y = wrench[1]
        msg.force.z = wrench[2]
        msg.torque.x = wrench[3]
        msg.torque.y = wrench[4]
        msg.torque.z = wrench[5]
        return msg
    
    
    def _callback_robot_state(self, options : int):
        
        msg = RobotState()
        msg.header.stamp = rospy.Time.now()
        
        msg.jointState = [self._make_jointstate_msg(self.state.joint_pos_r, self.state.joint_vel_r, self.state.grip_r), 
                          self._make_jointstate_msg(self.state.joint_pos_l, self.state.joint_vel_l, self.state.grip_l)]
        msg.jointStateName = ["arm_r", "arm_l"]
        
        msg.jacobian = []
        msg.jacobianName = []
        msg.pose = []
        msg.poseTwist = []
        msg.poseWrench = []
        msg.poseName = []
        
        if options & RIGHT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_gripper_r))
            msg.jacobianName.append("gripper_r")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_gripper_r))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_gripper_r.vel))
            msg.poseWrench.append(self._make_wrench_msg(self.state.pose_wrench_r))
            msg.poseName.append("gripper_r")
            
        if options & LEFT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_gripper_l))
            msg.jacobianName.append("gripper_l")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_gripper_l))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_gripper_l.vel))
            msg.poseWrench.append(self._make_wrench_msg(self.state.pose_wrench_l))
            msg.poseName.append("gripper_l")
        
        if options & ABSOLUTE:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_coordinated_abs))
            msg.jacobianName.append("absolute")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_abs))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_abs.vel))
            msg.poseWrench.append(self._make_wrench_msg(self.state.pose_wrench_abs))
            msg.poseName.append("absolute")
        
        if options & RELATIVE:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_coordinated_rel))
            msg.jacobianName.append("relative")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_rel))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_rel.vel))
            msg.poseWrench.append(self._make_wrench_msg(self.state.pose_wrench_rel))
            msg.poseName.append("relative")
        
        if options & ELBOW_RIGHT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_elbow_r))
            msg.jacobianName.append("elbow_r")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_elbow_r))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_elbow_r.vel))
            msg.poseWrench.append(Wrench())
            msg.poseName.append("elbow_r")
        
        if options & ELBOW_LEFT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian_elbow_l))
            msg.jacobianName.append("elbow_l")
            # pose + twist + wrench
            msg.pose.append(self._make_pose_msg(self.state.pose_elbow_l))
            msg.poseTwist.append(self._make_twist_msg(self.state.pose_elbow_l.vel))
            msg.poseWrench.append(Wrench())
            msg.poseName.append("elbow_l")
        
        if options & ARM_RIGHT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian[:6, :7]))
            msg.jacobianName.append("arm_r")
        
        if options & ARM_LEFT:
            # jacobian
            msg.jacobian.append(self._make_jacobian_msg(self.state.jacobian[6:, 7:]))
            msg.jacobianName.append("arm_l")
        
        self._robot_state_publisher.publish(msg)
        

def main():
    
    # starting ROS node
    rospy.init_node("robot_state_updater", anonymous=False)
    
    coordinated_symmetry = rospy.get_param("~coordinated_symmetry")
    update_options = rospy.get_param("~update_options")
        
    robot_state = YumiCoordinatedRobotState(symmetry=coordinated_symmetry)
    robot_state_updater = YumiStateUpdater(robot_state, update_options)
        
    rospy.spin()

if __name__ == "__main__":
    main()
