#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

import rospy
import numpy as np

import pinocchio as pin

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from abb_egm_msgs.msg import EGMState

from core.robot_state import YumiCoordinatedRobotState
from dynamics.utils import jacobian_combine
from core import msg_utils


# oMdes = pin.SE3(np.eye(3), np.array([1.0, 0.0, 3.0]))
# quat = pin.Quaternion(oMdes.rotation)
# print(quat)
# print(pin.log(oMdes))

# m1 = pin.Motion(np.array([1, 2, 3]), np.array([-1, 0, 1]))
# m2 = pin.Motion(np.array([1, -2, 4]), np.array([1, -1, 0]))
# print(m1 + m2, np.random.random((6,6)) * m2)
# print(pin.exp(m1 + m2))


class YumiStateUpdater(object):
    def __init__(self, 
        state : YumiCoordinatedRobotState, 
        urdf_path : str,
        topic_egm_state : str, 
        topic_joint_state : str, 
        topic_sensor_r : str, 
        topic_sensor_l : str, 
        topic_robot_state : str
    ):
        self.state = state

        model = pin.buildModelFromUrdf(urdf_path)
        # data = model.createData()  # stores all the data (q, v, a) of the model
        # model.joints
        # data.oMi  # joint placements (SE3 object)
        # model.frames
        # data.oMf  # frame placements (SE3 object)

        # create a reduced model without the non-actuated joints
        joint_names = ["gripper_l_joint", "gripper_l_joint_m", "gripper_r_joint", "gripper_r_joint_m"]
        joint_ids = [model.getJointId(name) for name in joint_names]
        joint_vals = np.zeros(18)
        self.model = pin.buildReducedModel(model, joint_ids, joint_vals)
        self.data = self.model.createData()

        abb_urdf_joint_names = [
            'yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l',
            'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r'
        ]
        # ATTENTION: this naming is inconsistent with the naming found in ABB driver!
        #            joint_7 is between joint_2 and joint_3, but still refers 
        #            to the third joint in the chain. Also, left and right are 
        #            switched, and the names have different prefixes in general...
        abb_egm_joint_state_names = [
            "yumi_robr_joint_1", "yumi_robr_joint_2", "yumi_robr_joint_3", "yumi_robr_joint_4", "yumi_robr_joint_5", "yumi_robr_joint_6", "yumi_robr_joint_7", 
            "yumi_robl_joint_1", "yumi_robl_joint_2", "yumi_robl_joint_3", "yumi_robl_joint_4", "yumi_robl_joint_5", "yumi_robl_joint_6", "yumi_robl_joint_7"
        ]
        self._urdf_to_egm_joint_names_map = {urdf_name: egm_name for urdf_name, egm_name in zip(abb_urdf_joint_names[7:14], abb_egm_joint_state_names[0:7])} \
                                          + {urdf_name: egm_name for urdf_name, egm_name in zip(abb_urdf_joint_names[0:7], abb_egm_joint_state_names[7:14])}
        self._urdf_joint_ids_map = {urdf_name: i for i, urdf_name in enumerate(abb_urdf_joint_names)}
        
        self._egm_active = False
        rospy.Subscriber(topic_egm_state, EGMState, self.callback_egm_state, queue_size=1, tcp_nodelay=False)
        self._q = np.zeros(14)
        self._dq = np.zeros(14)
        rospy.Subscriber(topic_joint_state, JointState, self.callback_joint_state, queue_size=1, tcp_nodelay=False)
        # read force sensors
        self._wrenches = np.zeros(12)  # [fR, mR, fL, mL]
        rospy.Subscriber(topic_sensor_r, WrenchStamped, self.callback_ext_force, callback_args="right", queue_size=1, tcp_nodelay=False)
        rospy.Subscriber(topic_sensor_l, WrenchStamped, self.callback_ext_force, callback_args="left", queue_size=1, tcp_nodelay=False)

    def callback_egm_state(self, msg: EGMState):
        self._egm_active = msg.egm_channels[0].active and msg.egm_channels[1].active
    
    def callback_ext_force(self, data: WrenchStamped, arm: str):
        if arm == "right":
            self._wrenches[0:6] = msg_utils.WrenchMsg_to_ndarray(data.wrench)
        elif arm == "left":
            self._wrenches[6:12] = msg_utils.WrenchMsg_to_ndarray(data.wrench)
    
    def callback_joint_state(self, msg: JointState):
        # `msg.name` contains `abb_egm_joint_state_names`
        # Pinocchio contains `abb_urdf_joint_names`
        egm_values = {egm_name: (pos, vel) for egm_name, pos, vel in zip(msg.name, msg.position, msg.velocity)}
        for urdf_name in self.model.names[1:]:
            egm_name = self._urdf_to_egm_joint_names_map.get(urdf_name, None)
            i = self._urdf_joint_ids_map.get(urdf_name, None)
            self._q[i] = egm_values[egm_name][0]
            self._dq[i] = egm_values[egm_name][1]
            
    def update(self):
        self._update(self._q, self._dq, self._wrenches)

    def _update(self, q: np.ndarray, dq: np.ndarray, he: np.ndarray):
        """ Update the internal state object using Pinocchio
        :param q: the joint state position
        :param dq: the joint state velocity
        :param he: the end effector exogenous wrenches
        """
        pin.forwardKinematics(self.model, self.data, q, dq)  # fills data with new joint frame info (pos, vel, acc) based on the current joint state (q, dq, ddq)
        pin.updateFramePlacements(self.model, self.data)  # fills data with new frame placements based on joint placements
        # compute poses
        world_to_base = self.data.oMf[self.model.getFrameId("yumi_base_link")]
        pose_grip_r = self.data.oMf[self.model.getFrameId("gripper_r_tip")]
        pose_grip_l = self.data.oMf[self.model.getFrameId("gripper_l_tip")]
        pose_elb_r = self.data.oMf[self.model.getFrameId("yumi_link_4_r")]
        pose_elb_l = self.data.oMf[self.model.getFrameId("yumi_link_4_l")]
        # compute jacobians
        # pin.computeJointJacobians(model, data, q)
        jac_grip_r = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId("gripper_r_tip"), pin.ReferenceFrame.WORLD)
        jac_grip_l = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId("gripper_l_tip"), pin.ReferenceFrame.WORLD)
        jac_elb_r = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId("yumi_link_4_r"), pin.ReferenceFrame.WORLD)
        jac_elb_l = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId("yumi_link_4_l"), pin.ReferenceFrame.WORLD)

        # update joint position, velocity ... 
        self.state.joint_pos = q
        self.state.joint_vel = dq
        # update gripper jacobian ...
        self.state.jacobian_grippers = jacobian_combine(jac_grip_r, jac_grip_l)
        # ... and pose and velocity
        self.state.pose_gripper_r = world_to_base.actInv(pose_grip_r)
        self.state.pose_gripper_l = world_to_base.actInv(pose_grip_l)
        # update elbow jacobian ... 
        self.state.jacobian_elbows = jacobian_combine(jac_elb_r, jac_elb_l)
        # ... and pose and velocity
        self.state.pose_elbow_r = world_to_base.actInv(pose_elb_r)
        self.state.pose_elbow_l = world_to_base.actInv(pose_elb_l)
        
        # force
        self.state.pose_wrench = he
        self.state.joint_torque = self.state.jacobian_grippers.T @ self.state.pose_wrench
        
        print(self.state.pose_gripper_r)


if __name__ == "__main__":
    
    # starting ROS node
    rospy.init_node("robot_state_updater", anonymous=False)
    
    urdf_path = "/home/seba/yumift_ws/src/yumift/yumi_description/urdf/yumi.urdf"
    topic_egm_state = "/yumi/egm/egm_states"
    topic_joint_state = "/yumi/egm/joint_states"
    topic_sensor_r = "/ftsensor_r/world"
    topic_sensor_l = "/ftsensor_l/world"
    topic_robot_state = "/yumi/robot_state_coordinated"
    symmetry = 0
    
    state = YumiCoordinatedRobotState(symmetry=symmetry)
    updater = YumiStateUpdater(state, urdf_path, topic_egm_state, topic_joint_state, topic_sensor_r, topic_sensor_l, topic_robot_state)
    
    rate = rospy.Rate(250)
    while not rospy.is_shutdown():
        # init = rospy.Time.now()
        updater.update()
        # print(1/(rospy.Time.now() - init).to_sec())
        rate.sleep()




import pinocchio as pin
import numpy as np
np.set_printoptions(linewidth=100)
urdf_path = "/home/seba/yumift_ws/src/yumift/yumi_description/urdf/yumi.urdf"
mesh_dir = "/home/seba/yumift_ws/src/yumift/yumi_description/meshes"
# model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, [mesh_dir], verbose=True)
# data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)
model = pin.buildModelFromUrdf(urdf_path)
joint_names = ["gripper_l_joint", "gripper_l_joint_m", "gripper_r_joint", "gripper_r_joint_m"]             
joint_ids = [model.getJointId(name) for name in joint_names]        
joint_vals = np.zeros(18)
model = pin.buildReducedModel(model, joint_ids, joint_vals)             
data = model.createData()

q = pin.randomConfiguration(model)  # get random configuration
q = pin.neutral(model)  # get neutral configuration (usually is just zeros)
dq = pin.randomConfiguration(model)
q = np.array([-0.7, -1.7,  0.8, 1.0,  2.2, 1.0, 0.0,   # left
               0.7, -1.7, -0.8, 1.0, -2.2, 1.0, 0.0])  # right
q = np.array([ 0.0, -2.270,  2.356, 0.524, 0.0, 0.670, 0.0,
               0.0, -2.270, -2.356, 0.524, 0.0, 0.670, 0.0])
dq = np.zeros(14)

pin.forwardKinematics(model, data, q, dq)
pin.updateFramePlacements(model, data)
# pin.updateFramePlacement(model, data, model.getFrameId("gripper_r_tip"))  # or single frame placement
# pin.updateGlobalPlacements(mode, data)  # this i don't know

pose_grip_r = data.oMf[model.getFrameId("gripper_r_tip")]
pose_grip_l = data.oMf[model.getFrameId("gripper_l_tip")]

jac_grip_r = pin.computeFrameJacobian(model, data, q, model.getFrameId("gripper_r_tip"), pin.ReferenceFrame.WORLD)
jac_grip_l = pin.computeFrameJacobian(model, data, q, model.getFrameId("gripper_l_tip"), pin.ReferenceFrame.WORLD)

vel_grip_r = pin.getVelocity(model, data, model.getFrameId("gripper_r_tip"), pin.ReferenceFrame.WORLD)
vel_grip_l = pin.getVelocity(model, data, model.getFrameId("gripper_l_tip"), pin.ReferenceFrame.WORLD)



# model.subtrees[5].tolist()

# model.name
# model.names  # joint names
# model.nbodies  # ????
# model.nframes  # overall number of frames (in the URDF)
# model.njoints  # number of joints

# model.joints  # list of joint objects specifying their indeces
# model.joints[3]  # shows various joint values 
#                  # (e.g. number of q and v variables assosiated to it, and 
#                  # their indeces in the list of joints)
#                  # indeces can be changed for more convenient numbering

# model.frames  # list of frame objects specifying their properties
#               # (e.g. inertia, relative placement wrt parent)

# model.jointPlacements  # ????

# model.nq  # number of joints
# model.nqs  # number of variables per joint
# model.nv  # number of velocities
# model.nvs  # number of variables per velocity

# model.parents  # index of the parent of each joint



# pin.computeAllTerms(model, data, q, dq)
# pin.forwardDynamics(model, data, tau)  # computes ddq (assumes computeAllTerms is called)
# pin.framesForwardKinematics(model, data, q)  #  forwardKinematics + updateFramePlacements

# pin.computeJointJacobiansTimeVariation(model, data, q, v)
# data.dJ

# pin.nonLinearEffects(model, data, q, v)


# pin.Jlog6(M)
# pin.Jexp6(M)


# pin.SE3ToXYZQUAT()
# pin.XYZQUATToSE3()

# pin.distance(model, q1, q2)
# pin.squaredDistance(model, q1, q2)
# pin.normalize(q)
# mat = pin.skew(vec)
# vec = pin.unSkew(mat)
# mat = pin.skewSquare(u, v)  # computes M s.t. Mw = u x v x w
# pin.rpy.rpyToMatrix(r, p, y)
# pin.utils.rotate('x', np.pi/4)
