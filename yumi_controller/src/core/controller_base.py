from typing import Any, Tuple
from abc import ABCMeta, abstractmethod

import rospy
import numpy as np
from threading import Lock

from std_msgs.msg import Float64MultiArray as Float64MultiArrayMsg
from abb_robot_msgs.msg import SystemState as SystemStateMsg
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv
from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv
from yumi_controller.msg import RobotState as RobotStateMsg

from .robot_state import YumiCoordinatedRobotState
from .parameters import Parameters
from .msg_utils import RobotStateMsg_to_YumiCoordinatedRobotState

from dynamics.controllers import AbstractController, AbstractDevice, AbstractDeviceState


from .ik_solver import IKSolver
from .ik_algorithms import HQPIKAlgorithm, PINVIKAlgorithm


# TODO maybe make YumiCoordinatedRobotState an AbstractDeviceState instead of this
class YumiDeviceState(YumiCoordinatedRobotState, AbstractDeviceState):
    def __init__(self) -> None:
        super().__init__()

class YumiDevice(AbstractDevice[YumiDeviceState]):
    
    def __init__(self):
        super().__init__()
        # yumi state subscriber
        self._cache_state: YumiDeviceState
        self._device_ready = False
        self._device_ready_changed = False
        rospy.Subscriber("/yumi/robot_state_coordinated", RobotStateMsg, self._callback_yumi_state, queue_size=1, tcp_nodelay=False)
        # ensure to start the controller with a real robot state 
        # (no wait means default state (all zeros), very bad)
        rospy.wait_for_message("/yumi/robot_state_coordinated", RobotStateMsg)
        
        # command publishers
        self._pub_yumi = YumiVelocityCommand()
        self._pub_grip = YumiGrippersCommand()
        
        # EGM error handler and status updater (updates `self._device_ready`)
        self._start_rapid = rospy.ServiceProxy("/yumi/rws/start_rapid", TriggerWithResultCodeSrv)
        # TODO handle other flags in the message (flags are: motors_on, auto_mode, rapid_running)
        rospy.Subscriber("/yumi/rws/system_states", SystemStateMsg, self._callback_yumi_rapid_state, queue_size=1, tcp_nodelay=False)
        rospy.wait_for_message("/yumi/rws/system_states", SystemStateMsg)
    
    def _callback_yumi_rapid_state(self, data: SystemStateMsg):
        self._cache_rws_auto_mode = data.auto_mode

    def _callback_yumi_state(self, data: RobotStateMsg):
        self._cache_state = RobotStateMsg_to_YumiCoordinatedRobotState(data)
        self._cache_state.time = rospy.Time.now()
    
    def did_ready_change(self):
        return self._device_ready_changed
    
    def is_ready(self) -> bool:
        return self._device_ready
    
    def read(self) -> YumiDeviceState:
        """ Stores the constantly updating state of Yumi inside the variables 
            actually used by the controller, effectively updating the state 
            in the controller. The data coming from Yumi might be old (because 
            of a disconnection), thus the RWS status is used as Yumi status.
        """
        # update status and set "status changed" flag
        status = self._cache_rws_auto_mode
        self._device_ready_changed = status != self.is_ready()
        self._device_ready = status
        return self._cache_state
    
    def send(self, command: Tuple[np.ndarray, float, float]):
        dq_target, grip_r, grip_l = command
        # yumi control command and gripper control command (if any)
        # avoid sendind commands all the time to optimize bandwidth
        self._pub_yumi.send_velocity_cmd(dq_target)
        if grip_r is not None or grip_l is not None:
            self._pub_grip.send_position_cmd(grip_r, grip_l)


# TODO why dual?
class YumiDualController(AbstractController[YumiDeviceState], metaclass=ABCMeta):
    """ Class for controlling YuMi, inherit this class and create your own 
        `.policy()` and `.clear()` functions. The `.policy()` function outputs 
        an action `dict`, which is then passed to the `._set_action()` function.
        This abstract class reads YuMi state from `/yumi/robot_state_coordinated`
        and sends velocity commands to `/yumi/egm/joint_group_velocity_controller/command`
        and gripper commands via service `/yumi/rws/sm_addin/set_sg_command`.
    """
    
    def __init__(self, robot_handle: YumiDevice, iksolver: str = "pinv"):
        self._device : YumiDevice
        super().__init__(robot_handle)
        
        # TODO extract me from here
        # setup the IK solvers
        self._iksolver = IKSolver()
        self._iksolver.register(PINVIKAlgorithm())
        self._iksolver.register(HQPIKAlgorithm())
        # select the IK solver
        self._iksolver.switch(iksolver)
        
    @property
    def yumi_state(self) -> YumiCoordinatedRobotState:
        return self._device_last_state
    
    @property
    def yumi_time(self) -> rospy.Time:
        return self._device_last_state.time
    
    def start(self):
        super().start(Parameters.update_rate)

    def _inner_loop(self, control_rate: float):
        """ ROS implementation of the original function.
        """
        rate = rospy.Rate(control_rate)
        while not rospy.is_shutdown():
            self.cycle()
            rate.sleep()
        # when the controller is shut down, send a stop command
        stop_commands = 3
        for i in range(stop_commands):
            command = (np.zeros(Parameters.dof), None, None)
            self._device.send(command)
            print(f"Sent stop command ({i+1}/{stop_commands})")
            
    def _on_device_lost(self):
        """ Decides what happens when control mode goes from "auto" to "manual".
        """
        print("Controller lost device after \"device_lost\" event")
    
    def _on_device_regained(self):
        """ Decides what happens when control mode goes from "manual" to "auto".
        """
        self.reset()
        print("Controller ran \"reset()\" after \"device_regained\" event")
        self._device._start_rapid.call()
        print("Restared RAPID")
    
    def _device_is_ready(self) -> bool:
        """ Calls the default `self._device.is_ready()` but then runs status 
            change logic before returning it.
        """
        ret = self._device.is_ready()
        if self._device.did_ready_change():
            if self._device.is_ready():
                # if auto_mode was off and now it's on (eg. after acknoledgment of EGM error)
                print("Regained control (auto_mode=true)")
                self._on_device_regained()
            else:
                # if auto_mode was on and now it's off (eg. after "joint contraint violation" error)
                print("Lost control (auto_mode=false)")
                self._on_device_lost()
        return ret
    
    @abstractmethod
    def reset(self):
        """ Method called when EGM stops.
        """
        raise NotImplementedError()
    
    def default_policy(self, state: YumiDeviceState) -> dict:
        action = {
            "control_space": "joint_space",
            "joint_velocities": np.zeros(Parameters.dof)}
        return action
    
    @abstractmethod
    def policy(self, state: YumiDeviceState) -> dict:
        """ This function should generate velocity commands for the controller.
            There are three control modes: 
            1. joint space control
            2. individual control in cartesian space with `yumi_base_link` as reference frame
            3. coordinated manipulation with absolute and relative control. 
            
            All the inverse kinematics required by this action will solved in 
            the `self._solve_action()` function using the selected solver. 
            The state of the robot is found in `self.yumi_state`, in particular 
            the `joint_pos`, `pose_gripper_r`, and `pose_gripper_l` variables.
            For more information, look at `self._solve_action()`.
        """
        raise NotImplementedError()

    def _solve_action(self, action: dict) -> Tuple[np.ndarray, float, float]:
        """ Sets an action and controls the robot.
            :param action: the action to solve; has to contain certain allowed keys.
            :key `action["routine_*"]`: specific commands (eg. `"routine_ready_pose"`)
            :key `action["control_space"]`: determines which control mode {`"joint_space"`, `"individual"`, `"coordinated"`}
            :key `action["joint_velocities"]`: [right, left] shape(14) with joint velocities (rad/s) (needed for mode `"joint_space"`)
            :key `action["right_velocity"]`: shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `"individual"`)
            :key `action["left_velocity"]`: shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `"individual"`)
            :key `action["absolute_velocity"]`: shape(6) with cartesian velocities in yumi base frame (m/s, rad/s) (needed for mode `"coordinated"`)
            :key `action["relative_velocity"]`: shape(6) with cartesian velocities in absolute frame (m/s, rad/s) (needed for mode `"coordinated"`)
            :key `action["timestep"]`: float with the current timestep, if needed by the IK solver (s) (needed for each mode except `"joint_space"`)
            :key `action["gripper_right"]`: float for gripper position (mm)
            :key `action["gripper_left"]`: float for gripper position (mm)
            For more information see examples or documentation.
        """
        # get joint velocities and publish them
        if action["control_space"] == "joint_space":
            dq_target = action["joint_velocities"]
        else:
            dq_target = self._iksolver(action, self.yumi_state)
            
        # log joints with clipping velocities
        vel_clip_r = np.abs(dq_target[0:7]) > Parameters.joint_velocity_bound
        vel_clip_l = np.abs(dq_target[7:14]) > Parameters.joint_velocity_bound
        if np.any(vel_clip_r) or np.any(vel_clip_l):
            idxs = np.arange(7) + 1
            labels = "".join([f" R{i}" for i in idxs[vel_clip_r]]) \
                   + "".join([f" L{i}" for i in idxs[vel_clip_l]])
            print(f"Joints [{labels} ] are clipping!")
        
        command = (dq_target, action.get("gripper_right"), action.get("gripper_left"))
        return command


from .routine_sm import RoutineStateMachine
from .routines import ReadyPoseRoutine, CalibPoseRoutine

class RoutinableYumiController(YumiDualController):

    def __init__(self, robot_handle: YumiDevice, iksolver: str = "pinv"):
        super().__init__(robot_handle, iksolver)
        
        # routine variables
        self._lock_routine_request = Lock()
        self._routine_request = None
        self._routine_machine = RoutineStateMachine()
        # TODO load these from constructor or method, not here
        self._routine_machine.register(ReadyPoseRoutine())
        self._routine_machine.register(CalibPoseRoutine())
    
    @abstractmethod
    def reset(self):
        """ Method called when EGM stops.
        """
        raise NotImplementedError()
    
    def request_routine(self, name: str):
        """ Set the routine to run. This can be done either internally via in
            the new `self.the_policy()` function or externally in another thread.
        """
        with self._lock_routine_request:
            self._routine_request = name
    
    def _inner_policy(self, state: YumiDeviceState) -> dict:
        """ New internal policy for the controller. Now, before computing the 
            policy, run the requested rountine, if any is requested or already
            running. Otherwise, run the policy.
        """
        # copy the request to avoid locking it for long
        with self._lock_routine_request:
            request = self._routine_request
            self._routine_request = None
        # execute the request (if exists)
        action, done = self._routine_machine.run(self.yumi_state, request)
        if done is True:
            # routine just finished, reset controller first
            self.reset()
        elif action is not None:
            # action from routine exists, return it
            return action
        
        return self.policy(state)
    
    @abstractmethod
    def policy(self, state: YumiDeviceState) -> dict: 
        raise NotImplementedError()


# UTILS

class YumiVelocityCommand(object):
    """ Used for storing the velocity command for yumi
    """
    def __init__(self):
        self._pub = rospy.Publisher("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArrayMsg, queue_size=1, tcp_nodelay=True)

    def send_velocity_cmd(self, joint_velocity: np.ndarray):
        """ Velocity should be an np.array() with 14 elements, [right arm, left arm]
        """
        # flip the arry to [left, right]
        msg = Float64MultiArrayMsg(
            data=joint_velocity[7:14].tolist() + joint_velocity[0:7].tolist())
        self._pub.publish(msg)

class YumiGrippersCommand(object):
    """ Class for controlling the grippers on YuMi, the grippers are controlled
        in [mm] and uses ros service
    """
    def __init__(self):
        # rosservice, for control over grippers
        self._service_SetSGCommand = rospy.ServiceProxy("/yumi/rws/sm_addin/set_sg_command", SetSGCommandSrv, persistent=True)
        self._service_RunSGRoutine = rospy.ServiceProxy("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCodeSrv, persistent=True)
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
