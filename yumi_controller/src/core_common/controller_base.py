from abc import ABCMeta, abstractmethod
from enum import Enum

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

from dynamics.controllers import (
    AbstractController, AbstractDevice, 
    AbstractDeviceState, AbstractDeviceAction, AbstractDeviceCommand
)

# TODO maybe make YumiCoordinatedRobotState an AbstractDeviceState instead of this
class YumiDualDeviceState(YumiCoordinatedRobotState, AbstractDeviceState):
    def __init__(self) -> None:
        super().__init__()

class YumiDualDeviceAction(AbstractDeviceAction, dict):
    """ Represents an action for a dual-control ABB Dual-Arm Yumi.
        This action is a subclass of `Dict`, which allows to write
        complex and hot-swappable action solvers due to the 
        flexibility of dictionaries. Common fields are listed here:
        
        - `control_space` : determines which control mode. Options are `joint_space`, `individual`, `coordinated`
        - `velocity_joints` : [right, left] shape(14) with joint velocities (rad/s) (needed for mode `joint_space`)
        - `timestep` : float with the current timestep, if needed by the IK solver (s) (needed for each mode except `joint_space`)
        - `velocity_right` : shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `individual`)
        - `velocity_left` : shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `individual`)
        - `velocity_absolute` : shape(6) with cartesian velocities in yumi base frame (m/s, rad/s) (needed for mode `coordinated`)
        - `velocity_relative` : shape(6) with cartesian velocities in absolute frame (m/s, rad/s) (needed for mode `coordinated`)
        - `gripper_right` : float for gripper position (mm)
        - `gripper_left` : float for gripper position (mm)
    """
    
    class ControlSpace(Enum):
        JOINT_SPACE = "joint_space"
        INDIVIDUAL = "individual"
        COORDINATED = "coordinated"
        
        @classmethod    
        def from_str(cls, name: str):
            if name == "joint_space":
                return cls.JOINT_SPACE
            elif name == "individual":
                return cls.INDIVIDUAL
            elif name == "coordinated":
                return cls.COORDINATED
            else:
                raise NameError(f"No control space for the provided name: {name}")
    
    def __init__(self) -> None:
        super().__init__()
    
    def control_space(self, space : "YumiDualDeviceAction.ControlSpace"):
        self["control_space"] = space
    
    def velocity_joints(self, velocity : np.ndarray):
        assert velocity.shape == (Parameters.dof,)
        self["velocity_joints"] = velocity
    
    def timestep(self, value : float):
        self["timestep"] = value
    
    def velocity_right(self, velocity : np.ndarray):
        assert velocity.shape == (Parameters.dof_c_right,)
        self["velocity_right"] = velocity
    
    def velocity_left(self, velocity : np.ndarray):
        assert velocity.shape == (Parameters.dof_c_left,)
        self["velocity_left"] = velocity
    
    def velocity_absolute(self, velocity : np.ndarray):
        assert velocity.shape == (6,)
        self["velocity_absolute"] = velocity
    
    def velocity_relative(self, velocity : np.ndarray):
        assert velocity.shape == (6,)
        self["velocity_relative"] = velocity
    
    def gripper_right(self, value : float):
        self["gripper_right"] = value
    
    def gripper_left(self, value : float):
        self["gripper_left"] = value
 
class YumiDualDeviceCommand(AbstractDeviceCommand):
    def __init__(self) -> None:
        super().__init__()
        self._dq_target : np.ndarray = np.zeros(Parameters.dof)
        self._grip_r : float = None
        self._grip_l : float = None

    def dq_target(self, command : np.ndarray):
        assert command.shape == (Parameters.dof,)
        self._dq_target = command

    def grip_right(self, command : float):
        self._grip_r = command

    def grip_left(self, command : float):
        self._grip_l = command

# TODO why not dual?
class YumiDevice(AbstractDevice[YumiDualDeviceState, YumiDualDeviceCommand]):
    
    def __init__(self):
        super().__init__()
        # yumi state subscriber
        self._cache_state: YumiDualDeviceState
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
    
    def did_status_change(self):
        return self._device_ready_changed
    
    def is_ready(self) -> bool:
        return self._device_ready
    
    def read(self) -> YumiDualDeviceState:
        """ Stores the constantly updating state of Yumi inside the variables 
            actually used by the controller, effectively updating the state 
            in the controller. The data coming from Yumi might be old (because 
            of a disconnection), thus the RWS status is used as Yumi status.
        """
        # update status and set "status changed" flag
        current_status = self._cache_rws_auto_mode
        self._device_ready_changed = current_status != self.is_ready()
        self._device_ready = current_status
        return self._cache_state
    
    def send(self, command: YumiDualDeviceCommand):
        # yumi control command and gripper control command (if any)
        # avoid sendind commands all the time to optimize bandwidth
        self._pub_yumi.send_velocity_cmd(command._dq_target)
        if (command._grip_r is not None) or (command._grip_l is not None):
            self._pub_grip.send_position_cmd(command._grip_r, command._grip_l)


from .ik_solver import IKSolver
from .ik_algorithms import HQPIKAlgorithm, PINVIKAlgorithm

# TODO why dual?
class YumiDualController(
    AbstractController[YumiDualDeviceState, YumiDualDeviceAction, YumiDualDeviceCommand], 
    metaclass=ABCMeta
):
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
    
    def _on_device_regained(self, state: YumiDualDeviceState):
        """ Decides what happens when control mode goes from "manual" to "auto".
        """
        self.reset(state)
        print("Controller ran \"reset()\" after \"device_regained\" event")
        self._device._start_rapid.call()
        print("Restared RAPID")
    
    def _device_is_ready(self) -> bool:
        """ Calls the default `self._device.is_ready()` but then runs status 
            change logic before returning it.
        """
        ret = self._device.is_ready()
        if self._device.did_status_change():
            if self._device.is_ready():
                # if auto_mode was off and now it's on (eg. after acknoledgment of EGM error)
                print("Regained control (auto_mode=true)")
                state = self._device_read()
                self._on_device_regained(state)
            else:
                # if auto_mode was on and now it's off (eg. after "joint contraint violation" error)
                print("Lost control (auto_mode=false)")
                self._on_device_lost()
        return ret
    
    @abstractmethod
    def reset(self, state: YumiDualDeviceState):
        """ Method called when EGM stops.
        """
        raise NotImplementedError()
    
    def default_policy(self, state: YumiDualDeviceState) -> YumiDualDeviceAction:
        action = YumiDualDeviceAction()
        action.control_space(YumiDualDeviceAction.ControlSpace.JOINT_SPACE)
        action.velocity_joints(np.zeros(Parameters.dof))
        
    @abstractmethod
    def policy(self, state: YumiDualDeviceState) -> YumiDualDeviceAction:
        """ This function should generate velocity commands for the controller.
            There are three control modes: 
            1. joint space control
            2. individual control in cartesian space with `yumi_base_link` as reference frame
            3. coordinated manipulation with absolute and relative control. 
            
            All the inverse kinematics required by this action will solved in 
            the `self._solve_action()` function using the selected solver. 
            The state of the robot is found in parameter `state`, in particular 
            the `joint_pos`, `pose_gripper_r`, and `pose_gripper_l` variables.
            For more information on how to create and action, look the docs of
            `YumiDualDeviceAction`
        """
        raise NotImplementedError()

    def _solve_action(self, state: YumiDualDeviceState, action: YumiDualDeviceAction) -> YumiDualDeviceCommand:
        """ Convert a desired action to the required command using an IK solver,
            if necessary, and clip the commands.
            
            :param action: the action to be converted
            :returns: the reuqired command
        """
        # get joint velocities and publish them
        if action["control_space"] == YumiDualDeviceAction.ControlSpace.JOINT_SPACE:
            dq_target = action["velocity_joints"]
        else:
            dq_target = self._iksolver.solve(action, state)
            
        # log joints with clipping velocities
        vel_clip_r = np.abs(dq_target[0:7]) > Parameters.joint_velocity_bound
        vel_clip_l = np.abs(dq_target[7:14]) > Parameters.joint_velocity_bound
        if np.any(vel_clip_r) or np.any(vel_clip_l):
            idxs = np.arange(7) + 1
            labels = "".join([f" R{i}" for i in idxs[vel_clip_r]]) \
                   + "".join([f" L{i}" for i in idxs[vel_clip_l]])
            print(f"Joints [{labels} ] are clipping!")
        
        command = YumiDualDeviceCommand()
        command.dq_target(dq_target)
        command.grip_right(action.get("gripper_right"))
        command.grip_left(action.get("gripper_left"))
        return command


from typing import List
from .routine_sm import RoutineStateMachine, Routine

class RoutinableYumiController(YumiDualController):

    def __init__(self, robot_handle: YumiDevice, iksolver: str = "pinv", routines: List[Routine] = []):
        super().__init__(robot_handle, iksolver)
        
        # routine variables
        self._lock_routine_request = Lock()
        self._routine_request = None
        self._routine_machine = RoutineStateMachine()
        for routine in routines:
            self._routine_machine.register(routine)
        
    @abstractmethod
    def reset(self, state: YumiDualDeviceState):
        """ Method called when EGM stops.
        """
        raise NotImplementedError()
    
    def request_routine(self, name: str):
        """ Set the routine to run. This can be done either internally in
            the `self.policy()` function or externally in another thread. 
            If you do it internally, it will be executed in the next cycle.
        """
        with self._lock_routine_request:
            self._routine_request = name
    
    def _inner_policy(self, state: YumiDualDeviceState) -> YumiDualDeviceCommand:
        """ New internal policy for the controller. Now, before computing the 
            policy, run the requested rountine, if any is requested or already
            running. Otherwise, run the policy.
        """
        # copy the request to avoid locking it for long
        with self._lock_routine_request:
            request = self._routine_request
            self._routine_request = None
        # execute the request (if exists)
        action, done = self._routine_machine.run(state, request)
        if done is True:
            # routine just finished, reset controller first
            self.reset(state)
        elif action is not None:
            # action from routine exists, return it
            return action
        
        return self.policy(state)
    
    @abstractmethod
    def policy(self, state: YumiDualDeviceState) -> YumiDualDeviceCommand: 
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
