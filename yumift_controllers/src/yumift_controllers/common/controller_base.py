from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List

import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray as Float64MultiArrayMsg
from abb_robot_msgs.msg import SystemState as SystemStateMsg
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv
from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv

from yumift_common.robot_state import YumiCoordinatedRobotState
from yumift_common.constants import YumiRobotConstants
from .parameters import ControllerParameters
from ..misc.utils import RobotStateMsg_to_YumiCoordinatedRobotState

from yumift_msgs.msg import RobotState as RobotStateMsg


###############################################################################
#                              CONTROLLER DEVICE                              #
###############################################################################

from dynamicals.common.controllers import AbstractController, AbstractControllerAction
from dynamicals.common.devices import AbstractDevice, AbstractDeviceState,  AbstractDeviceCommand


# TODO maybe make YumiCoordinatedRobotState an AbstractDeviceState instead of this
class YumiDualDeviceState(AbstractDeviceState, YumiCoordinatedRobotState):
    def __init__(self) -> None:
        super().__init__()

class YumiDualDeviceCommand(AbstractDeviceCommand):
    def __init__(self, 
        dq_target : np.ndarray = np.zeros(YumiRobotConstants.DOF), 
        grip_r : float = None, 
        grip_l : float = None
    ) -> None:
        super().__init__()
        self._dq_target = dq_target
        self._grip_r = grip_r
        self._grip_l = grip_l

    def dq_target(self, target : np.ndarray):
        assert target.shape == (YumiRobotConstants.DOF,)
        self._dq_target = target

    def grip_right(self, target : float):
        self._grip_r = target

    def grip_left(self, target : float):
        self._grip_l = target

# TODO why not dual?
# TODO remove "/yumi" from everywhere
class YumiDevice(AbstractDevice[YumiDualDeviceState, YumiDualDeviceCommand]):
    
    def __init__(self, coordinated_balance : float = 0.5):
        super().__init__()
        # yumi state subscriber
        self._cache_state: YumiDualDeviceState
        self._device_ready = False
        self._device_ready_changed = False
        rospy.Subscriber("/yumi/unified/robot_state_coordinated", RobotStateMsg, self._callback_received_state, queue_size=1, tcp_nodelay=False)
        # ensure to start the controller with a real robot state 
        # (no wait means default state (all zeros), very bad)
        rospy.wait_for_message("/yumi/unified/robot_state_coordinated", RobotStateMsg)
        
        # command publishers
        self._pub_vel = YumiVelocityCommand()
        self._pub_grip = YumiGrippersCommand()
        
        # EGM error handler and status updater (updates `self._device_ready`)
        self._start_rapid = rospy.ServiceProxy("/yumi/rws/start_rapid", TriggerWithResultCodeSrv)
        rospy.Subscriber("/yumi/rws/system_states", SystemStateMsg, self._callback_received_rapid_state, queue_size=1, tcp_nodelay=False)
        rospy.wait_for_message("/yumi/rws/system_states", SystemStateMsg)
    
    def _callback_received_rapid_state(self, data: SystemStateMsg):
        self._cache_rws_auto_mode = data.auto_mode
        # TODO handle other flags in the message
        # data.motors_on
        # data.rapid_running
        # data.rapid_tasks
        # data.mechanical_units

    def _callback_received_state(self, data: RobotStateMsg):
        # TODO this is broken, type mismatch
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
        self._pub_vel.send_velocity_cmd(command._dq_target)
        if (command._grip_r is not None) or (command._grip_l is not None):
            self._pub_grip.send_position_cmd(command._grip_r, command._grip_l)

# UTILS

class YumiVelocityCommand(object):
    """ Used for storing the velocity command for yumi
    """
    def __init__(self):
        self._pub = rospy.Publisher("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArrayMsg, queue_size=1, tcp_nodelay=True)

    def send_velocity_cmd(self, joint_velocity: np.ndarray):
        """ Velocity should be an np.array() with 14 elements, [right arm, left arm]
        """
        # flip the array to [left, right] as required by ros_control velocity controller
        msg = Float64MultiArrayMsg(
            data=joint_velocity[7:14].tolist() + joint_velocity[0:7].tolist())
        self._pub.publish(msg)

class YumiGrippersCommand(object):
    """ Class for controlling the grippers on YuMi, the grippers are controlled
        in [mm] and uses ros service
    """
    def __init__(self):
        # ROS services for SmartGrippers
        self._service_SetSGCommand = rospy.ServiceProxy("/yumi/rws/sm_addin/set_sg_command", SetSGCommandSrv, persistent=True)
        self._service_RunSGRoutine = rospy.ServiceProxy("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCodeSrv, persistent=True)
        self._prev_gripper_r = 0
        self._prev_gripper_l = 0

    def _set_position_service(self, name: str, current: float, target: float, tol: float = 1e-5):
        """ Set the new position for a gripper
            :param name: EGM task to use
            :param current: current position value of the gripper
            :param target: target position value of the gripper
            :returns changed: position changed flag
        """
        if abs(current - target) >= tol:
            if target <= 0.1:
                # CLOSE command
                self._service_SetSGCommand.call(task=name, command=6)
            else:
                # GOTO command
                self._service_SetSGCommand.call(task=name, command=5, target_position=target)
            return True
        return False

    def send_position_cmd(self, gripper_r: float = None, gripper_l: float = None):
        """ Set new gripping position
            :param gripper_r: position in millimeters
            :param gripper_l: position in millimeters
        """
        did_something = False
        try:
            # stacks/set the commands for the grippers 
            # do not send the same command twice as grippers will momentarily regrip
            
            # set position values for the grippers
            if gripper_r is not None:
                if self._set_position_service("T_ROB_R", self._prev_gripper_r, gripper_r):
                    self._prev_gripper_r = gripper_r
                    did_something = True
            
            if gripper_l is not None:
                if self._set_position_service("T_ROB_L", self._prev_gripper_l, gripper_l):
                    self._prev_gripper_l = gripper_l
                    did_something = True

            # send the commands to the robot
            if did_something:
                self._service_RunSGRoutine.call()

        except Exception as ex:
            print(f"SmartGripper error : {ex}")

###############################################################################
#                                 CONTROLLERS                                 #
###############################################################################

class YumiDualDeviceAction(AbstractControllerAction, dict):
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
        assert velocity.shape == (YumiRobotConstants.DOF,)
        self["velocity_joints"] = velocity
    
    def timestep(self, value : float):
        self["timestep"] = value
    
    def velocity_right(self, velocity : np.ndarray):
        assert velocity.shape == (YumiRobotConstants.DOF_EE_RIGHT,)
        self["velocity_right"] = velocity
    
    def velocity_left(self, velocity : np.ndarray):
        assert velocity.shape == (YumiRobotConstants.DOF_EE_LEFT,)
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
 
from ..ik.solver import IKSolver, IKAlgorithm

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
    
    def __init__(self, yumi_device: YumiDevice, iksolvers : List[IKAlgorithm]):
        self._device : YumiDevice
        super().__init__(yumi_device)
        
        # TODO extract me from here
        # setup the IK solvers
        self._iksolver = IKSolver()
        for algo in iksolvers:
            self._iksolver.register(algo)
        # select first algorithm as default
        self._iksolver.switch(iksolvers[0].name)
    
    def start(self):
        super().start(ControllerParameters.update_rate)

    def stop(self):
        print("Controller shutting down")
        super().stop()
    
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
            command = YumiDualDeviceCommand(np.zeros(YumiRobotConstants.DOF), None, None)
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
    
    def fallback(self, state: YumiDualDeviceState) -> YumiDualDeviceAction:
        action = YumiDualDeviceAction()
        action.control_space(YumiDualDeviceAction.ControlSpace.JOINT_SPACE)
        action.velocity_joints(np.zeros(YumiRobotConstants.DOF))
        return action
        
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

    def solve_action(self, state: YumiDualDeviceState, action: YumiDualDeviceAction) -> YumiDualDeviceCommand:
        """ Convert a desired action to the required command using an IK solver,
            if necessary, and clip the commands.
            
            :param action: the action to be converted
            :returns: the required command
        """
        # solve action for joint velocities
        if action["control_space"] == YumiDualDeviceAction.ControlSpace.JOINT_SPACE:
            dq_target = action["velocity_joints"]
        else:
            dq_target = self._iksolver.solve(action, state)
                    
        # log joints with clipping velocities
        vel_clip_r = np.abs(dq_target[0:7]) > YumiRobotConstants.JOINT_VEL_AB
        vel_clip_l = np.abs(dq_target[7:14]) > YumiRobotConstants.JOINT_VEL_AB
        if np.any(vel_clip_r) or np.any(vel_clip_l):
            idxs = np.arange(7) + 1
            labels = "".join([f" R{i}" for i in idxs[vel_clip_r]]) \
                   + "".join([f" L{i}" for i in idxs[vel_clip_l]])
            print(f"Joints [{labels} ] are clipping!")
        
        # create command
        command = YumiDualDeviceCommand()
        command.dq_target(dq_target)
        command.grip_right(action.get("gripper_right"))
        command.grip_left(action.get("gripper_left"))
        return command
