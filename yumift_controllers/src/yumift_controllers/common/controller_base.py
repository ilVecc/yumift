from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List
from typing_extensions import override

import rospy
import numpy as np

from .parameters import ControllerParameters
from .device import YumiDualDeviceState, YumiDualDeviceCommand, YumiDevice
from yumift_common.constants import YumiRobotConstants

from dynamicals.common.controllers import AbstractControllerAction
from dynamicals.impl import AbstractROSController


class MixedVelocityYumiAction(AbstractControllerAction, dict):
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
        self.control_space(MixedVelocityYumiAction.ControlSpace.JOINT_SPACE)
        self.velocity_joints(np.zeros(YumiRobotConstants.DOF))
    
    def control_space(self, space : "MixedVelocityYumiAction.ControlSpace"):
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
    AbstractROSController[YumiDualDeviceState, MixedVelocityYumiAction, YumiDualDeviceCommand], 
    metaclass=ABCMeta
):
    """ Class for controlling YuMi, inherit this class and create your own 
        `.policy()` and `.clear()` functions. The `.policy()` function outputs 
        an action `dict`, which is then passed to the `._set_action()` function.
        This abstract class reads YuMi state from `/yumi/robot_state_coordinated`
        and sends velocity commands to `/yumi/egm/joint_group_velocity_controller/command`
        and gripper commands via service `/yumi/rws/sm_addin/set_sg_command`.
    """
    
    def __init__(self, yumi_device: YumiDevice, ikalgorithms : List[IKAlgorithm]):
        self._device : YumiDevice
        self._device_ready_prev = False
        super().__init__(yumi_device)
        
        # TODO extract me from here
        # setup the IK solvers and select the first one as default
        self._iksolver = IKSolver(ikalgorithms)
        self._iksolver.switch(ikalgorithms[0].name)
    
    @override
    def start(self):
        super().start(ControllerParameters.update_rate)

    def _on_device_lost(self):
        """ Decides what happens when e.g. control mode goes from "auto" to "manual".
        """
        print("Controller lost device after \"device_lost\" event")
    
    def _on_device_regained(self, state: YumiDualDeviceState):
        """ Decides what happens when e.g. control mode goes from "manual" to "auto".
        """
        self.reset(state)
        print("Controller ran \"reset()\" after \"device_regained\" event")
        self.device_reset()
        print("Restared RAPID")
    
    @override
    def device_is_ready(self) -> bool:
        """ Calls the default `self._device.is_ready()` but then runs status 
            change logic before returning it.
        """
        device_ready_curr = self._device.is_ready()
        # handle change in device readyness
        if device_ready_curr != self._device_ready_prev:
            self._device_ready_prev = device_ready_curr
            if device_ready_curr:
                # if auto_mode was off and now it's on (eg. after acknoledgment of EGM error)
                print("Regained control (auto_mode=true)")
                state = self.device_read()
                self._on_device_regained(state)
            else:
                # if auto_mode was on and now it's off (eg. after "joint contraint violation" error)
                print("Lost control (auto_mode=false)")
                self._on_device_lost()
        
        return device_ready_curr
    
    @abstractmethod
    @override
    def reset(self, state: YumiDualDeviceState):
        """ Method called when EGM stops.
        """
        raise NotImplementedError()
        
    @abstractmethod
    @override
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
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

    @override
    def solve_action(self, state: YumiDualDeviceState, action: MixedVelocityYumiAction) -> YumiDualDeviceCommand:
        """ Convert a desired action to the required command using an IK solver,
            if necessary, and clip the commands.
            
            :param action: the action to be converted
            :returns: the required command
        """
        # solve action for joint velocities
        if action["control_space"] == MixedVelocityYumiAction.ControlSpace.JOINT_SPACE:
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
