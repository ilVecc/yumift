from typing_extensions import override

import rospy
import numpy as np

from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv
from abb_robot_msgs.msg import SystemState as SystemStateMsg, ServiceResponses as ServiceResponsesMsg
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv, TriggerWithResultCodeResponse
from std_msgs.msg import Float64MultiArray as Float64MultiArrayMsg
from yumift_msgs.msg import RobotState as RobotStateMsg

from yumift_common.constants import YumiRobotConstants
from yumift_common.robot_state import YumiCoordinatedRobotState
from yumift_controllers.misc.utils import RobotStateMsg_to_YumiCoordinatedRobotState

from dynamicals.common.devices import AbstractDevice, AbstractDeviceCommand, AbstractDeviceState


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


# UTILS
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

# TODO why not dual?
# TODO remove "/yumi" from everywhere
class YumiDevice(AbstractDevice[YumiDualDeviceState, YumiDualDeviceCommand]):

    def __init__(self, coordinated_balance : float = 0.5):
        """ Initialize a device for I/O interaction with Yumi.

            :param coordinated_balance: asymmetry parameter when working in 
                                        coordinated mode
        """
        super().__init__()
        # yumi state subscriber
        self._cache_state: YumiDualDeviceState
        self._device_ready = False
        rospy.Subscriber("/yumi/unified/robot_state_coordinated", RobotStateMsg, self._callback_received_state, queue_size=1)
        # ensure to start the controller with a real robot state 
        # (no wait means default state (all zeros), very bad)
        rospy.wait_for_message("/yumi/unified/robot_state_coordinated", RobotStateMsg)

        # command publishers
        self._pub_vel = rospy.Publisher("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArrayMsg, queue_size=1)
        self._pub_grip = YumiGrippersCommand()

        # EGM error handler and status updater (updates `self._device_ready`)
        self._start_rapid = rospy.ServiceProxy("/yumi/rws/start_rapid", TriggerWithResultCodeSrv)
        rospy.Subscriber("/yumi/rws/system_states", SystemStateMsg, self._callback_update_status, queue_size=1)
        rospy.wait_for_message("/yumi/rws/system_states", SystemStateMsg)

    def _callback_update_status(self, data: SystemStateMsg):
        # update status and set "status changed" flag
        self._device_ready = data.auto_mode and data.motors_on and data.rapid_running
        # TODO handle other flags in the message
        # /yumi/egm/...
        # data.rapid_tasks
        # data.mechanical_units
        
    def _callback_received_state(self, data: RobotStateMsg):
        # TODO this is broken, type mismatch
        self._cache_state = RobotStateMsg_to_YumiCoordinatedRobotState(data)
        self._cache_state.time = rospy.Time.now()
    
    @override
    def reset(self) -> bool:
        ret : TriggerWithResultCodeResponse = self._start_rapid.call()
        return ret.result_code == ServiceResponsesMsg.RC_SUCCESS
    
    @override
    def is_ready(self) -> bool:
        return self._device_ready

    @override
    def read(self) -> YumiDualDeviceState:
        """ Stores the constantly updating state of Yumi inside the variables 
            actually used by the controller, effectively updating the state 
            in the controller. The data coming from Yumi might be old (because 
            of a disconnection), thus the RWS status is used as Yumi status.
        """
        return self._cache_state

    @override
    def send(self, command: YumiDualDeviceCommand):
        # flip the array to [left, right] as required by ros_control velocity controller
        vel = command._dq_target.tolist()
        msg = Float64MultiArrayMsg(data=vel[7:14]+vel[0:7])
        self._pub_vel.publish(msg)
        # avoid sendind commands all the time to optimize bandwidth
        if (command._grip_r is not None) or (command._grip_l is not None):
            self._pub_grip.send_position_cmd(command._grip_r, command._grip_l)