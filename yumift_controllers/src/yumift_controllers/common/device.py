import rospy
import numpy as np

from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv
from abb_robot_msgs.msg import SystemState as SystemStateMsg
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv
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