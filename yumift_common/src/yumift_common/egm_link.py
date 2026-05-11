#!/usr/bin/env python3
import rospy, rospkg, rosparam, genpy
from typing import List

import argparse
import yaml
from pathlib import Path

from abb_robot_msgs.srv import TriggerWithResultCode, GetIOSignal
from abb_rapid_sm_addin_msgs.srv import SetEGMSettings
from controller_manager_msgs.srv import SwitchController

from abb_robot_msgs.msg import SystemState, ServiceResponses
from abb_rapid_sm_addin_msgs.msg import RuntimeState, StateMachineState
from abb_egm_msgs.msg import EGMState
import message_filters


# this script calls the necessary ROS services to set the settings of yumi and start EGM.

def call_abb_service(
    service : rospy.ServiceProxy, error_message : str, message_args : dict = {}, 
    extra_ok_codes : List[int] = [], print_cause : bool = True
):
    message = service.request_class()
    genpy.message.fill_message_args(message, message_args)
    res : TriggerWithResultCode._response_class = service.call(message)
    if res.result_code not in [ServiceResponses.RC_SUCCESS] + extra_ok_codes:
        rospy.logerr_throttle(10.0, f"{error_message}{f'. Cause: {res.message}' if print_cause else ''}")
        raise rospy.ROSException(f"{error_message}. Cause: {res.message}")
    rospy.sleep(0.5)


class EGMLink():
    
    def __init__(self):
        self.srv_emergency_signal = rospy.ServiceProxy(f"rws/get_io_signal", GetIOSignal)
        self.srv_set_motors_on = rospy.ServiceProxy(f"rws/set_motors_on", TriggerWithResultCode)
        self.srv_stop_rapid = rospy.ServiceProxy(f"rws/stop_rapid", TriggerWithResultCode)
        self.srv_pp_to_main = rospy.ServiceProxy(f"rws/pp_to_main", TriggerWithResultCode)
        self.srv_start_rapid = rospy.ServiceProxy(f"rws/start_rapid", TriggerWithResultCode)
        self.srv_sm_set_egm_settings = rospy.ServiceProxy(f"rws/sm_addin/set_egm_settings", SetEGMSettings)
        self.srv_sm_start_egm_joint = rospy.ServiceProxy(f"rws/sm_addin/start_egm_joint", TriggerWithResultCode)
        self.srv_switch_controller = rospy.ServiceProxy(f"egm/controller_manager/switch_controller", SwitchController)
        # # self.srv_sm_start_egm_pose = rospy.ServiceProxy(f"rws/sm_addin/start_egm_pose", TriggerWithResultCode)
        self.srv_sm_stop_egm = rospy.ServiceProxy(f"rws/sm_addin/stop_egm", TriggerWithResultCode)
        
        # be sure everything is ready
        self.srv_emergency_signal.wait_for_service()
        self.srv_set_motors_on.wait_for_service()
        self.srv_stop_rapid.wait_for_service()
        self.srv_pp_to_main.wait_for_service()
        self.srv_start_rapid.wait_for_service()
        self.srv_sm_set_egm_settings.wait_for_service()
        self.srv_sm_start_egm_joint.wait_for_service()
        self.srv_switch_controller.wait_for_service()
        # # self.srv_sm_start_egm_pose.wait_for_service()
        self.srv_sm_stop_egm.wait_for_service()
        
        # stop rapid when shutting down
        rospy.on_shutdown(self._shutdown_hook)
        
        # load EGM settings file
        settings = rospy.get_param("~egm_settings")
        self.task_settings_L = settings["task_settings_L"]
        self.task_settings_R = settings["task_settings_R"]
        
        # start the "keep-alive" callback: if anything is not ok, then kill the node
        # if the node was launched with `respawn=true` then the link is established again 
        self.successfully_started = False
        sub_rws_states = message_filters.Subscriber("/yumi/rws/system_states", SystemState, queue_size=1)
        sub_rws_sm_addin_states = message_filters.Subscriber("/yumi/rws/sm_addin/runtime_states", RuntimeState, queue_size=1)
        # sub_egm_states = message_filters.Subscriber("/yumi/egm/system_states", EGMState, queue_size=1)
        # sub_overall_state = message_filters.TimeSynchronizer([sub_rws_states, sub_rws_sm_addin_states, sub_egm_states], queue_size=1)
        sub_overall_state = message_filters.TimeSynchronizer([sub_rws_states, sub_rws_sm_addin_states], queue_size=1)
        sub_overall_state.registerCallback(self._keepalive_link_callback)
    
    def _shutdown_hook(self):
        try:
            call_abb_service(self.srv_sm_stop_egm, "Could not stop EGM")
            call_abb_service(self.srv_stop_rapid, "Could not stop RAPID")
        except rospy.ROSException as ex:
            rospy.logwarn(f"Could not close RAPID/EGM cleanly. Cause: {ex.__cause__}")
        rospy.loginfo("Communication closed")
    
    def _keepalive_link_callback(self, msg1: SystemState, msg2: RuntimeState): #, msg3: EGMState):
        state_machines_ok = not any([sm.sm_state != StateMachineState.SM_STATE_RUN_EGM_ROUTINE for sm in msg2.state_machines])
        
        if not self.successfully_started:
            rospy.loginfo("Initializing EGM link")
            try:
                self.refresh_link(msg1.auto_mode, msg1.motors_on, msg1.rapid_running, state_machines_ok)
                self.successfully_started = True
                rospy.loginfo("Successful EGM link")
            except rospy.ROSException as ex:
                rospy.loginfo(f"Could not establish EGM link. Cause: {ex}")
                rospy.signal_shutdown("Could not establish EGM link")
            return
        
        if not (msg1.auto_mode and msg1.motors_on and msg1.rapid_running and state_machines_ok):
            # if respawn=true for this node, then the link will automatically be restarted
            rospy.signal_shutdown("Robot status changed. EGM link broken.")
            
    def refresh_link(self, auto_mode: bool, motors_on: bool, rapid_running: bool, state_machines_ok: bool) -> bool:
        ###############################
        # PERFORM RAPID INITIALIZATION
        ###############################
        
        # required_mech_units = ["ROB_L", "ROB_R"]
        # for unit in msg1.mechanical_units:
        #     if unit.activated:
        #         required_mech_units.remove(unit)
        # if required_mech_units is not []:
        #     rospy.logerr_throttle(10.0, f"Some mechanical units are not activated. Missing: {required_mech_units}")
        #     return
        
        if not auto_mode:
            rospy.logerr_throttle(10.0, "Robot is in MANUAL mode. Manually switch to AUTO from the TeachPendant.")
            raise rospy.ROSException("Robot is in MANUAL mode.")
        
        if not motors_on:
            # res = self.srv_emergency_signal.call(signal="ES1")
            # res = self.srv_emergency_signal.call(signal="ES2")
            res : TriggerWithResultCode._response_class = self.srv_set_motors_on.call()
            if res.result_code in [0]:
                rospy.logerr_throttle(10.0, 
                    "Emergency stop state can only be resumed from the TeachPendant, " +
                    "by manually acknowledging errors and tapping on the MOTORS ON button")
                raise rospy.ROSException("Robot must be manually recovered after Emergency Stop.")
            rospy.sleep(0.5)
            
            rospy.loginfo("Motors started")
        
        if not rapid_running:
            call_abb_service(self.srv_stop_rapid, "Could not stop RAPID", {}, [ServiceResponses.RC_RAPID_NOT_RUNNING])
            call_abb_service(self.srv_pp_to_main, "Could not move PP to main in RAPID")
            call_abb_service(self.srv_start_rapid, "Could not start RAPID")
            
            rospy.loginfo("RAPID restarted")
        
        # required_rapid_tasks = ["T_ROB_R", "T_ROB_L", "T_WATCHDOG"]
        # for task in msg1.rapid_tasks:
        #     if task.activated:
        #         required_rapid_tasks.remove(task.name)
        # if required_rapid_tasks is not []:
        #     rospy.logerr_throttle(10.0, f"Some RAPID tasks are not activated. Missing: {required_rapid_tasks}")
        #     return
        
        ##################################################
        # PERFORM StateMachine Add-In STATE SWITCH TO EGM
        ##################################################
        
        # from abb_egm_msgs.msg import EGMState
        # /yumi/egm/egm_states
        #
        # # EGM client states:
        # uint8 EGM_RUNNING   = 4
        # # Motor states:
        # uint8 MOTORS_ON     = 2
        # # RAPID states:
        # uint8 RAPID_RUNNING = 3

        if not state_machines_ok:
            call_abb_service(self.srv_sm_set_egm_settings, "Could not send EGM settings for ROB_L", self.task_settings_L)
            call_abb_service(self.srv_sm_set_egm_settings, "Could not send EGM settings for ROB_R", self.task_settings_R)
            
            rospy.loginfo("EGM configuration sent")

            call_abb_service(self.srv_sm_start_egm_joint, "Could not start EGM Joint Mode")
            
            rospy.loginfo("EGM session started")
        
            ##############################################
            # START ros_control CONTROLLER RELYING ON EGM
            ##############################################
            
            # switch controller
            res = self.srv_switch_controller.call(
                start_controllers=["joint_group_velocity_controller"],
                stop_controllers=[""],
                strictness=1,
                start_asap=False,
                timeout=0.0)
            
            if not res.ok:
                rospy.logerr("CONTROLLER NOT SWITCHED!")
                rospy.logerr(" - is the firewall turned off?)")
                rospy.logerr(" - is the robot in AUTO mode?)")
                rospy.logerr(" - has the robot just recovered from an Emergency Stop?)")
                rospy.logerr(" - are revolute counters up-to-date?)")
                rospy.logerr(" - is your IP the one expected by YuMi in \"Transmission Protocol\"?)")
                raise rospy.ROSException("Could not switch controller")
            
            rospy.loginfo("Controller configuration switched")


if __name__ == "__main__":
    
    rospy.init_node("resilient_egm_link", anonymous=False)
    
    link = EGMLink()

    rospy.spin()
