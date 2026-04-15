#!/usr/bin/env python3
import rospy, rospkg, rosservice

import argparse
import yaml
from pathlib import Path

# this scrips calls the necessary ROS services to set the settings of yumi and start EGM.
nickname = ""


class status():
    def __init__(self, msg):
        self.message = msg
    
    def __enter__(self):
        print(self.message + " ... ", end="", flush=True) 
    
    def __exit__(self, *args):
        print("done", flush=True)

def wait_and_call(service, args={}, timeout=5, sleep=0.5):
    rospy.wait_for_service(service, timeout)
    ret = rosservice.call_service(service, args)
    rospy.sleep(sleep)
    return ret

def shutdown_hook():
    global nickname
    
    with status("Shutting down"):
        wait_and_call(f"{nickname}/rws/sm_addin/stop_egm")
        wait_and_call(f"{nickname}/rws/stop_rapid")

def main():
    global nickname
    
    if nickname != "" and nickname[0] != "/":
        nickname = "/" + nickname
    
    # from abb_robot_msgs.msg import SystemState
    # done = False
    # while not done:
    #     state : SystemState = rospy.wait_for_message("/yumi/rws/system_states", SystemState, timeout=5.0)
    #     if not state.auto_mode:
    #         print("Robot is in MANUAL mode. Switch to AUTO from the TeachPendant.")
    #         input("Hit [ENTER] when done...")
    
    # start the motors
    with status("start MOTORS"):
        wait_and_call(f"{nickname}/rws/set_motors_on")
    
    # makes sure that anything running is stopped
    with status("restart RAPID"):
        wait_and_call(f"{nickname}/rws/stop_rapid")
        wait_and_call(f"{nickname}/rws/pp_to_main")  # execution back to beginning
        wait_and_call(f"{nickname}/rws/start_rapid")
    
    # load EGM settings file
    pkg_path = Path(rospkg.RosPack().get_path("yumift_common")) / "config" / "egm_settings.yaml"
    with open(str(pkg_path)) as f:
        settings = yaml.safe_load(f)
    
    # send settings for each arm
    with status("sending EGM arm configuration"):
        wait_and_call(f"{nickname}/rws/sm_addin/set_egm_settings", settings["task_settings_L"])
        wait_and_call(f"{nickname}/rws/sm_addin/set_egm_settings", settings["task_settings_R"])
    
    # starts the EGM session
    with status("starting EGM"):
        wait_and_call(f"{nickname}/rws/sm_addin/start_egm_joint")
    
    # switch controller
    with status("switching controller"):
        res = wait_and_call(
            f"{nickname}/egm/controller_manager/switch_controller",
            dict(start_controllers=["joint_group_velocity_controller"],
                 stop_controllers=[""],
                 strictness=1,
                 start_asap=False,
                 timeout=0.0),
            sleep=0)
    print()
    
    if res[1].ok:
        print("CONTROLLER CONFIG")
        print(res[0])
    else:
        print("CONTROLLER NOT STARTED!")
        print("checklist:")
        print(" - are revolute counters up-to-date?)")
        print(" - is the robot in AUTO mode?)")
        print(" - is the firewall turned off?)")
        print(" - is your IP the one expected by YuMi in \"Transmission Protocol\"?)")
        rospy.signal_shutdown("Cannot start EGM communication")

    rospy.spin()


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser("start_egm")
    # parser.add_argument("nickname", type=str, default="yumi", help="Name of the robot")
    # args = parser.parse_args()
    # nickname = args.nickname
    
    nickname = "yumi"
    
    # rospy.init_node
    
    # TODO this is not a node tough
    rospy.on_shutdown(shutdown_hook)
    
    main()



# rosservice call /yumi/rws/set_motors_on "{}"
# /yumi/rws/sm_addin/start_egm_pose


# TODO this should be taken care of

from abb_robot_msgs.msg import SystemState

# # RAPID execution states:
# uint8 EXECUTION_STATE_UNKNOWN       = 1
# uint8 EXECUTION_STATE_READY         = 2
# uint8 EXECUTION_STATE_STOPPED       = 3
# uint8 EXECUTION_STATE_STARTED       = 4
# uint8 EXECUTION_STATE_UNINITIALIZED = 5

# /yumi/rws/system_states
#
# motors_on: True
# auto_mode: True
# rapid_running: True
# rapid_tasks:
#   -
#     name: "T_ROB_R"
#     activated: True
#     execution_state: 4
#     motion_task: True
#   -
#     name: "T_ROB_L"
#     activated: True
#     execution_state: 4
#     motion_task: True
#   -
#     name: "T_WATCHDOG"
#     activated: True
#     execution_state: 4
#     motion_task: False
# mechanical_units:
#   -
#     name: "ROB_L"
#     activated: True
#   -
#     name: "ROB_R"
#     activated: True

from abb_egm_msgs.msg import EGMState

# # EGM client states:
# uint8 EGM_UNDEFINED = 1
# uint8 EGM_ERROR     = 2
# uint8 EGM_STOPPED   = 3
# uint8 EGM_RUNNING   = 4

# # Motor states:
# uint8 MOTORS_UNDEFINED = 1
# uint8 MOTORS_ON        = 2
# uint8 MOTORS_OFF       = 3

# # RAPID states:
# uint8 RAPID_UNDEFINED = 1
# uint8 RAPID_STOPPED   = 2
# uint8 RAPID_RUNNING   = 3

# WORKING 
# 
# /yumi/egm/egm_states
#
# egm_channels:
#   -
#     name: "channel_1"
#     active: True
#     egm_client_state: 4           # AUTO + RUNNING
#     motor_state: 2                # ON
#     rapid_execution_state: 3      # RUNNING
#

# NOT WORKING 
# 
# /yumi/egm/egm_states
#
# egm_channels:
#   -
#     name: "channel_1"
#     active: False
#     egm_client_state: 3           # AUTO + NOT RUNNING
#     motor_state: 2                # ON
#     rapid_execution_state: 2      # NOT RUNNING
#
#
# egm_channels:
#   -
#     name: "channel_1"
#     active: True
#     egm_convergence_met: False
#     egm_client_state: 3           # AUTO + NOT RUNNING
#     motor_state: 3                # OFF
#     rapid_execution_state: 2      # NOT RUNNING

from abb_rapid_sm_addin_msgs.msg import RuntimeState

# # Possible "StateMachine Add-In" RAPID states:
# uint8 SM_STATE_UNKNOWN           = 1
# uint8 SM_STATE_IDLE              = 2
# uint8 SM_STATE_INITIALIZE        = 3
# uint8 SM_STATE_RUN_RAPID_ROUTINE = 4
# uint8 SM_STATE_RUN_EGM_ROUTINE   = 5

# # Possible "StateMachine Add-In" RAPID EGM actions:
# uint8 EGM_ACTION_UNKNOWN      = 1
# uint8 EGM_ACTION_NONE         = 2
# uint8 EGM_ACTION_RUN_JOINT    = 3
# uint8 EGM_ACTION_RUN_POSE     = 4
# uint8 EGM_ACTION_STOP         = 5
# uint8 EGM_ACTION_START_STREAM = 6
# uint8 EGM_ACTION_STOP_STREAM  = 7

# /yumi/rws/sm_addin/runtime_states
#
# state_machines:
#   -
#     rapid_task: "T_ROB_R"
#     sm_state: 2
#     egm_action: 2
#   -
#     rapid_task: "T_ROB_L"
#     sm_state: 2
#     egm_action: 2