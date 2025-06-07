#!/usr/bin/env python3

import rospy, rospkg, rosservice

import yaml
from pathlib import Path

# this scrips calls the necessary ROS services to set the settings of yumi and start EGM.

def shutdown_hook():
    print("Shutting down ... ", end="")
    # stop EGM
    rospy.wait_for_service("/yumi/rws/sm_addin/stop_egm", timeout=5)
    rosservice.call_service("/yumi/rws/sm_addin/stop_egm", {})
    rospy.sleep(0.5)
    # stop RAPID
    rospy.wait_for_service("/yumi/rws/stop_rapid", timeout=5)
    rosservice.call_service("/yumi/rws/stop_rapid", {})
    rospy.sleep(0.5)
    print("done", flush=True)


def main():
    # first makes sure that anything running is stopped
    print("restart RAPID ... ", end="", flush=True)
    # stop RAPID
    rospy.wait_for_service("/yumi/rws/stop_rapid")
    rosservice.call_service("/yumi/rws/stop_rapid", {})
    rospy.sleep(0.5)
    # resets and starts rapid scripts
    rospy.wait_for_service("/yumi/rws/pp_to_main")
    rosservice.call_service("/yumi/rws/pp_to_main", {})
    rospy.sleep(0.5)
    # start RAPID
    rospy.wait_for_service("/yumi/rws/start_rapid")
    rosservice.call_service("/yumi/rws/start_rapid", {})
    rospy.sleep(0.5)
    print("done", flush=True)

    # load EGM settings file
    pkg_name = "yumi_controller"
    pkg_path = Path(rospkg.RosPack().get_path(pkg_name)) / "config" / "egm_settings.yaml"
    with open(str(pkg_path)) as f:
        settings = yaml.safe_load(f)
    
    # set settings for both arms
    print("sending EGM arm configuration ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/rws/sm_addin/set_egm_settings")
    rosservice.call_service("/yumi/rws/sm_addin/set_egm_settings", settings["task_settings_R"])
    rospy.sleep(0.5)
    rospy.wait_for_service("/yumi/rws/sm_addin/set_egm_settings")
    rosservice.call_service("/yumi/rws/sm_addin/set_egm_settings", settings["task_settings_L"])
    rospy.sleep(0.5)
    print("done", flush=True)

    # starts the EGM session
    print("starting EGM ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/rws/sm_addin/start_egm_joint")
    rosservice.call_service("/yumi/rws/sm_addin/start_egm_joint", {})
    rospy.sleep(0.5)
    print("done", flush=True)

    print("switching controller ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/egm/controller_manager/switch_controller")
    res = rosservice.call_service("/yumi/egm/controller_manager/switch_controller",
                                  dict(start_controllers=["joint_group_velocity_controller"],
                                       stop_controllers=[""],
                                       strictness=1,
                                       start_asap=False,
                                       timeout=0.0))
    print("done", flush=True)
    print()
    
    if res[1].ok:
        print(f"CONTROLLER CONFIG")
        print(res[0])
    else:
        print("CONTROLLER NOT STARTED!")
        print("(are revolute counters up-to-date?)")
        print("(is the robot in AUTO mode?)")
        print("(is the firewall turned off?)")
        rospy.signal_shutdown("controller not started")

    rospy.spin()


if __name__ == "__main__":
    
    rospy.on_shutdown(shutdown_hook)
    
    main()





# TODO this should be taken care of

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
