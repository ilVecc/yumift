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
    
    # first makes sure that anything running is stopped
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
    
    # TODO this is not a node tough
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
