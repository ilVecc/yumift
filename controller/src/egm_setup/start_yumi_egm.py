#!/usr/bin/env python3
import rospy
import rosservice

# this scrips calls the necessary ROS services to set the settings of yumi and start EGM.

def shutdown_hook():
    print("Shutting down ... ", end="")
    # stop EGM
    rospy.wait_for_service("/yumi/rws/sm_addin/stop_egm", timeout=5)
    rosservice.call_service("/yumi/rws/sm_addin/stop_egm", {})
    rospy.sleep(0.7)
    # stop RAPID
    rospy.wait_for_service("/yumi/rws/stop_rapid", timeout=5)
    rosservice.call_service("/yumi/rws/stop_rapid", {})
    rospy.sleep(0.7)
    print("done")
    
def main():
    # first makes sure that anything running is stopped
    print("restart RAPID ... ", end="", flush=True)
    # stop RAPID
    rospy.wait_for_service("/yumi/rws/stop_rapid")
    rosservice.call_service("/yumi/rws/stop_rapid", {})
    rospy.sleep(1.5)
    # resets and starts rapid scripts
    rospy.wait_for_service("/yumi/rws/pp_to_main")
    rosservice.call_service("/yumi/rws/pp_to_main", {})
    rospy.sleep(1.5)
    # start RAPID
    rospy.wait_for_service("/yumi/rws/start_rapid")
    rosservice.call_service("/yumi/rws/start_rapid", {})
    rospy.sleep(1.5)
    print("done", flush=True)

    # settings, see wiki for more information
    setup_uc = dict(use_filtering=True, comm_timeout=1.0)
    xyz = dict(x=0.0, y=0.0, z=0.0)
    quat = dict(q1=1.0, q2=0.0, q3=0.0, q4=0.0)
    tframe = dict(trans=xyz, rot=quat)
    # Gripper info page 21-22 in https://abb.sluzba.cz/Pages/Public/IRC5UserDocumentationRW6/en/3HC054949%20PM%20IRB%2014000%20Gripper-en.pdf
    # mass [kg], center_of_gravity [mm], inertia [kgm^2]
    total_load_R = dict(mass=0.230, cog=dict(x=8.2, y=11.7, z=52.0), aom=quat, ix=0.00021, iy=0.00024, iz=0.00009)
    total_load_L = dict(mass=0.230, cog=dict(x=8.2, y=11.7, z=52.0), aom=quat, ix=0.00021, iy=0.00024, iz=0.00009)
    tool_R = dict(robhold=True, tframe=tframe, tload=total_load_R)
    tool_L = dict(robhold=True, tframe=tframe, tload=total_load_L)
    work_obj = dict(robhold=False, ufprog=True,  ufmec="", uframe=dict(trans=xyz, rot=quat), oframe=dict(trans=xyz, rot=quat))
    correction_frame=dict(trans=xyz, rot=quat)
    sensor_frame=dict(trans=xyz, rot=quat)
    activate_R = dict(
        tool=tool_R, wobj=work_obj, correction_frame=correction_frame, sensor_frame=sensor_frame, 
        cond_min_max=0.0, lp_filter=20.0, sample_rate=4, max_speed_deviation=90.0)
    activate_L = dict(
        tool=tool_L, wobj=work_obj, correction_frame=correction_frame, sensor_frame=sensor_frame,
        cond_min_max=0.0, lp_filter=20.0, sample_rate=4, max_speed_deviation=90.0)
    run = dict(cond_time=60.0, ramp_in_time=1.0, offset=dict(trans=xyz, rot=quat), pos_corr_gain=0.0)
    stop = dict(ramp_out_time=1.0)
    settings_R = dict(allow_egm_motions=True, use_presync=False, setup_uc=setup_uc, activate=activate_R, run=run, stop=stop)
    settings_L = dict(allow_egm_motions=True, use_presync=False, setup_uc=setup_uc, activate=activate_L, run=run, stop=stop)

    # set settings for both arms
    print("sending EGM arm configuration ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/rws/sm_addin/set_egm_settings")
    rosservice.call_service("/yumi/rws/sm_addin/set_egm_settings", dict(task="T_ROB_R", settings=settings_R))
    rospy.sleep(1.5)
    rospy.wait_for_service("/yumi/rws/sm_addin/set_egm_settings")
    rosservice.call_service("/yumi/rws/sm_addin/set_egm_settings", dict(task="T_ROB_L", settings=settings_L))
    rospy.sleep(1.5)
    print("done", flush=True)

    # starts the EGM session
    print("starting EGM ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/rws/sm_addin/start_egm_joint")
    rosservice.call_service("/yumi/rws/sm_addin/start_egm_joint", {})
    rospy.sleep(1.5)
    print("done", flush=True)

    print("switching controller ... ", end="", flush=True)
    rospy.wait_for_service("/yumi/egm/controller_manager/switch_controller")
    res = rosservice.call_service("/yumi/egm/controller_manager/switch_controller",
                                  dict(start_controllers=["joint_group_velocity_controller"],
                                       stop_controllers=[""],
                                       strictness=1,
                                       start_asap=False,
                                       timeout=0.0))
    print("done\n", flush=True)
    
    if res[1].ok:
        print(f"CONTROLLER CONFIG")
        print(res[0])
    else:
        print("CONTROLLER NOT STARTED!")
        rospy.signal_shutdown("controller not started")

    rospy.spin()


if __name__ == "__main__":
    
    rospy.on_shutdown(shutdown_hook)
    
    main()





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
