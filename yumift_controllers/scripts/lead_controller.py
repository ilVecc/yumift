#!/usr/bin/env python3
from typing_extensions import override

import rospy

from yumift_controllers.common.device import YumiDualDeviceState, YumiDevice
from yumift_controllers.common.controller_base import YumiDualController, MixedVelocityYumiAction
from yumift_controllers.common.control_laws import YumiDualAdmittanceControlLaw
from yumift_controllers.ik.algorithms import PINVIKAlgorithm
from yumift_controllers.misc.utils import load_config


# TODO no need for this to be dual
class YumiLeadController(YumiDualController):
    """ Class for running lead-through control using an instance of `YumiDualController`.
    """
    def __init__(self, device : YumiDevice):
        super().__init__(yumi_device=device, ikalgorithms=[PINVIKAlgorithm()])
        self.control_law = YumiDualAdmittanceControlLaw(load_config("gains_lead.yaml"), discretization="forward")
    
    @override
    def reset(self, state: YumiDualDeviceState):
        self.control_law.clear()
        rospy.loginfo("Controller reset")
    
    @override
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        try:
            dt = (rospy.Time.now() - state.time).to_sec()
            vel_1, vel_2 = self.control_law.update_and_compute(state, state, dt)
            
            action = MixedVelocityYumiAction()
            action.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
            action.timestep(dt)
            action.velocity_right(vel_1)
            action.velocity_left(vel_2)
        
        except Exception as ex:
            rospy.logfatal(f"Could not compute action (exception: {ex})")
            rospy.logfatal("Manually invoking fallback policy")
            action = self.fallback(state)
        
        return action


if __name__ == "__main__":
    # starting ROS node
    rospy.init_node("lead_controller", anonymous=False) 
    
    device = YumiDevice()
    controller = YumiLeadController(device)
    
    rospy.on_shutdown(controller.stop)
    
    controller.ready()
    controller.start()  # locking
