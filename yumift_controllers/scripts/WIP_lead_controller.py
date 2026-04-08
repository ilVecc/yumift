#!/usr/bin/env python3
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
        
        # TODO maybe set M=??, D=??, K=0
        self.control_law = YumiDualAdmittanceControlLaw(load_config("gains.yaml")["ADM"], discretization="forward")
        
    def reset(self, state: YumiDualDeviceState):
        self.control_law.clear()
        rospy.loginfo("Controller reset")
    
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        try:
            dt = (rospy.Time.now() - state.time).to_sec()
            vel_1, vel_2 = self.control_law.update_and_compute(current_state=state, desired_state=state, timestep=dt)
            
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


def main():
    # starting ROS node
    rospy.init_node("lead_controller", anonymous=False) 
    
    yumi = YumiDevice()
    yumi_controller = YumiLeadController(yumi)
    
    rospy.on_shutdown(yumi_controller.stop)
    
    yumi_controller.ready()
    yumi_controller.start()  # locking


if __name__ == "__main__":
    main()
