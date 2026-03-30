#!/usr/bin/env python3
import argparse

import rospy

from yumift_controllers.misc.utils import load_config
from yumift_controllers.impl.trajectory_controller import YumiTrajectoryController
from yumift_controllers.common.control_laws import (
    YumiIndividualCartesianVelocityControlLaw, 
    YumiDualCartesianVelocityControlLaw, 
    YumiDualWrenchFeedbackControlLaw, 
    YumiDualAdmittanceControlLaw,
)


class SimpleTrajectoryController(YumiTrajectoryController):
    def __init__(self, gains):
        super().__init__("/trajectory", YumiIndividualCartesianVelocityControlLaw(gains))

class DualTrajectoryController(YumiTrajectoryController):
    def __init__(self, gains):
        super().__init__("/trajectory", YumiDualCartesianVelocityControlLaw(gains))
        
class WrenchedTrajectoryController(YumiTrajectoryController):
    def __init__(self, gains):
        super().__init__("/trajectory", YumiDualWrenchFeedbackControlLaw(gains))

class CompliantTrajectoryController(YumiTrajectoryController):
    def __init__(self, gains):
        super().__init__("/trajectory", YumiDualAdmittanceControlLaw(gains, discretization="forward"))


def main():
    
    parser = argparse.ArgumentParser("Various trajectory controllers")
    parser.add_argument("type", nargs="?", choices=["simple", "dual", "wrenched", "compliant"], default="dual", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("trajectory_controllers", anonymous=False)
    
    if args.type == "simple":
        yumi_controller = SimpleTrajectoryController(load_config("gains.yaml")["CLIK"])
    elif args.type == "dual":
        yumi_controller = DualTrajectoryController(load_config("gains.yaml")["CLIK"])
    elif args.type == "wrenched":
        yumi_controller = WrenchedTrajectoryController(load_config("gains.yaml")["DFF"])
    elif args.type == "compliant":
        yumi_controller = CompliantTrajectoryController(load_config("gains.yaml")["ADM"])
    else:
        raise AttributeError(f"no such option '{parser.type}'")
    
    rospy.on_shutdown(yumi_controller.stop)
    rospy.sleep(3)
    
    yumi_controller.ready()
    yumi_controller.start()  # locking


if __name__ == "__main__":
    main()
