#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np

import rospy
from geometry_msgs.msg import WrenchStamped, Vector3

from dynamics.filters import LPFilterTustin


def main() -> None:
    rospy.init_node("lpfilter", anonymous=True)
    
    lp = LPFilterTustin(6, 1, 0.02)
    
    pub_f = rospy.Publisher("/ftsensor_l/original", Vector3, queue_size=1)
    pub_lp = rospy.Publisher("/ftsensor_l/filtered", Vector3, queue_size=1)
    
    def callback(data: WrenchStamped):
        f = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        lp_f = lp.compute(f)
        pub_f.publish(Vector3(*f))
        pub_lp.publish(Vector3(*lp_f))
    
    rospy.Subscriber("/ftsensor_l/world", WrenchStamped, callback)
    
    rospy.spin()

if __name__ == "__main__":
    main()
