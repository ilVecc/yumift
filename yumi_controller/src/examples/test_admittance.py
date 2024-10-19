#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import dynamics.quat_utils as quat_utils

import rospy
from geometry_msgs.msg import WrenchStamped, Vector3

from dynamics.filters import AdmittanceForce, AdmittanceTorque
from dynamics.utils import normalize


def main() -> None:
    rospy.init_node("admittance", anonymous=True)
    
    sensor = "ftsensor_l"
    
    adm_f = AdmittanceForce(1.5, 25, None, 0.001, "tustin")  # this h is fake and will be updated at each timestep
    adm_m = AdmittanceTorque(0.08, 0.6, None, 0.001, "tustin")
    
    pub_f = rospy.Publisher(f"/{sensor}/f", Vector3, queue_size=1)
    pub_m = rospy.Publisher(f"/{sensor}/t", Vector3, queue_size=1)
    pub_ep = rospy.Publisher(f"/{sensor}/ep", Vector3, queue_size=1)
    pub_ev = rospy.Publisher(f"/{sensor}/ev", Vector3, queue_size=1)
    pub_er = rospy.Publisher(f"/{sensor}/er", Vector3, queue_size=1)
    pub_ew = rospy.Publisher(f"/{sensor}/ew", Vector3, queue_size=1)
    
    ts = rospy.Time.now()
    
    def callback(data: WrenchStamped):
        
        # calculate dt
        nonlocal ts
        ts, h = data.header.stamp, (data.header.stamp - ts).to_sec()
        if h == 0:
            h = None
        
        f = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        m = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
        ep, dep = adm_f.compute(f, h)
        er, der = adm_m.compute(m, h)
        
        pub_f.publish(Vector3(*(f)))
        pub_ep.publish(Vector3(*ep))
        pub_ev.publish(Vector3(*dep))
        
        k, a = normalize(quat_utils.log(er), return_norm=True)
        pub_m.publish(Vector3(*(m)))
        pub_er.publish(Vector3(*(a*k)))
        pub_ew.publish(Vector3(*der))
    
    rospy.Subscriber(f"/{sensor}/world", WrenchStamped, callback)
    
    rospy.spin()

if __name__ == "__main__":
    main()
