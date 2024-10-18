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
    
    adm_f = AdmittanceForce(1.5, 100, 25, 0.001)
    adm_m = AdmittanceTorque(0.08, 10, 0.6, 0.001)
    
    pub_f = rospy.Publisher("/ftsensor_l/f", Vector3, queue_size=1)
    pub_m = rospy.Publisher("/ftsensor_l/m", Vector3, queue_size=1)
    pub_err_pos = rospy.Publisher("/ftsensor_l/err_pos", Vector3, queue_size=1)
    pub_err_vel = rospy.Publisher("/ftsensor_l/err_vel", Vector3, queue_size=1)
    pub_err_rot = rospy.Publisher("/ftsensor_l/err_rot", Vector3, queue_size=1)
    pub_err_wel = rospy.Publisher("/ftsensor_l/err_wel", Vector3, queue_size=1)
    
    ts = None
    
    def callback(data: WrenchStamped):
        nonlocal ts
        if ts is None:
            ts = data.header.stamp
            return
        ts, h = data.header.stamp, (data.header.stamp - ts).to_sec()
        if h == 0:
            return
        f = np.array([data.wrench.force.x, data.wrench.force.y, data.wrench.force.z])
        m = np.array([data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z])
        ep, dep = adm_f.compute(f, h)
        er, der = adm_m.compute(m, h)
        
        pub_f.publish(Vector3(*(f/100)))
        pub_err_pos.publish(Vector3(*ep))
        pub_err_vel.publish(Vector3(*dep))
        
        k, a = normalize(quat_utils.log(er), return_norm=True)
        pub_m.publish(Vector3(*(m)))
        pub_err_rot.publish(Vector3(*(a*k)))
        pub_err_wel.publish(Vector3(*der))
    
    rospy.Subscriber("/ftsensor_l/world", WrenchStamped, callback)
    
    rospy.spin()

if __name__ == "__main__":
    main()
