#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# A simple wrench simulator adapted from the teleop_twist_keyboard
# https://github.com/ros-teleop/teleop_twist_keyboard/
#
# TODO for the future, might check out joy_teleop
# https://github.com/ros-teleop/teleop_tools/blob/master/joy_teleop/
#
# Authors:
#   * Sebastiano Fregnan

import threading

import roslib; roslib.load_manifest("yumi_controller")
import rospy

from geometry_msgs.msg import Wrench
from geometry_msgs.msg import WrenchStamped

import sys
from select import select

if sys.platform == "win32":
    import msvcrt
else:
    import termios
    import tty


WrenchMsg = Wrench

msg = """
Reading from the keyboard and Publishing to Wrench!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

dofBindings = {
        "u":(-1, 1, 0, 0, 0, 0),
        "i":( 0, 1, 0, 0, 0, 0),
        "o":( 1, 1, 0, 0, 0, 0),
        "j":(-1, 0, 0, 0, 0, 0),
        "k":( 0, 0, 0, 0, 0, 0),
        "l":( 1, 0, 0, 0, 0, 0),
        "m":(-1,-1, 0, 0, 0, 0),
        ",":( 0,-1, 0, 0, 0, 0),
        ".":( 1,-1, 0, 0, 0, 0),
        "t":( 0, 0, 1, 0, 0, 0),
        "b":( 0, 0,-1, 0, 0, 0),
        #
        "U":( 0, 0, 0,-1, 1, 0),
        "I":( 0, 0, 0, 0, 1, 0),
        "O":( 0, 0, 0, 1, 1, 0),
        "J":( 0, 0, 0,-1, 0, 0),
        "K":( 0, 0, 0, 0, 0, 0),
        "L":( 0, 0, 0, 1, 0, 0),
        "M":( 0, 0, 0,-1,-1, 0),
        ";":( 0, 0, 0, 0,-1, 0),
        ":":( 0, 0, 0, 1,-1, 0),
        "T":( 0, 0, 0, 0, 0, 1),
        "B":( 0, 0, 0, 0, 0,-1),
    }

wrenchBindings={
        "q":(1.1, 1.1),
        "z":(0.9, 0.9),
        "w":(1.1, 1.0),
        "x":(0.9, 1.0),
        "e":(1.0, 1.1),
        "c":(1.0, 0.9),
    }
class PublishThread(threading.Thread):
    def __init__(self, topic, rate):
        super().__init__()
        self.publisher = rospy.Publisher(topic, WrenchMsg, queue_size=1)
        self.fx = 0.0
        self.fy = 0.0
        self.fz = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.force = 0.0
        self.torque = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update(self, fx, fy, fz, tx, ty, tz, force, torque):
        self.condition.acquire()
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.force = force
        self.torque = torque
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        wrench_msg = WrenchMsg()

        if stamped:
            wrench = wrench_msg.wrench
            wrench_msg.header.stamp = rospy.Time.now()
            wrench_msg.header.frame_id = wrench_frame
        else:
            wrench = wrench_msg
        
        while not self.done:
            if stamped:
                wrench_msg.header.stamp = rospy.Time.now()
            
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            wrench.force.x = self.fx * self.force
            wrench.force.y = self.fy * self.force
            wrench.force.z = self.fz * self.force
            wrench.torque.x = self.tx * self.torque
            wrench.torque.y = self.ty * self.torque
            wrench.torque.z = self.tz * self.torque

            self.condition.release()

            # Publish.
            self.publisher.publish(wrench_msg)

        # Publish stop message when thread exits.
        wrench.force.x = 0
        wrench.force.y = 0
        wrench.force.z = 0
        wrench.torque.x = 0
        wrench.torque.y = 0
        wrench.torque.z = 0
        self.publisher.publish(wrench_msg)


def getKey(settings, timeout):
    if sys.platform == "win32":
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        key = sys.stdin.read(1) if rlist else ""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def saveTerminalSettings():
    if sys.platform == "win32":
        return None
    return termios.tcgetattr(sys.stdin)

def restoreTerminalSettings(old_settings):
    if sys.platform == "win32":
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def vels(speed, turn):
    return "currently:\tforce %s\ttorque %s " % (speed, turn)

if __name__=="__main__":
    settings = saveTerminalSettings()

    rospy.init_node("teleop_wrench_keyboard")

    force = rospy.get_param("~force", 0.5)
    torque = rospy.get_param("~torque", 1.0)
    force_limit = rospy.get_param("~force_limit", 1000)
    torque_limit = rospy.get_param("~torque_limit", 1000)
    repeat = rospy.get_param("~repeat_rate", 0.0)
    key_timeout = rospy.get_param("~key_timeout", 0.5)
    stamped = rospy.get_param("~stamped", True)
    wrench_frame = rospy.get_param("~frame_id", "")
    if stamped:
        WrenchMsg = WrenchStamped

    pub_thread = PublishThread("/ftsensor_l/world_tip", repeat)

    fx = 0
    fy = 0
    fz = 0
    tx = 0
    ty = 0
    tz = 0
    status = 0

    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(fx, fy, fz, tx, ty, tz, force, torque)

        print(msg)
        print(vels(force, torque))
        while(1):
            key = getKey(settings, key_timeout)
            if key in dofBindings.keys():
                fx = dofBindings[key][0]
                fy = dofBindings[key][1]
                fz = dofBindings[key][2]
                tx = dofBindings[key][3]
                ty = dofBindings[key][4]
                tz = dofBindings[key][5]
            elif key in wrenchBindings.keys():
                force = min(force_limit, force * wrenchBindings[key][0])
                torque = min(torque_limit, torque * wrenchBindings[key][1])
                if force == force_limit:
                    print("Force limit reached!")
                if torque == torque_limit:
                    print("Torque limit reached!")
                print(vels(force, torque))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating the topic if key timeout and robot already stopped.
                if key == "" and fx == 0 and fy == 0 and fz == 0 and tx == 0 and ty == 0 and tz == 0:
                    continue
                fx = 0
                fy = 0
                fz = 0
                tx = 0
                ty = 0
                tz = 0
                if (key == "\x03"):
                    break

            pub_thread.update(fx, fy, fz, tx, ty, tz, force, torque)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()
        restoreTerminalSettings(settings)