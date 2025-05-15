# Wrenched YuMi control
Modular controllers for the ABB Dual-Arm YuMi robot over ROS Melodic. 

## Table of Contents
* [General Architecture](#general-architecture)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)


## General Architecture
This package contains interfaces and control algorithms for the ABB Dual-Armed YuMi. 
The core functionality includes Inverse Kinematics (IK) for YuMi in both individual and coordinated manipulation frames of the arms. 

Controllers create joint velocity commands, which are then sent to a `ros_control` velocity controller talking with the `hardware_interface` of YuMi. 
The idea is that you create a controller class that inherits the `YumiDualController` class and that you can then build functionally on top of it, see `yumi_controller/src/examples/dummy_controller.py`. There are also a trajectory following controllers implemented using the same structure, see `yumi_controller/src/controllers/trajectory_controllers.py`. This controller is mainly made to operate with slow motions as no dynamical effects have been included in the control loop. The IK problem is solved with Hierarchical Quadratic Programing (HQP) implemented using the `quadprog` solver. The control objectives are solved together with feasibility objectives (see documentation). The robot is visualized in `rviz` in real time and a very simple kinematic simulator is included when working with real hardware is not possible. 


## Dependencies
This package expects Ubuntu 18.04 and uses ROS Melodic Desktop with Python3, so ROS and Catkin must be reconfigured to use Python3 instead of Python2.
If you plan to use WSL2 on an Hyper-V enable device, a good setup would be to set mirrored networking by adding 
```
[wsl2]
networkingMode=mirrored
firewall=false
```
in the `.wslconfig` file found in your user folder and to disable Windows Firewall (even better if you set specific rules without disabling it). Then run `wsl --shutdown` in a PowerShell terminal.

* for ROS Melodic Desktop, simply follow the [official installation guide](https://wiki.ros.org/melodic/Installation/Ubuntu)

* this controller relies on `joint_group_velocity_controller` installed via
```
sudo apt install ros-melodic-ros-controllers
```

* for ROS packages and `catkin-tools` in Python3
```
sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy python3-defusedxml python3-catkin-tools
```

* for Orokos KDL
```
sudo apt-get install libeigen3-dev libcppunit-dev
```

* Python3 packages, can be installed with
``` 
sudo apt install python3-pip
python3 -m pip install numpy numpy-quaternion scipy quadprog
python3 -m pip uninstall em
python3 -m pip install empy==3.3.4
```

* to build the workspace, cmake 3.12 is needed, so upgrade it following [this guide](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu)


## Installation
Create a folder for the catkin workspace
```
mkdir -p ~/yumift_ws/src && cd ~/yumift_ws/src
```
then clone [`abb_robot_driver`](https://github.com/ros-industrial/abb_robot_driver) and run the required commands (possibly outdated, taken from the repo's README)
```
git clone https://github.com/ros-industrial/abb_robot_driver.git
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xAB17C654  # might fail, if so ignore it
sudo apt update; sudo apt install python3-vcstool
vcs import . --input https://github.com/ros-industrial/abb_robot_driver/raw/master/pkgs.repos 
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro melodic
```
__IF__ `rosdep` doesn't work, you need to manually install the driver interfaces with
```
git clone https://github.com/ros-industrial/abb_robot_driver_interfaces.git
```

Now clone [`orocos_kinematics_dynamics`](https://github.com/orocos/orocos_kinematics_dynamics) package (for Python3)
```
git clone https://github.com/orocos/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics/
git submodule update --init
```
and downgrade its `pybind` version to `v2.9` using
```
cd python_orocos_kdl/pybind11
git checkout "v2.9"
cd ~/yumift_ws/src
```

Now clone [`geometry2`](https://github.com/ros/geometry2) package (for Python3)
```
git clone -b melodic-devel https://github.com/ros/geometry2
```
and finally clone this respository
```
git clone https://github.com/ilVecc/yumift.git
```

Now build `yumift_ws` workspace
``` 
source ~/yumift_ws
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
``` 
and add source to `.bashrc` for easy startup
``` 
echo "source ~/yumift_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
``` 

Finally, to `rosrun` files in the workspace, you need to mark them as executable, i.e.
```
sudo chmod +x ~/yumift_ws/src/yumift/yumi_controller/src/controllers/trajectory_controllers.py
```
and in general with
```
sudo chmod +x ~/yumift_ws/src/yumift/yumi_controller/src/<FILENAME>
```


## Usage
The controllers can be launched with roslaunch. Checkout `yumi_controller/src/core/parameters.py` for some accessible parameters that can be set for IK solvers and routines. To tune the controllers found in `trajectory_controllers.py` shipped with the package, checkout `gains.py`.

### Working with simulation
For testing in a simulation, first start the simulator, this can be replaced by a more realistic simulator as long as it has the same ROS interface as the ABB ros driver. 
```
roslaunch yumi_controller use_sim.launch
```

Another option is to have a Windows machine running RobotStudio with a Virtual Controller. 

### Using the trajectory controllers
To start the trajectory controller use
```
rosrun yumi_controller trajectory_controllers.py simple
```
and send some example trajectories to the controller with
```
rosrun yumi_controller demo_1_some_trajectories.py
```
One could also send commands through the command line:
``` 
rostopic pub /trajectory yumi_controller/YumiTrajectory "
trajectory:
- positionLeft:  [0.4,  0.3, 0.6]
  positionRight: [0.4, -0.3, 0.6]
  orientationLeft:  [0, 0, 0, 1]
  orientationRight: [0, 0, 0, 1]
  gripperLeft: 20
  gripperRight: 20
  pointTime: 4.0
mode: 'individual'"
--once
``` 

### Working with hardware
First of all, you will need to setup the F/T sensors and set the following settings:

|               | IP                | Config                 | Filter | Software Bias Vector (might be wrong) |
|---------------|-------------------|------------------------|--------|---------------------------------------|
| LEFT gripper  | `192.168.125.166` | `#2 - YuMi FT (LEFT)`  | 5 Hz   | `3215 3680   92 791 3867 -2096`       |
| RIGHT gripper | `192.168.125.167` | `#2 - YuMi FT (RIGHT)` | 5 Hz   | `1948  238 -774 631  802 -2714`       |

You can find more information on the hardware here
- [NetBox manual](https://www.ati-ia.com/app_content/documents/9610-05-1022.pdf)
- [Calibration files](https://www.ati-ia.com/Library/Software/FTDigitaldownload/GetCalFiles.aspx) using serials `FT14166` and `FT14167`.
- [FT Sensors](https://schunk.com/us/en/automation-technology/force/torque-sensors/ft/ftd-mini-40-si-20-1/p/EPIM_ID-30832)

When you're done, connect the YuMi via the service port (`XP23` label). 
This will set YuMi as the DHCP server of the network, with IP `192.168.125.1`. 
If you instead want to use YuMi with the WAN port, you must change its IP when launching `yumi_controller/launch/use_real.launch` using the parameter `ip:=X.X.X.X`, and don't forget to change the IPs of the F/T sensors.

Finally, use `start_egm.py` to open a connection with YuMi.  
Also, don't forget to change velocity limits in the EGM config to suit your need; 
this is done by changing the EGM configuration file `yumi_controller/launch/bringup/config_hardware_egm.yaml`.

To conclude, to run the overall system on hardware, run the following commands in two separate terminals
```
roslaunch yumi_controller use_real.launch ip:=192.168.125.1
rosrun yumi_controller start_egm.py
```

### Using force sensors
To use force-based trajectory controllers, first connect to the NetBoxes using
```
roslaunch yumi_controller sensors.launch sensor_ip_right:=192.168.125.167 sensor_ip_left:=192.168.125.166
```
which includes gravity compensation for the ABB SmartGrippers.
Then, start a force-based trajectory controller
```
rosrun yumi_controller trajectory_controllers.py wrenched
rosrun yumi_controller trajectory_controllers.py compliant
```

### Running the `dummy_controller.py`:
Only run this file in simulation as it only serves as a tutorial on how to build controllers.
``` 
roslaunch yumi_controller use_sim_individual.launch 
``` 
then start the controller
``` 
rosrun yumi_controller dummy_controller.py
```

It will simply reach a pre-defined pose, but its scope is to be a simple example of how to write your own controller.


## Notes
This package comes as is. Use it at your own risk.

**Maintainer**: Sebastiano Fregnan
