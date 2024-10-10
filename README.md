# Wrenched YuMi control
ROS implementation for advanced YuMi control. 
This package contains a modular interface to create controllers for the ABB Dual-Arm YuMi robot.


## Table of Contents
* [General](#general)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)


## General
<!-- For more in-depth information and comprehensive installation guide, look in the [wiki page](https://github.com/CRIS-Chalmers/yumi/wiki).  -->
This package contains interfaces and control algorithms for the ABB Dual-Armed YuMi. 
The core functionality includes inverse kinematics for YuMi in both individual and coordinated manipulation frames of the arms. 

The idea is that you create a controller class that inherits the `YumiDualController` class and that you can then build functionally on top of it, see `yumi_controller/src/examples/controller_tutorial.py`. There is also a trajectory following controller implemented using the same structure, see `yumi_controller/src/controllers/trajectory_controllers.py`. This controller is mainly made to operate with slow motions as no dynamical effects have been included in the control loop. The inverse kinematics problem is solved with Hierarchical Quadratic Programing (HQP) implemented using the `quadprog` solver. The control objectives are solved together with feasibility objectives (wiki page for more info). The robot is visualized in `rviz` in real time and a very simple kinematics simulator is included. 


## Dependencies
This package expects Ubuntu 18.04 and uses ROS Melodic Desktop with Python3, so ROS and Catkin must be reconfigured to use Python3 instead of Python2.
If you plan to use WSL2 on an Hyper-V enable device, a good setup would be to set mirrored networking by adding 
```
[wsl2]
networkingMode=mirrored
```
in the `.wslconfig` file found in your user folder andto disable Windows Firewall (even better if you set specific rules without disabling it).


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
python3 -m pip install numpy scipy quadprog
python3 -m pip uninstall em
python3 -m pip install empy==3.3.4
```

* to build the workspace, cmake 3.12 is needed, so upgrade it following [this guide](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu)


## Installation
Create a folder for the catkin workspace
```
mkdir -p ~/yumift_ws/src && cd ~/yumift_ws/src
```
then clone [`abb_robot_driver`](https://github.com/ros-industrial/abb_robot_driver) __OR__ run these commands (possibly outdated)
```
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xAB17C654  # might fail, if so ignore it
sudo apt update; sudo apt install python3-vcstool
vcs import . --input https://github.com/ros-industrial/abb_robot_driver/raw/master/pkgs.repos 
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro melodic
```
__IF__ `rosdep` doesn't work, you need to manually install the driver interfaces with
```
git clone abb_robot_driver_interfaces
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
cd geometry2/
cd ..
```
and finally clone this respository
```
git clone https://github.com/ilVecc/yumi_controller.git
```

Now build `yumift_ws` workspace
``` 
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
``` 
and add source to bashrc for easy startup
``` 
echo "source ~/yumift_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
``` 

Finally, to `rosrun` files in the workspace, you might need to mark them as executable with
```
sudo chmod +x src/yumi/yumi_controller/src/<FILENAME>
```


## Usage
The controllers can be launched with roslaunch. Checkout `yumi_controller/src/core/parameters.py` for some accessible parameters that can be set. To tune the controllers found in `trajectory_controllers.py` shipped with the package, checkout `gains.py`.

### Using simulation
For testing in a simulation, first start the simulator, this can be replaced by a more realistic simulator as long as it has the same ROS interface as the ABB ros driver. 
```
roslaunch yumi_controller use_sim.launch
```

### Using hardware
Use `start_egm.py` to open a connection with YuMi (when linked via the service port, the IP is `192.168.125.1`; find this argument in `yumi_controller/launch/use_real.launch`). Also, don't forget to change velocity limits in ABB driver; this is done by changing the EGM configuration file `yumi_controller/launch/bringup/config_hardware_egm.yaml`.
```
roslaunch yumi_controller use_real.launch
rosrun yumi_controller start_egm.py
```

### Using the trajectory controllers
To start the trajectory controller use
```
rosrun yumi_controller trajectory_controller.py
```
and send some example trajectories to the controller with
```
rosrun yumi_controller example_1_some_trajectories.py
```
One could also send commands through the command line:
``` 
rostopic pub /trajectory yumi_controller/Trajectory_msg "
trajectory:
- positionLeft:  [0.4,  0.3, 0.6]
  positionRight: [0.4, -0.3, 0.6]
  orientationLeft:  [0, 0, 0, 1]
  orientationRight: [0, 0, 0, 1]
  pointTime: 4.0
mode: 'individual'"
--once
``` 

To use force-based trajectory controllers, first connect to the NetBoxes using
```
roslaunch yumi_controller sensors.launch
```
which includes gravity compensation for the ABB SmartGrippers.
Then, start a force-based trajectory controller
```
rosrun yumi_controller trajectory_controller.py wrenched
rosrun yumi_controller trajectory_controller.py compliant
```


<!-- ### Running the customControllerTutorial.py:
Only run this file in simulation as it only serves as a demonstration purposes i.e. build your own custom controller for your use case. Start by launching the simulation as above. The launch the base.launch, this launches the visualization and some background nodes. 
``` 
roslaunch controller base.launch 
``` 
then to start the controller run
``` 
rosrun controller customControllerTutorial.py
```  -->


## Notes

This package comes as is. Use it at your own risk.

**Maintainer**: Sebastiano Fregnan
