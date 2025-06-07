# Wrenched YuMi control
Modular controllers for the ABB Dual-Arm YuMi robot over ROS Melodic. 

This package contains interfaces and control algorithms for the ABB Dual-Armed YuMi, and in particular for admittance controllers. The core functionality includes Inverse Kinematics (IK) solvers for YuMi in both **individual** (i.e. right and left) and **coordinated** (i.e. absolute and relative) manipulation of the arms. In general, this package is robot and sensor agnostic, as different robots can be controlled using the same abstract controller architecture. The IK problem is solved either with classical pseudo-inverted Jacobian or with Hierarchical Quadratic Programming (HQP); in the latter method, the control objectives are solved together with feasibility objectives (see documentation). The robot is visualized in `rviz`, and a very simple kinematic simulator is included when working with real hardware is not possible. 


## Table of Contents
* [General Architecture](#general-architecture)
* [Requirements](#requirements)
  * [Using WSL2](#using-wsl2)
* [Building the package](#build)
* [Robot bring-up](#robot-bring-up)
  * [Working with hardware](#working-with-hardware)
    * [RobotWare requirements](#robotware-requirements)
    * [Networking requirements](#networking-requirements)
    * [Using F/T sensors](#using-f/t-sensors)
  * [Working with simulation](#working-with-simulation)
* [Using off-the-shelf controller](#using-off-the-shelf-controller)
  * [Tutorial controller](#tutorial-controller)
  * [Tracking controllers](#tracking-controllers)
  * [Trajectory controllers](#trajectory-controllers)
  * [Force controllers](#force-controllers)


## General Architecture
The idea of this package is to provide extendable generic controllers for specific cases/robots. In a sense, this is a Python alternative of `ros_control` controllers, but works on top of the `ros_control` stack. In detail, the Python controllers in this package are abstract and can have any input/output connection to handle read/write instruction to a hardware setup (this is similar to a ROS controller talking via hardware_interface to a hardware). Since the package was created to work with ROS, the states/commands come/go to a `hardware_interface`, creating a nested controller architecture. 

In the specific case of ABB YuMi, the provided semi-concrete Python controller creates joint velocity commands which are sent to the `ros_control` joint velocity controller provided by the `abb_egm_hardware_interface` package, which simply communicates them via Externally Guided Motion (EGM) to the actual EGM controller inside the robot, which is a P-action velocity controller on top of the (finally) actual RAPID current controller of the robot joints.

## Requirements
This package expects an Ubuntu 20.04 system and uses ROS Noetic (Desktop version is better).

* for ROS Noetic Desktop, follow the [official installation guide](https://wiki.ros.org/noetic/Installation/Ubuntu), and do not skip installing `rosdep`;

* to ease the building process, install `catkin-tools`
```
sudo apt install -Y python3-catkin-tools
```

### Using WSL2
If one want to use ROS Noetic over WSL2, it is best to do so on an Hyper-V enabled device so to make use of the mirrored networking option by adding in `.wslconfig`
```
[wsl2]
networkingMode=mirrored
firewall=false
```
Also, disable Windows Firewall, or even better, set specific rules without disabling it. 
When done, run `wsl --shutdown` in a PowerShell terminal to restart the WSL instance.


## Building the package
Create a folder for the Catkin workspace
```
mkdir -p ~/yumift_ws/src && cd ~/yumift_ws/src && catkin init
```

Next, clone [`abb_robot_driver`](https://github.com/ros-industrial/abb_robot_driver) and run the required commands with
```
git clone https://github.com/ros-industrial/abb_robot_driver.git
gpg --keyserver hkps://keyserver.ubuntu.com --recv-key 0xAB17C654
sudo apt update; sudo apt install python3-vcstool
vcs import . --input abb_robot_driver/pkgs.repos
```

If force feedback is needed (e.g. for admittance/force controllers), clone the `netft_utils` repository with
```
git clone https://github.com/UTNuclearRoboticsPublic/netft_utils.git -b master
```

Next, clone this respository with
```
git clone https://github.com/ilVecc/yumift.git
```

Finally, install all the required dependencies with
```
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro noetic
```
and build workspace with
``` 
catkin build
```

For easy startup, add to `.bashrc` the source command
``` 
echo "source ~/yumift_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
``` 

To `rosrun` files in the workspace, one may need to mark them as executable with
```
sudo chmod +x ~/yumift_ws/src/yumift/yumi_controller/src/<FILENAME>
```


## Robot bring-up
Before running anything from this package, YuMi must be brang-up. There are two operative modes (hardware and simulation) and two "state update" mode (individual and dual). When running dual controllers, append `individual:=false` to the `roslaunch` commands in the subsections; this separation allows avoidance  of unnecessary computation, making the state update faster (up to 250Hz in individual mode).

### Working with hardware
Before using the controllers on YuMi, it must be set up in a specific way (see info in the following sections). When done, to bring-up YuMi run
```
roslaunch yumi_controller bringup.launch robot_ip:=<yumi_ip>
```
and in another terminal enable EGM communication using
```
rosrun yumi_controller start_egm.py
```
If this step returns errors, follow the instructions displayed in the terminal.

If different joint velocity limits are needed, they must changed in `config/config_hardware_egm.yaml`. 

#### RobotWare requirements
YuMi must be running a RobotWare version with `EGM` option and `State Machine` add-in. 

#### Networking requirements
Connect the YuMi via the service port (`XP23` label) or the WAN port (`XP28` label). 
These two networks use different settings:
- *Service port*: YuMi acts as DHCP server, and uses the fixed IP `192.168.125.1`;
- *WAN port*: YuMi acts as a device on an external network, and the IP must be set (or added) in YuMi settings `Control Panel / Configuration / Communication / IP Settings`.

Next, set the IP of the remote device sending EGM commands. To do so, in YuMi settings `Control Panel / Configuration / Communication / Trasmission Protocol` set (or add):

| Name     | Type  | Remote address | Remote port | Local port |
|----------|-------|----------------|-------------|------------|
| ROB_L    | UDPUC | <remote_ip>    | 6511        | 0          |
| ROB_R    | UDPUC | <remote_ip>    | 6512        | 0          |
| UCdevice | UDPUC | 127.0.0.1      | 6510        | 0          |

#### Using F/T sensors
In general, any sensor can be used, as the robot state updater simply requires topics sending `geometry_msg/Wrench`.
However, this package was tested only with Schunk SI-40 sensors connected via ATI NetBox netboxes, which can be made available in ROS using the `netft_utils` package. 

If working with ATI NetBox, set IP, configuration file (the specific one for the F/T sensor in use), filter frequency (e.g. 5Hz) and Software Bias Vector. At each power cycle, the Software Bias Vector is reset, so remember to set it back to the required values, or use CGI requests to do it automatically (see ATI NetBox documentation). Find more information on the hardware here:
- [NetBox manual](https://www.ati-ia.com/app_content/documents/9610-05-1022.pdf)
- [Calibration files](https://www.ati-ia.com/Library/Software/FTDigitaldownload/GetCalFiles.aspx) using serials `FT14166` and `FT14167`.
- [FT Sensors](https://schunk.com/us/en/automation-technology/force/torque-sensors/ft/ftd-mini-40-si-20-1/p/EPIM_ID-30832)

Next, to achieve gravity compensation of the ABB SmartGrippers, bring-up YuMi and connect to the netboxes using
```
roslaunch yumi_controller sensors.launch sensor_ip_right:=<right_ip> sensor_ip_left:=<left_ip>
```
Otherwise, if gravity compensation is not needed (i.e. no grippers are mounted), use
```
roslaunch yumi_controller sensors.launch sensor_ip_right:=<right_ip> sensor_ip_left:=<left_ip> use_raw:=true
```

### Working with simulation
To work safely without hardware, start the kinematic simulator instead of EGM using
```
roslaunch yumi_controller bringup.launch
```
This is a very basic kinematic simulator, and can be replaced by a more realistic one as long as it has the same ROS interface as the ABB ROS drivers. Another option is to have a Windows machine running RobotStudio with a Virtual Controller. To use this method, follow the same instructions as in the previous section. 

## Using off-the-shelf controllers
A set of velocity controllers is shipped with this package. These are categorized as *individual* (acting separately on each arm) or *dual* (acting in either *individual* or *coordinated* fashion, i.e. absolute and relative pose control). All controllers receive either a `yumi_controller/YumiPosture` (tracking controllers) or a `yumi_controller/YumiTrajectory` (trajectory controllers) message (the latter is simply a list of the former). Trajectory controllers are *"routinable"*, meaning that they can execute pre-registered routines that achieve particular motions (e.g. "go back home", "grippers point down", etc.). 

The available controllers are:
- `SingleTrackingController` achieves independent cartesian tracking control (i.e. each arm receives a dedicated posture, from two separate topics);
- `WholeTrackingController` achieves whole-body cartesian tracking control (i.e. both arms receive a the same posture, from a single separate topic);
- `SimpleTrajectoryController` achieves individual cartesian trajectory control;
- `DualTrajectoryController` achieves dual cartesian trajectory control;
- `WrenchedTrajectoryController` achieves dual direct wrench cartesian trajectory control (i.e. wrench is scaled and directly added to the velocity control action);
- `CompliantTrajectoryController` achieves dual admittance cartesian trajectory control (i.e. wrench is fed into an admittance which compensates the deviation from the desired trajectory).

In each controller, a `YumiPosture` message can be set to represent six different modalities: 
- `right`, i.e. only use fields related to the right arm; 
- `left`, i.e. ditto but left;
- `individual`, i.e. `right` and `left` simultaneously;
- `absolute`, i.e. interpret the right arm fields ase absolute pose;
- `relative`, i.e. interpret the left arm fields ase relative pose;
- `coordinated`, i.e. `absolute` and `relative` simultaneously.

Checkout `core_controllers/gains.py` to tune the gains of the various controllers. For trajectory controllers, a set of example trajectories is also provided (see following subsections for more info).

### Tutorial controller:
This controller only serves as a tutorial on how to build controllers. This controller simply commands to achieve a pre-defined pose.
For safety/educational purpose, only run this file in simulation. To start the controller use
``` 
rosrun yumi_controller dummy_controller.py
```

Checkout `core_common/parameters.py` for some useful parameters for IK solvers and routines.


### Trajectory controllers
To start a trajectory controller use
```
rosrun yumi_controller trajectory_controllers.py simple
```
with available options `simple`, `dual`, `wrenched`, `compliant`.

Trajectories are cubic polynomials and are automatically computed based on the received `YumiPosture` list.

Send some example trajectories to the controller with
```
rosrun yumi_controller demo_1_some_trajectories.py
```
Have a look at the various `demo_*` files for a description of the trajectories.

Alternatively, send commands directly through the command line with
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

### Tracking controllers
To start a tracking controller use
```
rosrun yumi_controller tracking_controllers.py single
```
with available options `single`, `whole`.

This controller is more versatile, as its intended use is to continuously receivce `YumiPosture` messages from a node, bypassing the trajectory creation described above.   


### Force controllers
Start a force-based trajectory controller with
```
rosrun yumi_controller trajectory_controllers.py wrenched
```
with available options `wrenched`, `compliant`.

When writing custom controllers, forces are available in the `RobotState` object received in the controller main loop. If no sensors are available, a simple "wrench simulator" can be used running
```
rosrun yumi_controller wrench_simulator.py
```

## Notes
This package comes as is. Use it at your own risk.

**Maintainer**: Sebastiano Fregnan ([sebastiano@fregnan.me](mailto:sebastiano@fregnan.me))
