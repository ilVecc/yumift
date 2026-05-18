# Wrenched YuMi control

> [!CAUTION]
> This README is subject to frequent updates. Sentences might be repetitive, half-written, or even straight-up wrong. 
> Please send me an email if you find anything wrong. I try to keep the installation section as updated as possible.

Modular controllers for the ABB Dual-Arm YuMi robot over ROS Noetic, with a focus on wrench feedback.

This repository contains multiple ROS packages with interfaces and control algorithms for the ABB Dual-Armed YuMi, and in particular for admittance controllers. The core functionality includes Inverse Kinematics (IK) solvers for YuMi in both **individual** (i.e. right and left) and **coordinated** (i.e. absolute and relative) manipulation of the arms. These packages are based on a robot and sensor agnostic Python modules, and different robots can be controlled using the same abstract controller architecture. The IK problem is solved either with classical pseudo-inverted Jacobian or with Hierarchical Quadratic Programming (HQP); in the latter method, the control objectives are solved together with feasibility objectives (see documentation). The robot is visualized in `rviz`, and a very simple kinematic simulator is included when working with real hardware is not possible. 


## Table of Contents
* [General Architecture](#general-architecture)
* [Requirements](#requirements)
  * [Using WSL2](#using-wsl2)
* [Building the package](#building-the-package)
  * [Final touch-ups](#final-touch-ups)
* [Standard workflow](#standard-workflow)
* [Working in simulation](#working-in-simulation)
  * [Execute bring-up command](#execute-bring-up-command)
* [Working with hardware](#working-with-hardware)
  * [RobotWare requirements](#robotware-requirements)
  * [Networking requirements](#networking-requirements)
  * [Execute bring-up command](#execute-bring-up-command-1)
  * [Using F/T sensors](#using-f/t-sensors)
* [Using off-the-shelf controllers](#using-off-the-shelf-controllers)
  * [Example controller](#tutorial-controller)
  * [Tracking controllers](#tracking-controllers)
  * [Trajectory controllers](#trajectory-controllers)
  * [Force controllers](#force-controllers)


## General Architecture
The idea of this package is to provide extendable generic controllers for specific cases/robots. In a sense, this is a Python alternative of `ros_control` controllers, but works on top of the `ros_control` stack. In detail, the Python controllers in this package are abstract and can have any input/output connection to handle read/write instruction to a hardware setup (this is similar to a ROS controller talking via hardware_interface to a hardware). Since the package was created to work with ROS, the states/commands come/go to a `hardware_interface`, creating a nested controller architecture. 

In the specific case of ABB YuMi, the provided semi-concrete Python controller creates joint velocity commands which are sent to the `ros_control` joint velocity controller provided by the `abb_egm_hardware_interface` package, which simply communicates them via Externally Guided Motion (EGM) to the actual EGM controller inside the robot, which is a P-action velocity controller on top of the (finally) actual RAPID current controller of the robot joints.

## Requirements
This package is `python3`-based and written for `ROS Noetic`, which runs on the `Ubuntu 20.04` operative system. Before doing anything else, please satisfy the following requirements:

* `Ubuntu 20.04` can be installed either [on bare metal](https://www.releases.ubuntu.com/focal/) (i.e. on your laptop, in a dual-boot setup for example) or, when using Windows, on the [Windows Subsystem for Linux](https://ubuntu.com/desktop/wsl) (more on this [later](#using-wsl2)). A Docker image will eventually be added for convenience.

* `Python3.8` with `pip`. While the first is the default choice in `Ubuntu 20.04`, the second is not always necessarily installed. Execute
  ```bash
  sudo apt install python3-pip
  ```
  or instead, if `pip` is already part of the system, be sure to use the latest version
  ```bash
  python3 -m pip install -U pip
  ```
  > [!WARNING]
  > **This point is crucial**, as some of this package's dependencies require the latest `pip` features to be compiled/installed.

* `ROS Noetic`, preferably the `Desktop Install` version, following the [official installation guide](https://wiki.ros.org/noetic/Installation/Ubuntu), and **DO NOT SKIP** installing `rosdep` ([section 1.6](https://wiki.ros.org/noetic/Installation/Ubuntu#:~:text=zshrc%0Asource%20~/.zshrc-,Dependencies%20for%20building%20packages,-Up%20to%20now));

* `catkin-tools` to ease the building process
  ```bash
  sudo apt install -y python3-catkin-tools
  ```

* `vcstool` to ease gathering requirements from additional repositories
  ```bash
  sudo apt install python3-vcstool
  ```

### Using WSL2
Although preferable, when a bare metal installation of `ROS` is not possible, we suggest resorting to the Windows Subsystem for Linux, or `WSL2`, as an alternative. On machines running Windows 11 22H2 you can use the convenient mirrored networking option by adding in `%USERPROFILE%\.wslconfig`
```
[wsl2]
networkingMode=mirrored
firewall=false
```

> [!WARNING]
> You might still need to disable Windows Firewall, or even better, set specific rules to avoid disabling it. 
> More info can be found on the [official WSL documentation page](https://learn.microsoft.com/en-us/windows/wsl/networking).

When done configuring the WSL, run `wsl --shutdown` in a PowerShell terminal and restart the WSL instance. 

> [!TIP]
> We suggest using the `Windows Terminal` app [from the Microsoft Store](https://aka.ms/terminal).


## Building the package
Create a folder for the Catkin workspace
```bash
mkdir -p ~/yumift_ws/src && cd ~/yumift_ws && catkin init && cd src
```

Clone [`abb_robot_driver`](https://github.com/ros-industrial/abb_robot_driver) and add its dependecies with
```bash
git clone https://github.com/ros-industrial/abb_robot_driver.git
vcs import . --input abb_robot_driver/pkgs.repos
```

If force feedback is needed (e.g. for admittance/force controllers) and you are using `ATI NetBox` hardware, clone the [`netft_utils`](https://github.com/ilVecc/netft_utils) repository with
```bash
git clone https://github.com/ilVecc/netft_utils.git
```

Next, clone this respository with
```bash
git clone https://github.com/ilVecc/yumift.git
```

Finally, install all the required dependencies and modules with
```bash
rosdep update --rosdistro=noetic
rosdep install --from-paths . --ignore-src --rosdistro noetic -y
python3 -m pip install -U -e yumift/dynamicals
python3 -m pip install -U -e yumift/pathfinder
```
and build workspace with
```bash
catkin build
```

> [!WARNING]
> Installing `pinocchio` requires `cmeel` and `manylinux_2_28`, which is only available on `pip>=20.3`. 
> If you skipped updating `pip`, you might incur into issues at this point.

To start using the packages, source the setup file with
```bash
source ~/yumift_ws/devel/setup.bash
```


### Final touch-ups
Instead of sourcing the packages at every startup, add the line above to `.bashrc` with
```bash
echo "source ~/yumift_ws/devel/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

Since ROS Noetic is now in its End Of Life stage, an annoying message will pop-up every time you launch `RViz` (the 3D visualization window). Use the following command to disable it
```bash
echo "export DISABLE_ROS1_EOL_WARNINGS=1" >> ~/.bashrc && source ~/.bashrc
```

> [!NOTE] 
> On some systems, especially if graphics hardware acceleration is not available (e.g. WSL2), `RViz` might not open, might open but show a black window, or might start and then exit immediately with a error log message. 
> If this happens, manually force to use software 3D computation using 
>  ```bash
>  echo "export LIBGL_ALWAYS_SOFTWARE=1" >> ~/.bashrc && source ~/.bashrc
>  ```
> Depending on your CPU, this might effect the frames per second of the 3D view.


## Standard workflow
There are well-known sets of steps you will need to perform to work with robots, especially in ROS. These usually are:

- **robot bring-up**, which loads the software components of the robot on your computer, and either starts its simulation environment (*simulation bring-up*), or connects to the real hardware (*hardware bring-up*). Here with *simulation* we mean any kind of "good enough" kind of replica of the real robot, that being anything from a naive kinematics simulation to a sophisticated physics engine, while with *hardware* we mean a device which "acts exactly" as the robot, that being the actual robot or its digital twin (not necessarily endowed with good physics simulation capability). In general, working with *hardware* will require a connection to an external interface, while working in *simulation* will usually translate to local API calls; 
- **remote control** approval, which enables external components (i.e. your computer) to issue commands to the robot (this step is usually only required when working with hardware). This tells the hardware interface to accept real-time commands coming from a different machine instead of using programs written in the manufacturer's proprietary programming language; 
- **sensor bring-up**, which turns on all the sensors on the robot (e.g. cameras, force/torque transducers, LiDARs) and in the workspace (e.g. GPS beacons) and connects to them. This step is required only when some components of the *scene* (the world in which the robot is operating) are not automatically started during the *robot bring-up*, perhaps because they are not related to the robot or have been mounted as an external addition to it;
- **calibration**, which resets the sensors to a desired working standard (e.g. add a bias to the IMU so to read zero velocity when at rest, estimate the internal parameters of a camera system so to perform accurate computer vision tasks) and places them in the correct pose with respect to the world reference frame, usually inserting them in a *transformation graph*;
- **controller start-up**, which finally starts the (external) controller sending commands to the robot. 

In this package, the steps are very similar, with a little deviation for the calibration part:
1. Bring-up the robot, either in simulation (i.e. start kinematics simulation) or with hardware (i.e. establish EGM connection)
2. (OPT) Communicate with hardware (i.e. start EGM communication)
now, if wrench feedback is needed, 
3. Bring-up the wrench sensors
4. Start the cartesian velocity controller
5. Calibrate sensors using the utility
6. Stop the controller
and finally
7. Start the desired controller

In the following sections you will find all the required information to achieve each of the steps above.


<!-- TODO 
There are two "state update" mode (individual and dual). When running dual controllers, append `individual:=false` to the `roslaunch` commands in the subsections; this separation allows avoidance  of unnecessary computation, making the state update faster (up to 250Hz in individual mode). -->

## Working in simulation
As a first step, it is recommended to work safely in simulation. To do so, the `yumift_simulation` package conventiently provides a simple kinematic simulator of Yumi, just enough to test your kinematic controllers. To start the simulator use the command
```bash
roslaunch yumift_common bringup.launch
```
You should now see an RViz window containing Yumi. The bring-up command loaded the kinematic structure and the 3D models of Yumi and started the simulator, which is now sending the current state of the robot and waiting for joint velocity commands. 

> [!NOTE]
> This is a very basic simulator, and can be replaced by a much more realistic one as long as it has the same ROS I/O interface dictated by the ABB ROS drivers. 
> A better alternative would be to have a Windows computer running ABB RobotStudio with a Virtual Controller of Yumi, essentially using a digital twin of the hardware. As previously said, to use this method, follow the same instructions as in the [hardware section](#working-with-hardware). 


## Working with hardware
Before anything else, the hardware must be set up in a specific way, so to allow external control. 
This step requires a Windows computer with [ABB RobotStudio](https://www.abb.com/global/en/areas/robotics/products/software/robotstudio-suite) installed.

### RobotWare requirements
YuMi must be flashed with a specific RobotWare configuration, with the `EGM` option enabled and the `State Machine` add-in installed. 

<!-- TODO 
add instructions found in Google Slides 
OR
https://github.com/kth-ros-pkg/yumi/wiki -->

### Networking requirements
To physically connect your computer to YuMi, you must use an Ethernet connection and choose between two different sockets on the side of Yumi: 
- *Service port* (`XP23` label) : YuMi acts as DHCP server, and uses the fixed IP `192.168.125.1`;
- *WAN port* (`XP28` label) : YuMi acts as a device on an (external) network, and its IP must be set (or added) in YuMi's settings under `Control Panel / Configuration / Communication / IP Settings`.

Next, set the IP of the remote device sending EGM commands: Yumi will accept only commands coming from this IP. 
In YuMi's settings under `Control Panel / Configuration / Communication / Trasmission Protocol`, set (or add) the following entries:

| Name     | Type  | Remote address | Remote port | Local port |
|----------|-------|----------------|-------------|------------|
| ROB_L    | UDPUC | <remote_ip>    | 6511        | 0          |
| ROB_R    | UDPUC | <remote_ip>    | 6512        | 0          |
| UCdevice | UDPUC | 127.0.0.1      | 6510        | 0          |


### Execute bring-up command
With all the setup done, to bring-up YuMi open a terminal and run
```bash
roslaunch yumift_common bringup.launch use_real:=true
```
Again, you should see a RViz window containing the robot in a configuration not necessarily similar to the real one; this is expected. In another terminal enable the EGM communication using
```bash
roslaunch yumift_common start_egm.launch
```
If the second command fails, follow the instructions in the error message and launch again the command until it succedes. Once the command succeded, you should now see the robot visualization in the same configuration as the real robot; now, the setup is ready to be used.

> [!TIP]
> EGM communication will be cut every time an hardware failure/error occurs, which can happen very often (e.g. payload too high, collision detected, command sent but ignored, etc.).
> Every time the communication is cut, you need to restart it with the command above. These frequent interruptions can become very frustrating during debugging. To avoid this, allow the EGM connection to automatically restart by adding to the launch command the following extra argument
> ```bash
> roslaunch yumift_common start_egm.launch resilient:=true
> ```
> The EGM communication nodes can also be started directly in the bring-up using
> ```bash
> roslaunch yumift_common bringup.launch use_real:=true start_egm_link:=true
> ```
> or in its resilient version via
> ```bash
> roslaunch yumift_common bringup.launch use_real:=true resilient_egm_link:=true
> ```

> [!CAUTION]
> Using the tip above is strongly discouraged for beginners and recommended to expert users only! 
> The robot will automatically restart moving few instants after it failed. 
> This is particularly dangerous if the robot self-collided, stopped near to a singularity, or if the controller you are using doesn't handle big position errors very well. 
> **DO NOT USE THIS OPTION** if you are not **certain** of what you are doing.

> [!IMPORTANT]
> The parameters of the robot's motion are setup at the beginning of the EGM communication, in particular joint velocity limits. If different values are needed, they must changed in `yumift_common/config/config_hardware_egm.yaml`. 


### Using F/T sensors
In general, any force/torque sensor system can be used, as under the hood the robot state updater simply requires `geometry_msg/Wrench` messages over the `sensors/wrench/right` and `sensors/wrench/left` topics. In this documentation, we will use two ATI's *Net F/T* sensor systems with [Schunk SI-40 F/T](https://schunk.com/us/en/automation-technology/force/torque-sensors/ft/ftd-mini-40-si-20-1/p/EPIM_ID-30832) transducers.

> [!NOTE]
> ATI's *Net F/T* sensor system is composed of a force/torque (or wrench) transducer connected to an ATI Net Box interpreting raw wrench data and sending it via TCP over the network ([more info](https://www.ati-ia.com/products/ft/ft_NetFT.aspx)). 
> This data is received, filtered and published in the ROS network using the [`netft_utils` package](https://github.com/ilVecc/netft_utils.git), containing both the drivers required to unpack the data and the ROS node (i.e. a wrapper) that relays it over the ROS ecosystem.
> 
> In the ATI NetBox web interface, set the IP, the transducer-specific configuration file (specific for each F/T sensor in use, [find it here](https://www.ati-ia.com/Library/Software/FTDigitaldownload/GetCalFiles.aspx) using the serial numbers found on your transducer), filter frequency (e.g. 5Hz) and Software Bias Vector (if necessary). At each power cycle, the Software Bias Vector (SBV) is reset, so remember to set it back to the required values at start-up, or use CGI requests to do it automatically (see [ATI NetBox manual](https://www.ati-ia.com/app_content/documents/9610-05-1022.pdf)). We strongly suggest to use the SBV only in strictly controlled environment and thus, in general, to avoid it, as it's just better to manually set constant and dynamic biases via external software (e.g. `netft_utils`). 

To use the sensors, first bring-up YuMi and then launch the sensor bring-up via
```bash
roslaunch yumift_common sensors.launch
```
This command will automatically start the communication with the two wrist-mounted sensors and publish a number of topics for both sensors. You can see the two wrenches visualised as vectors in the RViz window. The topics are the raw and biased data in three different frames: the sensor, the tool, and the world frames. With *biased* here we mean 
- *fixed-orientation bias*, constant and set once, at equilibrium, enough for applications where the orientation remains constant (i.e. at bias time, the current wrench is used as bias);
- *static/dynamic bias*, with a constant and a variable compensation, the first set once, and the second updated at each reading based on the tool weight (i.e. the static bias compensates for a constant sensor error, while the dynamic bias compensates the gravity influence on the tool attached to the sensor).

To set a bias, please read `netft_utils` documentation. For convenience, we provide a script that automatically finds the static and dynamic bias when ABB SmartGrippers are attached to the sensors. To run it, first start the provided trajectory controller
```bash
rosrun yumift_controllers trajectory_controllers.py
``` 
and then, in another terminal, start the data collection and biasing script
```bash
rosrun yumift_controllers ftsensor_calibration.py
```
Once done, you'll see the forces in RViz completely compensated. You can now stop the trajectory controller. 

> [!IMPORTANT]
> Any other wrench sensor system can be used. In that case, different procedures apply. Refer to the specific manual and the ROS wrappers for your case.
> The launch files provided in this package only work with the hardware/software stack described above.


## Using off-the-shelf controllers
A set of velocity controllers is shipped with this package. These are categorized as *individual* (acting separately on each arm) or *dual* (acting in either *individual* or *coordinated* fashion, i.e. absolute and relative pose control). All controllers receive either a `yumift_msgs/YumiPosture` (tracking controllers) or a `yumift_msgs/YumiTrajectory` (trajectory controllers) message (the latter is simply a list of the former). Trajectory controllers are *"routinable"*, meaning that they can execute pre-registered routines that achieve particular motions (e.g. "go back home", "grippers point down", etc.). 

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

> [!IMPORTANT]
> Checkout `yumift_controllers/config/gains.yaml` to tune the gains of the various controllers. 
> For trajectory controllers, a set of example trajectories is also provided (see following subsections for more info).

### Example controller:
This controller only serves as a tutorial on how to build controllers. This controller simply commands to achieve a pre-defined pose.
For safety/educational purpose, only run this file in simulation. To start the controller use
```bash
rosrun yumift_controllers example_controller.py
```

> [!IMPORTANT]
> Checkout `yumift_controllers/common/parameters.py` for some useful generic controller parameters and `yumift_controllers/ik/` for IK solvers and routines.

### Trajectory controllers
To start a trajectory controller use
```bash
rosrun yumift_controllers trajectory_controllers.py simple
```
with available options `simple`, `dual`, `wrenched`, `compliant`.

Trajectories are cubic polynomials and are automatically computed based on the received `YumiPosture` list.

Send some example trajectories to the controller with
```bash
rosrun yumift_examples example_1_simple_trajectory.py
```
Have a look at the various `example_*` files for a description of the trajectories.

Alternatively, although discouraged, you can send commands directly through the command line with
```bash
rostopic pub /trajectory yumift_msgs/YumiTrajectory "
trajectory:
- primary_pose:
    position: 
      x:  0.4
      y: -0.3
      z:  0.6
    orientation: 
      w: 1
  secondary_pose:
    position:
      x:  0.4
      y:  0.3
      z:  0.6
    orientation:
      w: 1
  gripper_right: 20
  gripper_left: 20
  time_to_execute: 4.0
mode: yumift_msgs/YumiTrajectory.INDIVIDUAL"
--once
``` 

### Tracking controllers
To start a tracking controller use
```bash
rosrun yumift_controllers tracking_controllers.py single
```
with available options `single`, `whole`.

This controller is more versatile, as its intended use is to continuously receivce `YumiPosture` messages from a node, bypassing the trajectory creation described above.   

### Force controllers
Start a force-based trajectory controller with
```bash
rosrun yumift_controllers trajectory_controllers.py wrenched
```
with available options `wrenched`, `compliant`.

When writing custom controllers, forces are available in the `RobotState` object received in the controller main loop. If no sensors are available, a simple "wrench simulator" can be used running
```bash
rosrun yumift_controllers wrench_simulator.py
```

## Notes
This code comes as is. Use it at your own risk.

**Maintainer**: Sebastiano Fregnan ([sebastiano@fregnan.me](mailto:sebastiano@fregnan.me))
