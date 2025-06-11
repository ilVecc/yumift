This is a quick guide on some of the functionality of the YuMi Hardware

## Table of Contents
* [General](#General)
* [Manually change the configuration of arms](#ManuallyConfiguration)
* [Calibrate grippers](#CalibrateGrippers)
* [Set up yumi for automatic control](#AutoControl)
* [Update revolute encoders](#Encoders)
* [Ethernet and network setup](#Ethernet)
* [Change grip force](#GripForce)

## General 
* Do not leave YuMi power off under long periods of time, the internal batteries can be discharged to a point where they can't be charged again. 
* Instead leave the robot powered on but the turn of the motors and put it in manual mode. 
* The robot is controlled through ethernet. 
* Do not control the robot when running a VM. 
* Make sure that your computer is not saturating the CPU when running the controller. 

## Manually change the configuration of arms <a name="ManuallyConfiguration"/>
First, make sure that the robot is in manual mode and that the motors are turned off (see _set up yumi for automatic control_). There are two buttons under/behind the shoulders of YuMi. Make sure that you hold the arm before releasing the breaks as they otherwise will fall. Then keep the brake release button pressed in while changing the configuration of the arm. Once released the breaks will apply again. 

<img src="images/breakRelease.jpg"  width="50%" height="50%">

## Calibrate grippers <a name="CalibrateGrippers"/>
Before the grippers can be used they need to be calibrated. First, open the menu.

<img src="images/Menu.jpg"  width="50%" height="50%">

Then navigate to the smart grippers. 

<img src="images/smartGripper1.jpg"  width="50%" height="50%">

Choose one of the grippers, this has to be done for both. 

<img src="images/smartGripper2.jpg"  width="50%" height="50%">

Jog the grippers until they are fully closed and then press the calibration button. When they have been calibrated, the rest of the functionality should unlock. 

<img src="images/smartGripper3.jpg"  width="50%" height="50%">

## Set up yumi for automatic control <a name="AutoControl"/>

To be able to control the robot from the ROS driver the robot needs to be in automatic mode with the motors on. Also make sure that the encoders and grippers are already calibrated. To access the operator pannel. 

<img src="images/AutomaticControl.jpg"  width="50%" height="50%">

And then turn on the motor and put the operator mode into Auto. If the operator panel is not showing press the top figure to the right edge of the screen. 

<img src="images/AutomaticControl2.jpg"  width="50%" height="50%">

In the same way the robot can be put into manual mode and the motors can be turned off. 

## Update revolute encoders <a name="Encoders"/>
If the robot has been turned off for long periods and the internal battery is depleted or the position of the robot is not correct. Then the revolution counters can be updated. First, the robot should be placed in the calibration pose. At each joint, there are markings or tabs that should be aligned. Once the arms are in the correct configuration, navigate to the menu and press calibration. 

<img src="images/calibrate1.jpg"  width="50%" height="50%">

Choose one of the arms to be calibrated. 

<img src="images/calibrate2.jpg"  width="50%" height="50%">

Call the calibration function. 

<img src="images/calibrate3.jpg"  width="50%" height="50%">

Press the start button. 

<img src="images/calibrate4.jpg"  width="50%" height="50%">

Choose 2 for updating the revolution counters. 

<img src="images/calibrate5.jpg"  width="50%" height="50%">

Choose which joints that should be calibrated and press next two times to start the calibration. For all joints it can take around 5 min. 

<img src="images/calibrate6.jpg"  width="50%" height="50%">


## Ethernet and network setup. <a name="Ethernet"/>
The ethernet cable should be connected to the XP23 Service port on YuMi. 

On the Ubuntu laptop, the wired network settings need to be changed to match what is set on the YuMi. Use the following. 

<img src="images/UbuntuWiredSettings2.png"  width="50%" height="50%">

The network settings on YuMi should not need to be changed, only in the case that system is reset or some other problems occure. To find the settings first enter the menu.   

<img src="images/YuMiNetworkSetup1.jpg"  width="50%" height="50%">

Then enter the configuration. 

<img src="images/YuMiNetworkSetup2.jpg"  width="50%" height="50%">

Change the topic to communication. 

<img src="images/YuMiNetworkSetup3.jpg"  width="50%" height="50%">

Enter the transmition protocal. 

<img src="images/YuMiNetworkSetup4.jpg"  width="50%" height="50%">

In here the network properties of the controlling computer can be changed. Needs to be changed for both arms and the UCDevice. 

<img src="images/YuMiNetworkSetup5.jpg"  width="50%" height="50%">



## Change grip force, RAPID <a name="GripForce"/>

This shows one way to change the grip force during automatic control. Any force changes that are made in the smart gripper (i.e. same place as we calibrated the grippers) gets overwritten. There might exist a better way to do this, anyhow, one way to do it is to make changes to the RAPID code. To achieve that, first enter the menu and then the program editor. Then choose on of the arms. 

<img src="images/GripForce1.jpg"  width="50%" height="50%">

Then enter the smart gripper module. 

<img src="images/GripForce2.jpg"  width="50%" height="50%">

Enter the initialization function.

<img src="images/GripForce3.jpg"  width="50%" height="50%">

Change the holding force. (max is 20 N)

<img src="images/GripForce4.jpg"  width="50%" height="50%">

