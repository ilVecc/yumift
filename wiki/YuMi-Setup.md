# Working with the YuMi

This guide will illustrate the process required to work with an ABB Dual-Arm YuMi robot, either with a digital-twin in RobotStudio or with EGM.

1. Create wired connection between Windows and Ubuntu computer
2. Create RobotStudio simulation on Windows computer
3. Install abb_robot_driver on Ubuntu computer
4. Get started with ROS driver 
5. Work with real robot


## WINDOWS
### Prerequisite 
- Install RobotStudio
- Go to the Add-Ins tab: 
    1. Search for **RobotWare for IRC5**: Add the latest version (>6.11.01)
    2. Search for **StateMachine Add-In 1.1**: Press Add
    3. (Optional) Search for **SmartGripper**: Add the latest version
    4. (Optional) Search for **Virtual SmartGripper**: Add the latest version 

### Create wired connection
1. Go to `Control Panel > Network > Internet > Network and Sharing center`, where you should see your active networks.
2. Select `Ethernet`, under the `Unidentified network`, and you’ll see the `Ethernet Status` window. 
3. Select `Internet Protocol Version 4 (TCP/IPv4)` item and then click on `Properties`.
4. Under `General`, select **Use the following IP address:** and set `192.168.125.2/24`, then click OK and close the windows.

### Create RobotStudio simulation
1. Open `RobotStudio`, and under tab `File`, select `New > Solution with Empty Station`.
2. Give a name to the solution, e.g. `yumi_egm`, and click `Create`.
3. Go to tab `Controller`, and under `Configuration`, select `Installation Manager > Installation Manager 6`.
4. Go to tab `Controllers > Virtual`. Click on the drop-down arrow to see `Virtual Controllers` menu. 
5. Click on `+`. This will add a menu to `Create New virtual controller`.
6. Name the controller, e.g. `yumi_ctrl`, and click `Next`.
7. In the `Added Product(s)` window, click `Add`, and select the following products:
    - RobotWare
    - StateMachine
    - (Optional) SmartGripper
8. Click Next, where you’ll see the Added Licences. You do not need to do anything here and can click Next again, to go to the System Options menu.
9. Under Engineering Tools, select **689-1 Externally Guided Motion (EGM)**. 
10. Under Communication, select **617-1 FlexPendant Interface**
11. In Drive Modules, under Robot  select **IRB 14000  (Dual arm YuMi)** > IRB 14000-0.5/0.5 .
12. Select **IRB 14000-0.5/0.5** for both Left and Right Arm configuration.
13. In Applications, under StateMachine Category > StateMachine Group, select **StateMachine Watchdog**.
14. Click Next to see the summary of the new virtual controller, and finally click Apply. You can close the Installation Manager window now.
15. Go back to the Home tab and under Build Station, select Virtual Controller > Existing Controller… , which will open the window to Add Existing Controller.
16. Select the controller you have just created and click OK. The controller may take a while to start.
17. After clicking on OK, you will be prompted to Select library for ‘14000_05_05 (ROB_R, ROB_R_7)’, where you should select IRB14000_0.5_0.5__01. 
18. The same prompt will show up for the left arm ‘14000_05_05 (ROB_L, ROB_L_7)’ and you should select the same as above.
19. You should now see the YuMi in View1.

20. (Optional) Adding grippers:
    1. In Home tab, under Build Station go to Import Library > Equipment > Tools and select **ABB Smart Gripper**.
    2. You will be prompted to select the type of Smart Gripper. At the lab we have the following grippers:
        - 1 Servo, Fingers
        - 1 Servo, One Vacuum Cup, Fingers
    3. On the Layout side menu, drag and drop the newly created mechanism into the appropriate arm, i.e. IR14000_0.5_0.5_R or R14000_0.5_0.5_L.
    4. Click Yes, when asked if you want to update the position, and repeat the whole process for the other gripper.

21. You can now check the communication with RobotStudio by going to a browser and typing in: 127.0.0.1 , i.e. the localhost IP.
22. This should prompt you to sign in to ABB’s API:
    - user: Default User
    - pass: robotics
If successful, you should see 6 vertical dots printed on the browser. However, in order to be able to connect from the Ubuntu side there are still a few steps that need to be done. This means that, at this point if you type the IP address of the Windows computer on your Ubuntu’s browser, you won’t see the same behaviour yet.

23. Going back to the Controller tab in RobotStudio, under Configuration select Configuration > Communication. Go to Transmission Protocol type, and you should see a table.
24. Change the Remote Address field for both ROB_L and ROB_R, by writing the IP address of the Ubuntu computer. Make sure you do not change anything else and close the window. You may need to restart the controller.
25. Try to log in to the ABB API from the Ubuntu side. If you see the following error: RAPID Undefined Error, then follow the steps on the next slide and try again.

Open a File Explorer window and navigate to %appdata% > Roaming > ABB Industrial IT > Robotics IT > RobVC
Either modify or create a file named: vcconf.xml with the following content:

```
<?xml version="1.0" encoding="UTF-8"?>
<VCConfiguration>
    <RemoteVCConfiguration PublicationEnabled="true"/>
    <hosts>
        <host ip="192.168.125.3"/>
    </hosts>
</VCConfiguration>
```

Where the host ip is the IP of the Ubuntu computer.


## UBUNTU
### Prerequisite 
- Install ROS Melodic 
- Create a catkin workspace

### Create wired connection
1. Connect the two computers with ethernet cable.
2. On Ubuntu, go to Settings > Network. Click on the plus sign next to the word: Wired (+). 
3. Name the new wired connection (if you want), e.g. yumi. 
4. Go to the IPv4 tab > Select Manual option and under Addresses set the following fields as `192.168.125.3/24` the click Apply

### Install abb_robot_driver
To install the abb_robot_driver follow the steps in:
https://github.com/ros-industrial/abb_robot_driver 

### RWS and EGM Example for YuMi
Before starting the example, it is useful to read the material in:
https://github.com/ros-industrial/abb_robot_driver/tree/master/abb_egm_hardware_interface 
After this, follow the steps as described here:
https://github.com/ros-industrial/abb_robot_driver/tree/master/abb_robot_bringup_examples#example-3-ex3_rws_and_egm_yumi_robot 
When following the instructions in the link above you should see the robot moving in RobotStudio. 

### Impactful EGM RAPID arguments
- The \MaxSpeedDeviation argument of EGMActJoint in deg/sec (e.g. limits EGM references on the robot controller side).
- The \PosCorrGain argument of EGMRunJoint (e.g. needs to be 0 for pure velocity control).
These relate to the RAPID files (robot code) in the simulation and also the robot later on. You can use the driver to change them, but you can also view them in RobotStudio in the RAPID tab. On the left side panel with the Controller there will be a file under RAPID > T_ROB_L > TRobEGM, where you can find the constants:
- DEFAULT_MAX_SPEED_DEVIATION
- DEFAULT_POSITION_CORR_GAIN
You can change these depending on what you want to do. The same file needs to be modified for the other arm, T_ROB_R. Note that to save changes to RAPID files you cannot use ctrl+s, and have to click on Apply > Apply changes. You may need to restart the controller again after this.

- The \Cond_min_max argument relates to the \J1-\J7 arguments of the EGMActJoint instruction. It provides the convergence criteria for joint 1 to 6 in degrees for 6-axis robots, and joint 1 to 7, in degrees for 7-axis robots. The default value is ±0.5 degrees. The convergence criteria data is used to decide if the robot has reached the ordered joint positions. If the difference between the ordered joint position and the actual joint position is within this range, the joint is regarded to have reached its ordered position. If no convergence criteria is specified for a joint, that was selected in EGMRunJoint, the default value is used.
- As soon as all joints that were specified in EGMRunJoint have reached their ordered positions, the robot itself has reached its ordered position and RAPID execution continues with the next RAPID instruction.

- The \CondTime argument relates to the time in seconds that the convergence criteria defined in EGMActJoint has to be fulfilled before the target point is considered to be reached and EGMRunJoint releases RAPID execution to continue to the next instruction. The default is 1 second.
- This argument is set in the EGMRunJoint instruction.


## REAL ROBOT
When you move to the real robot, instead of connecting to your Windows computer, you will connect to the robot directly through the Ethernet cable. 
You will also have to change the IP addresses:
- Use the YuMi's FlexPendant to change the IP address of the two arms (i.e. ROB_R and ROB_L) to your Ubuntu's IP. This is done in in Controller Options, found in the FlexPendant’s Main Menu > Control Panel. Look for Communication > Transmission Protocol, in the tab next to Settings. 
- When calling the driver node, you should use the robot's IP: 192.168.125.1
- Sometimes when using the real robot, the grippers may not be working right away.
- Follow the instructions on the right to fix it. You may need to repeat the process and restart the controller.

- Before using the ROS Driver to control the robot, make sure to put the YuMi in Automatic Mode. To do this, go to the Quickset Menu and click on the Operator Panel.
- Here, you should also turn On the Motors.

- If you want to teleoperate the YuMi, switch to Manual Mode and go to the Main Menu. To control the grippers, select Smart Gripper, and to control the arms, select Jogging. For more details, read through ABB’s documentation.
- If you wish to manually move the arms, there are two buttons on the belly of the YuMi which will release the joint locks and allow you to reposition them. Make sure to be in Manual Mode, with motors turned Off and that you are holding the arm firmly before pressing the button!
- If you want to test your own controllers, make sure to always have your finger over the Emergency Stop button found on the FlexPendant. If something is about to go wrong, press it all the way down.
- If something went wrong and you pressed the Emergency Stop button, you can reset it by rotating it.
