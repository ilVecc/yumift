# Wrenched YuMi URDF

The URDF file is from https://github.com/kth-ros-pkg/yumi/tree/egm_modifications/yumi_description. 

Generate URDF file using
```
xacro yumi.urdf.xacro yumi_setup:=robot_centric arms_interface:=VelocityJointInterface grippers_interface:=EffortJointInterface > yumi.urdf
```

Load the URDF with
```
roslaunch yumi_description load.launch
```

Visualize the URDF in Rviz with
```
roslaunch yumi_description view.launch
```
