cmake_minimum_required(VERSION 3.12)
project(yumift_msgs)

add_compile_options(-std=c++17)

find_package(
  catkin 
  REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES
  YumiKinematics.msg
  YumiPosture.msg
  YumiTrajectory.msg
  Jacobian.msg
  RobotState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  std_msgs
  sensor_msgs
  geometry_msgs
)
