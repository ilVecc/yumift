cmake_minimum_required(VERSION 3.12)
project(yumi_description)

find_package(catkin REQUIRED)

catkin_package()

#if(CATKIN_ENABLE_TESTING)
#  find_package(roslaunch REQUIRED)
#  roslaunch_add_file_check(launch)
#endif()

install(
  DIRECTORY launch meshes rviz urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
