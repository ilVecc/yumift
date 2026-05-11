cmake_minimum_required(VERSION 3.12)
project(sleipner_controllers)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/example_controller.py
  scripts/trajectory_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
