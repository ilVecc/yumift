cmake_minimum_required(VERSION 3.12)
project(kentaur_controllers)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/whole_body_compliant_controller.py
  scripts/whole_body_controller.py
  scripts/trajectory.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
