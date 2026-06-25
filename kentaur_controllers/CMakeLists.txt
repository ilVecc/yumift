cmake_minimum_required(VERSION 3.12)
project(kentaur_controllers)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/example_whole_body_controller.py
  scripts/whole_body_tracking_controller.py
  scripts/example_trajectory.py
  scripts/whole_body_trajectory_controller.py
  scripts/whole_body_compliant_controller.py
  scripts/whole_body_force_position_controller.py
  scripts/demo_whiteboard_trajectory.py
  scripts/demo_whole_body_steady_hands_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
