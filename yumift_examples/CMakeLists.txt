cmake_minimum_required(VERSION 3.12)
project(yumift_examples)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/examples/example_1_simple_trajectory.py
  scripts/examples/example_2_reference_frames_basic.py
  scripts/examples/example_3_reference_frames_advanced.py
  scripts/examples/example_4_coordinated_mode.py
  scripts/examples/example_5_coordinated_nullspace.py
  scripts/examples/example_6_robot_introspection.py
  scripts/examples/example_wiggle.py
  scripts/postures/face_back.py
  scripts/postures/face_down.py
  scripts/postures/face_front.py
  scripts/postures/face_inside.py
  scripts/postures/face_outside.py
  scripts/postures/face_up.py
  scripts/postures/move_down.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
