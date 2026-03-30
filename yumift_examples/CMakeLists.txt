cmake_minimum_required(VERSION 3.12)
project(yumift_examples)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/examples/example_1_simple_trajectory.py
  # scripts/examples/example_2_coordinated_mode.py
  # scripts/examples/example_3_incremental_trajectory.py
  scripts/routines/face_back.py
  scripts/routines/face_down.py
  scripts/routines/face_front.py
  scripts/routines/face_inside.py
  scripts/routines/face_outside.py
  scripts/routines/face_up.py
  # scripts/routines/move_down.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
