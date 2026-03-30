cmake_minimum_required(VERSION 3.12)
project(yumift_controllers)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/dummy_controller.py
  scripts/tracking_controllers.py
  scripts/trajectory_controllers.py
  scripts/routine_calib.py
  scripts/routine_ready.py
  scripts/ftsensor_calibration.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
