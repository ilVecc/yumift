cmake_minimum_required(VERSION 3.12)
project(yumift_simulation)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/simulator.py
  scripts/wrench_simulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
