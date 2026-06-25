cmake_minimum_required(VERSION 3.12)
project(yumift_emulation)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/yumi_emulator.py
  scripts/wrench_emulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
