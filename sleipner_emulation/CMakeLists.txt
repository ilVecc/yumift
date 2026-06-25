cmake_minimum_required(VERSION 3.12)
project(sleipner_emulation)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sleipner_emulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
