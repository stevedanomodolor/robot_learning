cmake_minimum_required(VERSION 2.8.3)
project(ball_shooter_sim)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
)

# Depend on system install of Gazebo
find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

# For tc_simple_world_plugin plugin
add_library(ball_launcher_plugin src/ball_launcher_plugin.cpp)
target_link_libraries(ball_launcher_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})



catkin_package(
  DEPENDS
    roscpp
    gazebo_ros
)
