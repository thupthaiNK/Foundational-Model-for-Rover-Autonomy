cmake_minimum_required(VERSION 3.8)
project(exomy_ros2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(exomy_ros2_msgs REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install executable Python scripts
install(PROGRAMS
  exomy_ros2/motor_node.py
  exomy_ros2/robot_node.py
  exomy_ros2/cmd_vel_to_rovercommand_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
