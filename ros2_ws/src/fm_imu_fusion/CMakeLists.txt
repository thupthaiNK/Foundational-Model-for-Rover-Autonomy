cmake_minimum_required(VERSION 3.8)
project(fm_imu_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install node executables
install(PROGRAMS
  fm_imu_fusion/imu_slope_fusion_node.py
  fm_imu_fusion/icm20948_driver_node.py
  fm_imu_fusion/gyro_odom_publisher_node.py
  fm_imu_fusion/traversability_score_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
