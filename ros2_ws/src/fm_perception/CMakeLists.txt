cmake_minimum_required(VERSION 3.8)
project(fm_perception)

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
  fm_perception/clip_terrain_node.py
  fm_perception/dinov2_terrain_node.py
  fm_perception/dinov2_attention_node.py
  fm_perception/generic_terrain_node.py
  fm_perception/terrain_controller_node.py
  fm_perception/safety_watchdog_node.py
  fm_perception/lidar_proximity_guard_node.py
  fm_perception/reactive_explorer_node.py
  fm_perception/stuck_detection_node.py
  fm_perception/real_stuck_detection_node.py
  fm_perception/blip2_scene_node.py
  fm_perception/smolvlm_terrain_node.py
  fm_perception/cascade_pipeline.py
  fm_perception/traversability_viz_node.py
  fm_perception/static_traversability_costmap_node.py
  fm_perception/live_traversability_costmap_node.py
  fm_perception/cmd_vel_relay_node.py
  fm_perception/l5_lite_planner_node.py
  fm_perception/camera_preview_node.py
  fm_perception/sensor_overlay_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install RViz2 configurations
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
