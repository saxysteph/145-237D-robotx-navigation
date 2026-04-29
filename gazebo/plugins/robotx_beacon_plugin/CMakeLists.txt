cmake_minimum_required(VERSION 3.10)
project(robotx_beacon_plugin)

find_package(gz-sim8 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(sdformat14 REQUIRED)

add_library(robotx_beacon_plugin SHARED
  src/robotx_beacon_plugin.cpp
)
add_library(robotx_bobbing_plugin SHARED
  src/robotx_bobbing_plugin.cpp
)

target_compile_features(robotx_beacon_plugin PUBLIC cxx_std_17)
target_compile_features(robotx_bobbing_plugin PUBLIC cxx_std_17)
target_include_directories(robotx_beacon_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(robotx_bobbing_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(robotx_beacon_plugin
  gz-sim8::gz-sim8
  gz-msgs10::gz-msgs10
  sdformat14::sdformat14
)
target_link_libraries(robotx_bobbing_plugin
  gz-sim8::gz-sim8
  gz-msgs10::gz-msgs10
  sdformat14::sdformat14
)

# Native CMake install layout, works on macOS/Linux without ROS/ament.
install(TARGETS robotx_beacon_plugin robotx_bobbing_plugin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
