cmake_minimum_required(VERSION 3.5)
project(tsdf_package)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

## DEBUG
# set this to path to AirSim root folder if you want your catkin workspace in a custom directory
set(AIRSIM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../AirSim/)
set (common_utility_src ${AIRSIM_ROOT}/ros/src/common)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)
## DEBUG

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
FIND_PACKAGE(CUDA REQUIRED)
find_package(PCL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tsdf_package_msgs REQUIRED)

include_directories(
  include
  include/cuda
)

SET(CUDA_SEPARABLE_COMPILATION ON)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
SET(CUDA_NVCC_FLAGS ${ARCH_FLAGS} CACHE STRING "nvcc flags" FORCE)

cuda_add_library(${PROJECT_NAME}_cuda SHARED
  src/cuda/tsdf_handler.cu
  src/cuda/tsdf_container.cu
  OPTIONS -g -shared -Xcompiler -fPIC -rdc=true #-Xptxas=-v
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME}_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

ament_target_dependencies(${PROJECT_NAME}_cuda 
  "pcl_conversions"
  "pcl"
)

add_library(tsdf_node 
  src/transformer.cpp
  src/publisher.cpp
)

ament_export_interfaces(export_tsdf_node HAS_LIBRARY_TARGET)

ament_export_dependencies(
  "CUDA"
  "rclcpp"
  "sensor_msgs"
  "pcl_conversions"
  "std_msgs"
  "PCL"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
  "tsdf_package_msgs"
)

ament_target_dependencies(tsdf_node 
  "CUDA"
  "rclcpp"
  "sensor_msgs"
  "pcl_conversions"
  "std_msgs"
  "pcl"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
  "tsdf_package_msgs"
)

target_link_libraries(tsdf_node 
  ${PROJECT_NAME}_cuda
  ${Boost_SYSTEM_LIBRARY}  # pcl_conversions requires this
  ${PCL_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
)

###### DEBUGGING
include_directories(
        include
        ${AIRSIM_ROOT}/AirLib/deps/eigen3
        ${AIRSIM_ROOT}/AirLib/include
        ${RPC_LIB_INCLUDES}
        ${AIRSIM_ROOT}/MavLinkCom/include
        ${AIRSIM_ROOT}/MavLinkCom/common_utils
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${common_utility_src}
)
add_executable(test_node src/test.cpp)
target_link_libraries(test_node AirLib tsdf_node)
### DEBUG

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  TARGETS tsdf_node ${PROJECT_NAME}_cuda test_node ## DEBUGGING: Remove last node when done
  EXPORT export_tsdf_node
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
  )
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/)

ament_export_include_directories(include)



ament_package()
