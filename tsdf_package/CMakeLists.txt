cmake_minimum_required(VERSION 3.5)
project(tsdf_package)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

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

#todo change to library and target link libraries
add_executable(tsdf_node 
  src/tsdf_node.cpp
  src/transformer.cpp
  src/publisher.cpp
)
target_include_directories(tsdf_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(tsdf_node 
  "visualization_msgs"
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

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
SET(CUDA_NVCC_FLAGS ${ARCH_FLAGS} CACHE STRING "nvcc flags" FORCE)

cuda_add_library(${PROJECT_NAME}_cuda SHARED
  src/cuda/tsdf_handler.cu
  src/cuda/tsdf_container.cu
  OPTIONS -g -shared -Xcompiler -fPIC -rdc=true #-Xptxas=-v
)
target_include_directories(${PROJECT_NAME}_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})

target_link_libraries(tsdf_node 
${PROJECT_NAME}_cuda
${Boost_SYSTEM_LIBRARY}  # pcl_conversions requires this
${PCL_LIBRARIES}
${tf2_LIBRARIES}
${tf2_ros_LIBRARIES}
)

ament_export_interfaces(${PROJECT_NAME}_cuda HAS_LIBRARY_TARGET)

install(TARGETS tsdf_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/)

install(
  TARGETS ${PROJECT_NAME}_cuda
  EXPORT ${PROJECT_NAME}_cuda
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

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

ament_package()
