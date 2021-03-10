# ROS_TSDF

ROS_TSDF is a library for integrating lidar scans into a Truncated Signed Distance Field (TSDF)

A distance field is a map where each pixel/voxel contains the distance to the nearest surface

A signed distance field is where pixels/voxels outside of objects have positive values and those inside have negative values

A truncated signed distance field is one which only voxels up to a certain distance on the exterior and interior of objects save their distance data

The library runs in ROS2 and is CUDA Accelerated

There are several customizable parameters including voxel size and truncation distance to meet the needs of your project

For the overarching implementation of the project please see the following wiki: 

For a deep dive into the code for modification or to contribute please see the following wiki: 

## Build
* Eigen >= 3.3.7

http://eigen.tuxfamily.org/index.php?title=Main_Page

http://eigen.tuxfamily.org/dox/GettingStarted.html

* Nvidia GPU that supports dynamic parallelism. The project has been tested on an RTX 2080 Super and Jetson Xavier
* CUDA >= 10.2
* CMake >= 3.5

# Results
Video

Timing

# How to Use
For an example use of the library, please see the following project that has integrated the TSDF into an autonomous cinematographer drone: https://github.com/nightduck/AirSim

For an example node utilizing the library please see the mapping.cpp file and the corresponding mapping_pipeline.launch.py launch file
The debug node in the cinematography package also provides an example visualization tool for a generated TSDF in RVIZ.
