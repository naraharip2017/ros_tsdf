# ROS_TSDF

ROS_TSDF is a library for integrating lidar scans into a Truncated Signed Distance Field (TSDF)

A distance field is a map where each pixel/voxel contains the distance to the nearest surface

A signed distance field is where pixels/voxels outside of objects have positive values and those inside have negative values

A truncated signed distance field is one which only voxels up to a certain distance on the exterior and interior of objects save their distance data

![](https://github.com/naraharip2017/ros_tsdf/blob/main/docs/images/TSDF.JPG)

[Source](https://www.researchgate.net/publication/276120886_Real-time_large-scale_dense_RGB-D_SLAM_with_volumetric_fusion)

The library runs in ROS2 and is CUDA Accelerated

There are several customizable parameters including voxel size and truncation distance to meet the needs of your project

Find documentation on the project [here](https://github.com/naraharip2017/ros_tsdf/wiki)

## Build
* Eigen >= 3.3.7

http://eigen.tuxfamily.org/index.php?title=Main_Page

http://eigen.tuxfamily.org/dox/GettingStarted.html

* Nvidia GPU that supports dynamic parallelism. The project has been tested on an RTX 2080 Super and Jetson Xavier
* CUDA >= 10.2
* CMake >= 3.5
* ROS2 Dashing

Create a ros2 workspace and clone the repo into the src directory

From the workspace directory run one of the following commands

**Debug Mode**
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8

If there is an initial build error related to dynamic parallelism, attempt to build once more

# Results

Video

### Benchmarking

The following tables provide benchmarking results for the library on an RTX 2080 Super and Jetson Xavier.

Note: The transformation of the lidar position to the point cloud frame took 10ms on average.

Data Set 1 had 371 point cloud scans over 92.75s 

Data Set 2 had 497 point cloud scans over 123.981s

Data Set 3 had 497 point cloud scans over 124.124s 

Each point cloud scan had roughly 1000 lidar points.

**GPU: RTX 2080 Super**

|Voxel Size (m) | Truncation Distance (m) | Data Set 1 Avg Integration Time (ms)|Data Set 2 Avg Integration Time (ms) | Data Set 3 Avg Integration Time (ms)| Overall Avg (ms)|
|:----:|:----:|:----:|:----:|:----:|:----:|
|0.5        | 0.1                 |20.4676             |21.6008                |21.0504         |21.0396 |
|0.5        | 2.0                 |28.4108             |29.4315                |28.9294         |28.9293 |
|0.5        | 4.0                 |34.4919             |36.494                 |35.4315         |35.4725 |
|1.0        | 0.1                 |19.0189             |19.7515                |19.5806         |19.4503 |
|1.0        | 2.0                 |20.5                |21.0909                |20.8407         |20.8105 |
|1.0        | 4.0                 |22.6784             |23.598                 |23.1028         |23.1236 |

**GPU: Jetson Xavier**

|Voxel Size (m) | Truncation Distance (m) | Data Set 1 Avg Integration Time (ms)|Data Set 2 Avg Integration Time (ms) | Data Set 3 Avg Integration Time (ms)| Overall Avg (ms)|
|:----:|:----:|:----:|:----:|:----:|:----:|
|0.5        | 0.1                 |45.9946             |48.8869                 |48.6016       |47.8277 |
|0.5        | 2.0                 |59.3693             |60.6141                 |59.7384       |59.9013 |
|0.5        | 4.0                 |73.2838             |74.9032                 |72.3448       |73.5106 |
|1.0        | 0.1                 |41.5405             |42.6351                 |42.6351       |42.2702 |
|1.0        | 2.0                 |44.4394             |46.7287                 |45.7399       |45.6360 |
|1.0        | 4.0                 |51.1432             |51.7717                 |52.5855       |51.8334 |


|Voxel Size (m) | Truncation Distance (m) |Data Set 1 Voxel Blocks Allocated| Data Set 2 Voxel Blocks Allocated|Data Set 3 Voxel Blocks Allocated |
|:----:|:----:|:----:|:----:|:----:|
|0.5        | 0.1                 |6676       |11110             |11849            |
|0.5        | 2.0                 |10197      |16778             |17269            |
|0.5        | 4.0                 |11840      |19386             |20217            |
|1.0        | 0.1                 |2156       |3592              |3749             |
|1.0        | 2.0                 |2833       |4707              |4652             |
|1.0        | 4.0                 |3080       |5083              |5122             |

Parameters: Besides voxel size and truncation distance the other parameters were held constant.

* Max Weight = 10000
* Publish Distance Squared = 2500.0
* Garbage Collection Distance Squared = 250000.0

Constants: The size of the block heap and hash table was kept constant as well.

* Voxels Per Side = 8
* Hash Table Num Buckets = 1,000,000
* Hash Table Hash Entries Per Bucket = 2
* Num Heap Blocks = 50000

In all these tests the voxel block hash table can store up to 2,000,000 hash entries and utilizes 44MB of space.

The block heap was set to hold 50000 blocks, which is much larger than any of these data sets needed and it utilized 307MB of space. No garbage collection occurred. So setting the size of the block heap to a sufficient level for your project and utilizing garbage collection can significantly decrease the memory necessary for allocating the block heap.


# How to Use
For an example use of the library, please see the following project that has integrated the TSDF into an autonomous cinematographer drone: https://github.com/nightduck/AirSim

[Example Node](https://github.com/nightduck/AirSim/blob/master/ros2/src/cinematography/src/mapping.cpp) 

[Example Launch File](https://github.com/nightduck/AirSim/blob/master/ros2/src/cinematography/launch/mapping_pipeline.launch.py)

Run the Launch File using the following:

```
ros2 launch cinematography mapping_pipeline.launch.py
```

[Example Visualization](https://github.com/nightduck/AirSim/blob/master/ros2/src/cinematography/src/debug_viz.cpp)


The debug node provides code for creating a sample visualization of the tsdf produced by ros_tsdf in RVIZ. The file has a subscriber for the tsdf with callback fetchTSDF. Delta timing is implemented so the tsdf renders only every 5s. 

There are two markers that are rendered. Green indicates >=0 sdf value for the voxel and red indicates <0 sdf.

### Important Parameter Info

When changing parameters there are two constants that may need to be changed within the package based off the value of the parameters and rebuilt. 

**NUM_HEAP_BLOCKS** in tsdf_container.cuh which is dependent on voxel size, truncation distance, garbage collection distance parameters, and the voxels per side constant in the same file. If print statements are outputting the num voxel blocks allocated will be given and if the block heap overflows an error statement will be given

**PUBLISH_VOXELS_MAX_SIZE** in tsdf_handler.cuh is dependent on voxel size, truncation distance, and publishing distance parameters. If print statements are outputting the number of voxels published per lidar scan is given and if more voxels can be published than this constanst an error statement is given


