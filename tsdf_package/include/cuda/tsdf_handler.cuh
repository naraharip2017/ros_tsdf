#ifndef _TSDF_HANDLER_CUH
#define _TSDF_HANDLER_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <math.h>
#include <assert.h>

#include "cuda/tsdf_container.cuh"

//expected max number of voxels to be published for one lidar scan. If this is set too small during processing a lidar frame a print error statement will be outputted
__constant__
const int PUBLISH_VOXELS_MAX_SIZE = 400000;

//Used for garbage collection to store at max this many blocks that are in linked lists that need to be removed sequentially.
__constant__
const int GARBAGE_LINKED_LIST_BLOCKS_MAX_SIZE = 1000; 

// Only declare these constants when being compiled in the tsdf_package library, to avoid double-declaration when linking
#ifdef TSDF_LIBRARY
__constant__ float VOXEL_SIZE; //param
__constant__ float HALF_VOXEL_SIZE;
__constant__ float VOXEL_BLOCK_SIZE; // = VOXELS_PER_SIDE * VOXEL_SIZE
__constant__ float HALF_VOXEL_BLOCK_SIZE;
__constant__ float BLOCK_EPSILON; //used for determining if floating point values are equal when comparing block positions
__constant__ float VOXEL_EPSILON; //used for determining if floating point values are equal when comparing voxel positions
__constant__ float TRUNCATION_DISTANCE; //param
__constant__ float MAX_WEIGHT; //max weight for updating voxels
__constant__ float PUBLISH_DISTANCE_SQUARED; //distance squared of publishing around drone
__constant__ float GARBAGE_COLLECT_DISTANCE_SQUARED; //distance squared from drone to delete voxel blocks

//max amount of voxels that are traversed per lidar point. Set in terms of truncation distance and voxel size
__constant__ int MAX_VOXELS_TRAVERSED_PER_LIDAR_POINT;
#endif

namespace tsdf {

class TSDFHandler{
public:
    __host__
    TSDFHandler();

    __host__
    ~TSDFHandler();

    __host__
    void processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, Vector3f * lidar_position_h, 
    Vector3f * publish_voxels_pos_h, int * publish_voxels_size_h, Voxel * publish_voxels_data_h);
    
    __host__
    void allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * lidar_points_d, Vector3f * lidar_position_d, int * lidar_points_size_d, 
    int point_cloud_size_h, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void getVoxelBlocks(int num_cuda_blocks, pcl::PointXYZ * lidar_points_d, Vector3f * point_cloud_voxel_blocks_d, 
    int * point_cloud_voxel_blocks_size_d, Vector3f * lidar_position_d, int * lidar_points_size_d);

    __host__
    void allocateVoxelBlocks(Vector3f * lidar_points_d, int * point_cloud_voxel_blocks_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void getVoxelsAndUpdate(int & num_cuda_blocks, pcl::PointXYZ * lidar_points_d, Vector3f * lidar_position_d, int * 
    lidar_points_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void retrievePublishableVoxels(Vector3f * lidar_position_d, Vector3f * publish_voxels_pos_h, int * publish_voxels_size_h, Voxel * 
    publish_voxels_data_h, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void garbageCollectDistantBlocks(Vector3f * lidar_position_d, HashTable * hash_table_d, BlockHeap * block_heap_d);
private:
    TSDFContainer * tsdf_container; //object to hold the hash table and block heap for the tsdf
    Vector3f * publish_voxels_pos_d; //the array to copy back to cpu with voxels positions to publish
    Voxel * publish_voxels_data_d; //the array to copy back to cpu with voxels sdf and weight values to publish
    cudaStream_t stream; // stream to use
};

void initGlobalVars(float voxel_size_input, float truncation_distance_input, float max_weight_input, float publish_distance_squared_input, 
float garbage_collect_distance_squared_input);

}


#endif