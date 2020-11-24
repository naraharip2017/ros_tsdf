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
#include "params.hpp"

__constant__
const int OCCUPIED_VOXELS_SIZE = 100000; //if holes appearing in visualization, increase this value

__constant__
const int MAX_LINKED_LIST_BLOCKS = 1000;

__device__ float VOXEL_SIZE; //param
__device__ float HALF_VOXEL_SIZE;
__device__ float VOXEL_BLOCK_SIZE; // = VOXELS_PER_SIDE * VOXEL_SIZE
__device__ float HALF_VOXEL_BLOCK_SIZE;
__device__ float BLOCK_EPSILON; //used for determining if floating point values are equal when comparing block positions
__device__ float VOXEL_EPSILON; //used for determining if floating point values are equal when comparing voxel positions
__device__ float TRUNCATION_DISTANCE; //param
__device__ float MAX_WEIGHT; //param
__device__ float PUBLISH_DISTANCE_SQUARED; //distance squared of publishing around drone
__device__ float GARBAGE_COLLECT_DISTANCE_SQUARED; //distance squared from drone to delete voxel blocks

class TSDFHandler{
public:
    __host__
    TSDFHandler();

    __host__
    ~TSDFHandler();

    __host__
    void processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h);
    
    __host__
    void allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, int pointcloud_size, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void getVoxelBlocks(int num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * pointcloud_voxel_blocks_d, int * pointcloud_voxel_blocks_d_index, Vector3f * origin_transformed_d, int * pointcloud_size_d);

    __host__
    void integrateVoxelBlockPointsIntoHashTable(Vector3f * points_d, int * pointcloud_voxel_blocks_d_index, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void updateVoxels(int & num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void publishOccupiedVoxels(Vector3f * origin_transformed_d, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h, HashTable * hash_table_d, BlockHeap * block_heap_d);

    __host__
    void garbageCollectDistantBlocks(Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d);
private:
    TSDFContainer * tsdfContainer; 
    Vector3f * occupied_voxels_d;
};

void initializeGlobalVars(Params params);

#endif