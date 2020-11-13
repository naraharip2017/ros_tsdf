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

typedef Eigen::Matrix<float, 3, 1> Vector3f;

__constant__
const float truncation_distance = .1;

__constant__
const float MAX_WEIGHT = 10000.0;

__constant__
const int OCCUPIED_VOXELS_SIZE = 200000;

__constant__
const float VISUALIZE_DISTANCE_SQUARED = 250000;

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
    void visualize(Vector3f * origin_transformed_d, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h, HashTable * hash_table_d, BlockHeap * block_heap_d);

private:
    TSDFContainer * tsdfContainer; 
    Vector3f * occupied_voxels_d;
};

#endif