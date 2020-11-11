#ifndef _POINTCLOUD_CUH_
#define _POINTCLOUD_CUH_
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
const float truncation_distance = 0.1;

__constant__
const float MAX_WEIGHT = 10000.0;

class Handler{
public:
    __host__
    Handler();

    __host__
    void processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, Vector3f * occupiedVoxels_h, int * index_h);

    __host__
    void integrateVoxelBlockPointsIntoHashTable(Vector3f points_h[], int size, HashTable * hashTable_d, BlockHeap * blockHeap_d);
    
private:
    TSDFContainer * tsdfContainer; 
};

#endif