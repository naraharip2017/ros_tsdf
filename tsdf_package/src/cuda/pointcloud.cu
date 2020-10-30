#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "pointcloud.cuh"
#include "tsdf_handler.cuh"
// #include <memory>
// #include <cstdio>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <math.h>

typedef Eigen::Matrix<float, 3, 1> Vector3f;

//rename to pointcloud_handler

__device__
size_t retrieveHash(Vector3f point){ //tested using int can get negatives
  return (((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS;
}

__device__ 
float FloorFun(float x, float scale){
  return floor(x*scale) / scale;
}

__device__
Vector3f GetVoxelBlockCenterFromPoint(Vector3f point){
  float scale = 1 / VOXEL_BLOCK_SIZE;
  Vector3f blockCenter;
  blockCenter(0) = FloorFun(point(0), scale) + HALF_VOXEL_BLOCK_SIZE;
  blockCenter(1) = FloorFun(point(1), scale) + HALF_VOXEL_BLOCK_SIZE;
  blockCenter(2) = FloorFun(point(2), scale) + HALF_VOXEL_BLOCK_SIZE;
  return blockCenter;
}

__device__
bool checkFloatingPointVectorsEqual(Vector3f A, Vector3f B){
  Vector3f diff = A-B;
  if((fabs(diff(0)) < EPSILON) && (fabs(diff(1)) < EPSILON) && (fabs(diff(2)) < EPSILON))
    return true;

  return false;
}

 __global__
 void getVoxelBlocksForPoint(pcl::PointXYZ * points_d){
  int threadIndex = threadIdx.x;
  pcl::PointXYZ point_d = points_d[threadIndex];
  // Point * origin = new Point(0,0,0); //make these vectors?
  // Point * Point_points_d = new Point(point_d.x, point_d.y, point_d.z); //converts the float to int
  // Point * direction = *Point_points_d - *origin;
  Vector3f u(210,568,123);
  Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
  Vector3f v = point_d_vector - u; //direction
  printf("V: (%f, %f, %f)\n", v(0), v(1), v(2));
  //equation of line is u+tv
  float vMag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / vMag;
  float truncation_distance = 2.0;
  Vector3f truncation_start = point_d_vector - truncation_distance*v_normalized;
  printf("Truncation start : (%f, %f, %f)\n", truncation_start(0), truncation_start(1), truncation_start(2));
  
  Vector3f truncation_end = point_d_vector + truncation_distance*v_normalized;  //get voxel block of this and then traverse through voxel blocks till it equals this one
  printf("Truncation end : (%f, %f, %f)\n", truncation_end(0), truncation_end(1), truncation_end(2));

  float distance_tStart_origin = sqrt(pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2));
  float distance_tEnd_origin = sqrt(pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2));

  if(distance_tEnd_origin < distance_tStart_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }

  Vector3f truncation_start_block = GetVoxelBlockCenterFromPoint(truncation_start);
  printf("Truncation start Block: (%f, %f, %f), hashes to %lu\n", truncation_start_block(0), truncation_start_block(1), truncation_start_block(2), retrieveHash(truncation_start_block));
  // printf("point in size_t: %d, %d, %d\n", (int)truncation_start_block(0), (int)truncation_start_block(1), (int)truncation_start_block(2));
  Vector3f truncation_end_block = GetVoxelBlockCenterFromPoint(truncation_end);
  printf("Truncation end Block: (%f, %f, %f), hashes to %lu\n", truncation_end_block(0), truncation_end_block(1), truncation_end_block(2), retrieveHash(truncation_end_block));
  // printf("point in size_t: %d, %d, %d\n", (int)truncation_end_block(0), (int)truncation_end_block(1), (int)truncation_end_block(2));
  float stepX = v(0) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float stepY = v(1) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float stepZ = v(2) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float tMaxX = fabs(v(0) < 0 ? (truncation_start_block(0) - HALF_VOXEL_BLOCK_SIZE - u(0)) / v(0) : (truncation_start_block(0) + HALF_VOXEL_BLOCK_SIZE - u(0)) / v(0));
  float tMaxY = fabs(v(1) < 0 ? (truncation_start_block(1) - HALF_VOXEL_BLOCK_SIZE - u(1)) / v(1) : (truncation_start_block(1) + HALF_VOXEL_BLOCK_SIZE - u(1)) / v(1));
  float tMaxZ = fabs(v(2) < 0 ? (truncation_start_block(2) - HALF_VOXEL_BLOCK_SIZE - u(2)) / v(2) : (truncation_start_block(2) + HALF_VOXEL_BLOCK_SIZE - u(2)) / v(2));
  float tDeltaX = fabs(VOXEL_BLOCK_SIZE / v(0));
  float tDeltaY = fabs(VOXEL_BLOCK_SIZE / v(1));
  float tDeltaZ = fabs(VOXEL_BLOCK_SIZE / v(2));
  Vector3f currentBlock(truncation_start_block(0), truncation_start_block(1), truncation_start_block(2));

  do{
    //add current block to blocks in current frame list or whatever
    printf("Current Block: (%f, %f, %f), hashes to %lu\n", currentBlock(0), currentBlock(1), currentBlock(2), retrieveHash
    (currentBlock));
    // printf("point in size_t: %d, %d, %d\n", (int)currentBlock(0), (int)currentBlock(1), (int)currentBlock(2));
    if(tMaxX < tMaxY){
      if(tMaxX < tMaxZ)
      {
        currentBlock(0) += stepX;
        tMaxX += tDeltaX;
      }
      else if(tMaxX > tMaxZ){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        currentBlock(0) += stepX;
        currentBlock(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxZ += tDeltaZ;
      }
    }
    else if(tMaxX > tMaxY){
      if(tMaxY < tMaxZ){
        currentBlock(1) += stepY;
        tMaxY += tDeltaY;
      }
      else if(tMaxY > tMaxZ){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        currentBlock(1) += stepY;
        currentBlock(2) += stepZ;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    }
    else{
      if(tMaxZ < tMaxX){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else if(tMaxZ > tMaxX){
        currentBlock(0) += stepX;
        currentBlock(1) += stepY;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
      }
      else{ //can remove equals statements if want to improve on performance
        currentBlock(0) += stepX;
        currentBlock(1) += stepY;
        currentBlock(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    }       
  } while(!checkFloatingPointVectorsEqual(currentBlock, truncation_end_block));
  printf("Current Block: (%f, %f, %f), hashes to %lu\n", currentBlock(0), currentBlock(1), currentBlock(2), retrieveHash(currentBlock));
  // printf("point in size_t: %d, %d, %d\n", (int)currentBlock(0), (int)currentBlock(1), (int)currentBlock(2));
  printf("Cloud with Points: %f, %f, %f\n", points_d[threadIndex].x,points_d[threadIndex].y,points_d[threadIndex].z);
  return;
 }



void pointcloudMain(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud)
{
  //retrieve sensor origin..can use transformation from point cloud time stamp drone_1/lidar frame to drone_1 frame then transform 0,0,0
  
  std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points = pointcloud->points;

  pcl::PointXYZ * points_h = &points[0];
  pcl::PointXYZ * points_d;
  int size = pointcloud->size();
  cudaMalloc(&points_d, sizeof(*points_h)*size);
  cudaMemcpy(points_d, points_h, sizeof(*points_h)*size, cudaMemcpyHostToDevice);
  getVoxelBlocksForPoint<<<1,size>>>(points_d);


  // for(size_t i=0; i<points.size(); ++i){
  //     printf("Cloud with Points: %f, %f, %f\n", points[i].x,points[i].y,points[i].z);
  //   } 
}

void testVoxelBlockTraversal(){
  // float f = 10.23423;
  pcl::PointXYZ * point = new pcl::PointXYZ(-7.23421,-278, 576.2342);
  pcl::PointXYZ * points_h = new pcl::PointXYZ[1];
  points_h[0] = *point;
  pcl::PointXYZ * points_d;
  cudaMalloc(&points_d, sizeof(*points_h)*1);
  cudaMemcpy(points_d, points_h, sizeof(*points_h)*1, cudaMemcpyHostToDevice);
  getVoxelBlocksForPoint<<<1,1>>>(points_d);

  cudaDeviceSynchronize();
}