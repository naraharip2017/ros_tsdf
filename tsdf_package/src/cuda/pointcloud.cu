#include <stdio.h>

#include "pointcloud.cuh"
#include <memory>
#include <cstdio>
#include <pcl/point_types.h>

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

void pointcloudMain(std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > points)
{
    for(size_t i=0; i<points.size(); ++i){
        printf("Cloud with Points: %f, %f, %f\n", points[i].x,points[i].y,points[i].z);
      } 
}