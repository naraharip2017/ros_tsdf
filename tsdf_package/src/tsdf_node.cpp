#include <memory>
#include <pcl_conversions/pcl_conversions.h>
#include <cstdio>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include "rclcpp/rclcpp.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "std_msgs/msg/string.hpp"
#include "transformPC.hpp"
#include "tsdf_handler.cuh"
#include "pointcloud.cuh"
#include <iostream>

const std::string target_frame = "front_left_custom_body";
rclcpp::Clock::SharedPtr clock_;
tf2_ros::Buffer* tfBuffer;
tf2_ros::TransformListener* tfListener;

typedef Eigen::Matrix<float, 3, 1> Vector3f;

extern void pointcloudMain( pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud);
extern void testVoxelBlockTraversal();

// float FloorFun(float x, float scale){
//   return floor(x*scale) / scale;
// }

// void GetVoxelBlockCenterFromPoint(Vector3f point){
//   float voxelBlock_size = VOXEL_SIZE * VOXEL_PER_BLOCK;
//   float halfVoxelBlock_size = voxelBlock_size/2;
//   float scale = 1 / voxelBlock_size;
//   Vector3f blockCenter;
//   blockCenter(0) = FloorFun(point(0), scale) + halfVoxelBlock_size;
//   blockCenter(1) = FloorFun(point(1), scale) + halfVoxelBlock_size;
//   blockCenter(2) = FloorFun(point(2), scale) + halfVoxelBlock_size;
//   std::cout << blockCenter;
//   std::cout << "\n";
// }

void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // sensor_msgs::msg::PointCloud2::SharedPtr target_frame_msg;
    // transformPointCloud(target_frame, *msg.get(), *target_frame_msg.get(), *tfBuffer);
    // pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    // pcl::fromROSMsg(*msg, pointcloud_pcl);
    //target frame, target time, fixed frame, tf listener, cloud in/out
    //convert point cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr converted_temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PCLHeader header = temp_cloud->header;  
    // std::string frame_id = header.frame_id;
    // pcl::uint64_t stamp = header.stamp;

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    pointcloudMain(temp_cloud);
   //I can do this transformation myself if I can get the transformation matrix, then in parallel carry out the transformation of the pc
    //printf("%lu\n", temp_cloud->size());
    // Eigen::Vector4f sensor_origin = temp_cloud->sensor_origin_;
    // std::cout << "Here is the vector v:\n" << sensor_origin << std::endl;
    /*
    This will convert the point cloud, then we can transfer to pointcloudMain. There we will find unique voxel block points for every point in pc..then
    call the tsdf.cu for each voxel block. So I can return list of previous points that gets called 
    */
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Vector3f test(-.75,.6,.3);

  // std::cout << test;

  // std::cout << "\n";

  // GetVoxelBlockCenterFromPoint(test);

  testVoxelBlockTraversal();


  // clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  // tfBuffer = new tf2_ros::Buffer(clock_);
  // tfListener = new tf2_ros::TransformListener(*tfBuffer);

  // // Transformer *transformer = new Transformer();

  // auto node = rclcpp::Node::make_shared("my_subscriber");

  // auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
  //   "/airsim_node/drone_1/lidar/LidarCustom", 1, callback
  // ); //todo: should it be 1? might be .1 check publishing rate in airsim but the mapping pipeline runs at 2hz?

  // rclcpp::spin(node);

  // rclcpp::shutdown();
  
  // //create hash table and everything here which is defined and implemented in tsdf_node.cuh and tsdf.cu. Then pass the table to pointCloudMain where point clouds are handled. Inside the class we hold all variables

  // int size= 2;
  //     Point point_h[size];
  //   Point * A = new Point(1,1,1);
  //   Point * B = new Point(5,5,5);
  //   Point * C = new Point(9,9,9);
  //   Point * D = new Point(9,9,9);
  //   // Point * E = new Point(4,4,4);
  //   // Point * F = new Point(12,12,12);
  //   point_h[0] = *A;
  //   point_h[1] = *B;
    

  //   // for(int i=1; i<=size; ++i){
  //   //   Point * p = new Point(i,i,i);
  //   //   point_h[i-1] = *p;
  //   // }

  //   //     for(int i=1; i<=size; ++i){
  //   //   Point * p = new Point(i+4,i+4,i+4);
  //   //   point_h[i-1] = *p;
  //   // }

  // // Point point_h[2];
  // // Point * A = new Point(1,1,1);
  // // Point * B = new Point(2,2,2);
  // // point_h[0] = *A;
  // // point_h[1] = *B;
  // TsdfHandler * tsdfHandler = new TsdfHandler();

  // //addPoints
  
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 2);
  //   Point * point_h2;
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  // point_h[0] = *C;
  // point_h[1] = *D;

  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 2);

  // // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  // //tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  

  // //add unallocated points first test with linked list etc
  // //change lidar to 300000 points and see size of pc per frame
  return 0;
}