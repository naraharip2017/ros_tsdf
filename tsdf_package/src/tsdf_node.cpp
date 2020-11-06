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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <geometry_msgs/msg/point_stamped.hpp>
#include <iostream>


// const rclcpp::Clock::SharedPtr clock_;
// const tf2_ros::Buffer* tfBuffer;
// const tf2_ros::TransformListener tfListener;

typedef Eigen::Matrix<float, 3, 1> Vector3f;

extern void pointcloudMain( pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, TsdfHandler * tsdfHandler);
extern void testVoxelBlockTraversal(TsdfHandler * tsdfHandler);
extern void testVoxelTraversal();

const std::string target_frame = "drone_1/LidarCustom";
rclcpp::Clock::SharedPtr clock_;
tf2_ros::Buffer* tfBuffer;
tf2_ros::TransformListener* tfListener;
//set to const?
TsdfHandler * tsdfHandler;
geometry_msgs::msg::PointStamped point_in;
 

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

void getOriginInPointCloudFrame(const std::string & target_frame, const sensor_msgs::msg::PointCloud2 & in, Vector3f & origin_transformed){
  // Get the TF transform
  geometry_msgs::msg::TransformStamped transform;
  geometry_msgs::msg::PointStamped point_out;
  point_in.header = in.header;
  try { //wait for a duration
  //transform from lidar frame to world frame
    transform =
      tfBuffer->lookupTransform(
      in.header.frame_id, target_frame, tf2_ros::fromMsg(in.header.stamp), tf2::Duration(1000000000));
      tf2::doTransform(point_in, point_out, transform);
      auto point = point_out.point;
      origin_transformed(0) = point.x;
      origin_transformed(1) = point.y;
      origin_transformed(2) = point.z;
  } catch (tf2::LookupException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  } catch (tf2::ExtrapolationException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  }
  
}

void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    auto start = std::chrono::high_resolution_clock::now();
    Vector3f origin_transformed;
    Vector3f * origin_transformed_h = &origin_transformed;
    getOriginInPointCloudFrame(target_frame, *msg, origin_transformed);
    
    // printf("(%f, %f, %f)\n", origin_transformed(0), origin_transformed(1), origin_transformed(2));
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

    //get sensor origin with PointCloud2 and tsdf_handler object pass that to pointcloudMain
    // tsdfHandler.

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    pointcloudMain(temp_cloud, origin_transformed_h, tsdfHandler);
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << duration.count() << std::endl; 
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Vector3f test(-.75,.6,.3);

  // std::cout << test;

  // std::cout << "\n";

  // GetVoxelBlockCenterFromPoint(test);

  clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tfBuffer = new tf2_ros::Buffer(clock_);
  tfListener = new tf2_ros::TransformListener(*tfBuffer);
  tsdfHandler = new TsdfHandler();
  // // Vector3f point_h[1];
  // // Vector3f A(1,1,1);
  // // point_h[0] = A;
  // // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 1);

  // point_in.point.x = 0.0;
  // point_in.point.y = 0.0;
  // point_in.point.z = 0.0;
  // // Transformer *transformer = new Transformer();

  auto node = rclcpp::Node::make_shared("my_subscriber");

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 1, callback
  ); //todo: should it be 1? might be .1 check publishing rate in airsim but the mapping pipeline runs at 2hz?

  rclcpp::spin(node);

  rclcpp::shutdown();
  
  // //create hash table and everything here which is defined and implemented in tsdf_node.cuh and tsdf.cu. Then pass the table to pointCloudMain where point clouds are handled. Inside the class we hold all variables

      // int size= 64;
      // Vector3f point_h[size];
      // Vector3f A(1,1,1);
      // Vector3f B(5,5,5);
      // Vector3f C(9,9,9);
      // Vector3f D(9,9,9);
      // point_h[0] = A;
      // point_h[1] = B;
      // point_h[2] = C;
      // point_h[3] = D;
      // point_h[4] = C;
      // point_h[5] = C;
      // point_h[6] = C;
      // point_h[7] = C;

    //  point_h[0] = A;
    

    // for(int i=1; i<=size; i++){
    //   Vector3f * p = new Vector3f(i,i,i);
    //   point_h[i-1] = *p;
    // }

    //     for(int i=1; i<=size; ++i){
    //   Point * p = new Point(i+4,i+4,i+4);
    //   point_h[i-1] = *p;
    // }

  // Point point_h[2];
  // Point * A = new Point(1,1,1);
  // Point * B = new Point(2,2,2);
  // point_h[0] = *A;
  // point_h[1] = *B;
    
  //  testVoxelBlockTraversal();
  // testVoxelBlockTraversal(tsdfHandler);

  //addPoints
  
  //tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, size);
  // Vector3f * point_h2;
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  // point_h[0] = *C;
  // point_h[1] = *D;

  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 2);

  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  //tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  

  //add unallocated points first test with linked list etc
  //change lidar to 300000 points and see size of pc per frame
  return 0;
}