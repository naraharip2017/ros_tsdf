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

const std::string target_frame = "front_left_custom_body";
rclcpp::Clock::SharedPtr clock_;
tf2_ros::Buffer* tfBuffer;
tf2_ros::TransformListener* tfListener;

extern void pointcloudMain(std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > points);

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
      std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > points = temp_cloud->points;
      pointcloudMain(points); //I can do this transformation myself if I can get the transformation matrix, then in parallel carry out the transformation of the pc

      /*
      This will convert the point cloud, then we can transfer to pointcloudMain. There we will find unique voxel block points for every point in pc..then
      call the tsdf.cu for each voxel block. So I can return list of previous points that gets called 
      */
  }

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

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
  
  //create hash table and everything here which is defined and implemented in tsdf_node.cuh and tsdf.cu. Then pass the table to pointCloudMain where point clouds are handled. Inside the class we hold all variables

      // Point point_h[size];
    // Point * A = new Point(1,1,1);
    // Point * B = new Point(5,5,5);
    // Point * C = new Point(9,9,9);
    // Point * D = new Point(13,13,13);
    // Point * E = new Point(4,4,4);
    // Point * F = new Point(12,12,12);
    // point_h[0] = *A;
    // point_h[1] = *D;

    // for(int i=1; i<=size; ++i){
    //   Point * p = new Point(i,i,i);
    //   point_h[i-1] = *p;
    // }

    //     for(int i=1; i<=size; ++i){
    //   Point * p = new Point(i+4,i+4,i+4);
    //   point_h[i-1] = *p;
    // }

  Point point_h[2];
  Point * A = new Point(1,1,1);
  Point * B = new Point(2,2,2);
  point_h[0] = *A;
  point_h[1] = *B;
  TsdfHandler * tsdfHandler = new TsdfHandler();
  tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 2);
  return 0;
}