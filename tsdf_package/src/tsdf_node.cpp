#include <memory>
#include <pcl_conversions/pcl_conversions.h>
#include <cstdio>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include "rclcpp/rclcpp.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include "std_msgs/msg/string.hpp"

extern int tsdfmain();

void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    // // pointcloud_pcl is modified below:
    // pcl::fromROSMsg(*msg, pointcloud_pcl);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
    temp_cloud->points;
    for(size_t i=0; i<temp_cloud->points.size(); ++i){
      printf("Cloud with Points: %f, %f, %f\n", temp_cloud->points[i].x,temp_cloud->points[i].y,temp_cloud->points[i].z);
    }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("my_subscriber");

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 1, callback
  ); //todo: should it be 1?

  rclcpp::spin(node);

  rclcpp::shutdown();

  //tsdfmain();
  return 0;
}