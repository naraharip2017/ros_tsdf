#ifndef _TRANSFORMER_HPP
#define _TRANSFORMER_HPP
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include "rclcpp/rclcpp.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "std_msgs/msg/string.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

typedef Eigen::Matrix<float, 3, 1> Vector3f;

class Transformer{
public:
    Transformer(std::string source_frame, rclcpp::Clock::SharedPtr clock_);
    void getOriginInPointCloudFrame(const sensor_msgs::msg::PointCloud2 & in, Vector3f & origin_transformed);
    void convertPointCloud2ToPointCloudXYZ(sensor_msgs::msg::PointCloud2::SharedPtr msg, pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud);
private:
    tf2_ros::Buffer* tf_buffer;
    tf2_ros::TransformListener* tf_listener;
    geometry_msgs::msg::PointStamped origin_in;
    std::string source_frame;
};

#endif