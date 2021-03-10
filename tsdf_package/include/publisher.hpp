#ifndef _PUBLISHER_HPP
#define _PUBLISHER_HPP
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include "tsdf_package_msgs/msg/tsdf.hpp"
#include "tsdf_package_msgs/msg/voxel.hpp"
#include "cuda/tsdf_container.cuh"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

class Publisher{
public:
    Publisher(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tsdf_occupied_voxels_pub, 
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tsdf_occupied_voxels_pc_pub,
        rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, 
        bool & visualize_published_voxels, float & publish_distance_squared, float & truncation_distance,
        float & voxel_size, rclcpp::Clock::SharedPtr clock, Vector3f * occupied_voxels, Voxel * sdf_weight_voxel_vals);
    
    void publish(int & num_voxels);
private:
    inline bool withinPublishDistance(Vector3f & a, Vector3f & b);
    void publishWithVisualization(int & num_voxels);

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tsdf_occupied_voxels_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tsdf_occupied_voxels_pc_pub;
    rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub; 
    bool visualize_published_voxels;
    float publish_distance_squared;
    float truncation_distance;
    float voxel_size;
    rclcpp::Clock::SharedPtr clock;  
    Vector3f * occupied_voxels;
    Voxel *  sdf_weight_voxel_vals;
    visualization_msgs::msg::MarkerArray marker_array;
    std::chrono::_V2::system_clock::time_point last_time;
};

#endif