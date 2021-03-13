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
    Publisher(
        rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, float & truncation_distance,
        float & voxel_size, Vector3f * publish_voxels_pos, Voxel * publish_voxels_data);
    
    void publish(int & publish_voxels_size);
private:
    rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub; //tsdf publisher
    float truncation_distance;
    float voxel_size;
    Vector3f * publish_voxels_pos; //reference to host side array that will be used to copy voxel positions from GPU for publishing
    Voxel *  publish_voxels_data; //reference to host sdie array that will be used to copy sdf and weight values for voxels from GPU for publishing
};

#endif