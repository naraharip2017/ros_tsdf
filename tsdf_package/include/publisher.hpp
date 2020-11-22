#ifndef _PUBLISHER_HPP
#define _PUBLISHER_HPP
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "tsdf_package_msgs/msg/tsdf.hpp"
#include "tsdf_package_msgs/msg/voxel.hpp"
#include "cuda/tsdf_container.cuh"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

class Publisher{
public:
    Publisher(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub, 
        rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, 
        bool & visualizePublishedVoxels, float & publishDistanceSquared, float & truncationDistance,
        rclcpp::Clock::SharedPtr clock, Vector3f * occupiedVoxels, Voxel * sdfWeightVoxelVals);
    
    void publish(int & numVoxels);
private:
    inline bool withinPublishDistance(Vector3f & a, Vector3f & b);
    void publishWithVisualization(int & numVoxels);

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub;
    rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub; 
    bool visualizePublishedVoxels;
    float publishDistanceSquared;
    float truncationDistance;
    rclcpp::Clock::SharedPtr clock_;  
    Vector3f * occupiedVoxels;
    Voxel *  sdfWeightVoxelVals;
};

#endif