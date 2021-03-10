#include "publisher.hpp"

//determines if voxel is within publishing distance of drone - move this logic to tsdf handler
inline bool Publisher::withinPublishDistance(Vector3f & a, Vector3f & b){
  Vector3f diff = b - a;
  float distance_squared  = pow(diff(0), 2) + pow(diff(1), 2) + pow(diff(2), 2);
  if(distance_squared <= publish_distance_squared){
    return true;
  }
  return false;
}

void Publisher::publishWithVisualization(int & num_voxels){   
    marker_array.markers.clear(); 
    visualization_msgs::msg::Marker marker_green, marker_red;
    marker_green.type = marker_red.type = visualization_msgs::msg::Marker::CUBE_LIST;
    marker_green.action = marker_red.action = visualization_msgs::msg::Marker::ADD;
    marker_green.header.frame_id = marker_red.header.frame_id= "world_ned";
    marker_green.ns = marker_red.ns = "occupied_voxels";
    marker_green.id = 0;
    marker_red.id = 1;
    // marker_green.pose.orientation.w = marker_red.pose.orientation.w = 1.0;
    marker_green.scale.x = marker_red.scale.x = voxel_size;
    marker_green.scale.y = marker_red.scale.y = voxel_size;
    marker_green.scale.z = marker_red.scale.z = voxel_size;
    marker_green.color.a = marker_red.color.a = 1.0; // Don't forget to set the alpha!
    marker_green.color.r = 0.0;
    marker_red.color.g = 0.0;
    marker_green.color.b = marker_red.color.b = 0.0;
    marker_red.color.r = 1.0;
    marker_green.color.g = 1.0;

    pcl::PointCloud<pcl::PointXYZ> pointcloud;

    auto message = tsdf_package_msgs::msg::Tsdf(); 

    for(int i=0;i<num_voxels;i++){
      Vector3f v = occupied_voxels[i];
      Voxel voxel = sdf_weight_voxel_vals[i];
      geometry_msgs::msg::Point p;
      //although in ned frame this swaps the points to enu for easier visualization in rviz
      p.x = -1 * v(1); //todo: fix, swapping so visualization looks right in rviz
      p.y = -1 * v(0);
      p.z = -1 * v(2);

      pcl::PointXYZ point(v(0),v(1),v(2));
      pointcloud.push_back(point);

      auto msg_voxel = tsdf_package_msgs::msg::Voxel();
      msg_voxel.sdf = voxel.sdf;
      msg_voxel.weight = voxel.weight;
      msg_voxel.x = v(0);
      msg_voxel.y = v(1);
      msg_voxel.z = v(2);
      message.voxels.push_back(msg_voxel);
      if(voxel.sdf >= 0){
        marker_green.points.push_back(p);
      }
      else{
        marker_red.points.push_back(p);
      }
    }

    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(pointcloud,pcl_pc2);

    sensor_msgs::msg::PointCloud2 msg;
    pcl_conversions::fromPCL(pcl_pc2, msg);

    // marker_green.header.stamp = marker_red.header.stamp = clock->now();
    marker_array.markers.push_back(marker_green);
    marker_array.markers.push_back(marker_red);
    tsdf_occupied_voxels_pub->publish(marker_array);
    msg.header.frame_id = "world_ned";
    tsdf_occupied_voxels_pc_pub->publish(msg);

    message.size = message.voxels.size();
    message.truncation_distance = truncation_distance;
    tsdf_pub->publish(message);
}

//publishes visualization and tsdf topic
void Publisher::publish(int & num_voxels){ 
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
  if(visualize_published_voxels && duration > 5){
    last_time = now;
    publishWithVisualization(num_voxels);
  }
  else{
    auto message = tsdf_package_msgs::msg::Tsdf(); 
    for(int i=0;i<num_voxels;i++){
      Vector3f v = occupied_voxels[i];
      Voxel voxel = sdf_weight_voxel_vals[i];
      geometry_msgs::msg::Point p;

      auto msg_voxel = tsdf_package_msgs::msg::Voxel();
      msg_voxel.sdf = voxel.sdf;
      msg_voxel.weight = voxel.weight;
      msg_voxel.x = v(0);
      msg_voxel.y = v(1);
      msg_voxel.z = v(2);
      message.voxels.push_back(msg_voxel);
    }

    message.size = message.voxels.size();
    message.truncation_distance = truncation_distance;
    message.voxel_size = voxel_size;
    tsdf_pub->publish(message);
  }
}

Publisher::Publisher(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tsdf_occupied_voxels_pub,
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tsdf_occupied_voxels_pc_pub, 
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, 
bool & visualize_published_voxels, float & publish_distance_squared, float & truncation_distance, float & voxel_size, rclcpp::Clock::SharedPtr clock,
Vector3f * occupied_voxels, Voxel * sdf_weight_voxel_vals){
  this->tsdf_occupied_voxels_pub = tsdf_occupied_voxels_pub;
  this->tsdf_occupied_voxels_pc_pub = tsdf_occupied_voxels_pc_pub;
  this->tsdf_pub = tsdf_pub;
  this->visualize_published_voxels = visualize_published_voxels;
  this->publish_distance_squared = publish_distance_squared;
  this->truncation_distance = truncation_distance;
  this->voxel_size = voxel_size;
  this->clock = clock;
  this->occupied_voxels = occupied_voxels;
  this->sdf_weight_voxel_vals = sdf_weight_voxel_vals;
  marker_array.markers.resize(2);
  last_time = std::chrono::high_resolution_clock::now();
}