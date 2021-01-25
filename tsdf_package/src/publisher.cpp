#include "publisher.hpp"

//determines if voxel is within publishing distance of drone - move this logic to tsdf handler
inline bool Publisher::withinPublishDistance(Vector3f & a, Vector3f & b){
  Vector3f diff = b - a;
  float distanceSquared  = pow(diff(0), 2) + pow(diff(1), 2) + pow(diff(2), 2);
  if(distanceSquared <= publishDistanceSquared){
    return true;
  }
  return false;
}

void Publisher::publishWithVisualization(int & numVoxels){   
    markerArray.markers.clear(); 
    visualization_msgs::msg::Marker markerGreen, markerRed;
    markerGreen.type = markerRed.type = visualization_msgs::msg::Marker::CUBE_LIST;
    markerGreen.action = markerRed.action = visualization_msgs::msg::Marker::ADD;
    markerGreen.header.frame_id = markerRed.header.frame_id= "world_ned";
    markerGreen.ns = markerRed.ns = "occupied_voxels";
    markerGreen.id = 0;
    markerRed.id = 1;
    // markerGreen.pose.orientation.w = markerRed.pose.orientation.w = 1.0;
    markerGreen.scale.x = markerRed.scale.x = voxel_size;
    markerGreen.scale.y = markerRed.scale.y = voxel_size;
    markerGreen.scale.z = markerRed.scale.z = voxel_size;
    markerGreen.color.a = markerRed.color.a = 1.0; // Don't forget to set the alpha!
    markerGreen.color.r = 0.0;
    markerRed.color.g = 0.0;
    markerGreen.color.b = markerRed.color.b = 0.0;
    markerRed.color.r = 1.0;
    markerGreen.color.g = 1.0;

    pcl::PointCloud<pcl::PointXYZ> pointcloud;

    auto message = tsdf_package_msgs::msg::Tsdf(); 

    for(int i=0;i<numVoxels;i++){
      Vector3f v = occupiedVoxels[i];
      Voxel voxel = sdfWeightVoxelVals[i];
      geometry_msgs::msg::Point p;
      //although in ned frame this swaps the points to enu for easier visualization in rviz
      p.x = -1 * v(1); //todo: fix, swapping so visualization looks right in rviz
      p.y = -1 * v(0);
      p.z = -1 * v(2);

      pcl::PointXYZ point(v(0),v(1),v(2));
      pointcloud.push_back(point);

      auto msgVoxel = tsdf_package_msgs::msg::Voxel();
      msgVoxel.sdf = voxel.sdf;
      msgVoxel.weight = voxel.weight;
      msgVoxel.x = v(0);
      msgVoxel.y = v(1);
      msgVoxel.z = v(2);
      message.voxels.push_back(msgVoxel);
      if(voxel.sdf >= 0){
        markerGreen.points.push_back(p);
      }
      else{
        markerRed.points.push_back(p);
      }
    }

    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(pointcloud,pcl_pc2);

    sensor_msgs::msg::PointCloud2 msg;
    pcl_conversions::fromPCL(pcl_pc2, msg);

    // markerGreen.header.stamp = markerRed.header.stamp = clock_->now();
    // markerArray.markers[0] = markerGreen;
    // markerArray.markers[1] = markerRed;
    markerArray.markers.push_back(markerGreen);
    markerArray.markers.push_back(markerRed);
    tsdf_occupied_voxels_pub->publish(markerArray);
    msg.header.frame_id = "world_ned";
    tsdf_occupied_voxels_pc_pub->publish(msg);

    message.size = message.voxels.size();
    message.truncation_distance = truncationDistance;
    tsdf_pub->publish(message);
}

//publishes visualization and tsdf topic
void Publisher::publish(int & numVoxels){ 
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
  if(visualizePublishedVoxels && duration > 5){
    last_time = now;
    publishWithVisualization(numVoxels);
  }
  else{
    auto message = tsdf_package_msgs::msg::Tsdf(); 
    for(int i=0;i<numVoxels;i++){
      Vector3f v = occupiedVoxels[i];
      Voxel voxel = sdfWeightVoxelVals[i];
      geometry_msgs::msg::Point p;

      auto msgVoxel = tsdf_package_msgs::msg::Voxel();
      msgVoxel.sdf = voxel.sdf;
      msgVoxel.weight = voxel.weight;
      msgVoxel.x = v(0);
      msgVoxel.y = v(1);
      msgVoxel.z = v(2);
      message.voxels.push_back(msgVoxel);
    }

    message.size = message.voxels.size();
    message.truncation_distance = truncationDistance;
    message.voxel_size = voxel_size;
    tsdf_pub->publish(message);
  }
}

Publisher::Publisher(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tsdf_occupied_voxels_pub,
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tsdf_occupied_voxels_pc_pub, 
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, 
bool & visualizePublishedVoxels, float & publishDistanceSquared, float & truncationDistance, float & voxel_size, rclcpp::Clock::SharedPtr clock,
Vector3f * occupiedVoxels, Voxel * sdfWeightVoxelVals){
  this->tsdf_occupied_voxels_pub = tsdf_occupied_voxels_pub;
  this->tsdf_occupied_voxels_pc_pub = tsdf_occupied_voxels_pc_pub;
  this->tsdf_pub = tsdf_pub;
  this->visualizePublishedVoxels = visualizePublishedVoxels;
  this->publishDistanceSquared = publishDistanceSquared;
  this->truncationDistance = truncationDistance;
  this->voxel_size = voxel_size;
  clock_ = clock;
  this->occupiedVoxels = occupiedVoxels;
  this->sdfWeightVoxelVals = sdfWeightVoxelVals;
  markerArray.markers.resize(2);
  last_time = std::chrono::high_resolution_clock::now();
}