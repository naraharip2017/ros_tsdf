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
    visualization_msgs::msg::MarkerArray markerArray;
    markerArray.markers.resize(2);
    visualization_msgs::msg::Marker markerGreen, markerRed;
    markerGreen.type = markerRed.type = visualization_msgs::msg::Marker::POINTS;
    markerGreen.action = markerRed.action = visualization_msgs::msg::Marker::ADD;
    markerGreen.header.frame_id = markerRed.header.frame_id= "drone_1";
    markerGreen.ns = markerRed.ns = "occupied_voxels";
    markerGreen.id = 0;
    markerRed.id = 1;
    markerGreen.pose.orientation.w = markerRed.pose.orientation.w = 1.0;
    markerGreen.scale.x = markerRed.scale.x = .1;
    markerGreen.scale.y = markerRed.scale.y = .1;
    markerGreen.scale.z = markerRed.scale.z = .1;
    markerGreen.color.a = markerRed.color.a = 1.0; // Don't forget to set the alpha!
    markerGreen.color.r = 0.0;
    markerRed.color.g = 0.0;
    markerGreen.color.b = markerRed.color.b = 0.0;
    markerRed.color.r = 1.0;
    markerGreen.color.g = 1.0;

    auto message = tsdf_package_msgs::msg::Tsdf(); 

    for(int i=0;i<numVoxels;i++){
      Vector3f v = occupiedVoxels[i];
      Voxel voxel = sdfWeightVoxelVals[i];
      geometry_msgs::msg::Point p;
      //although in ned frame this swaps the points to enu for easier visualization in rviz
      p.x = v(1);
      p.y = v(0);
      p.z = v(2)*-1;

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

    markerGreen.header.stamp = markerRed.header.stamp = clock_->now();
    markerArray.markers[0] = markerGreen;
    markerArray.markers[1] = markerRed;
    vis_pub->publish(markerArray);

    message.size = message.voxels.size();
    message.truncation_distance = truncationDistance;
    tsdf_pub->publish(message);
}

//publishes visualization and tsdf topic
void Publisher::publish(int & numVoxels){ 
  if(visualizePublishedVoxels){
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

Publisher::Publisher(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub, 
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, 
bool & visualizePublishedVoxels, float & publishDistanceSquared, float & truncationDistance, float & voxel_size, rclcpp::Clock::SharedPtr clock,
Vector3f * occupiedVoxels, Voxel * sdfWeightVoxelVals){
  this->vis_pub = vis_pub;
  this->tsdf_pub = tsdf_pub;
  this->visualizePublishedVoxels = visualizePublishedVoxels;
  this->publishDistanceSquared = publishDistanceSquared;
  this->truncationDistance = truncationDistance;
  this->voxel_size = voxel_size;
  clock_ = clock;
  this->occupiedVoxels = occupiedVoxels;
  this->sdfWeightVoxelVals = sdfWeightVoxelVals;
}