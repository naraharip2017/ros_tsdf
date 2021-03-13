#include "publisher.hpp"

/**
 * publishes voxels on tsdf topic
 * @param publish_voxels_size the amount of voxels to publish. The data is stored between 0 - publish_voxels_size in publish_voxels_pos and publish_voxels_data
 */
void Publisher::publish(int & publish_voxels_size){ 
  auto message = tsdf_package_msgs::msg::Tsdf(); 
  for(int i=0;i<publish_voxels_size;i++){
    Vector3f publish_voxel_pos = publish_voxels_pos[i];
    Voxel publish_voxel_data = publish_voxels_data[i];

    auto msg_voxel = tsdf_package_msgs::msg::Voxel();
    msg_voxel.sdf = publish_voxel_data.sdf;
    msg_voxel.weight = publish_voxel_data.weight;
    msg_voxel.x = publish_voxel_pos(0);
    msg_voxel.y = publish_voxel_pos(1);
    msg_voxel.z = publish_voxel_pos(2);
    message.voxels.push_back(msg_voxel);
  }

  message.size = message.voxels.size();
  message.truncation_distance = truncation_distance;
  message.voxel_size = voxel_size;
  tsdf_pub->publish(message);
}

/**
 * publisher constructor
 * @param tsdf_pub publisher to publish the tsdf voxels
 * @param truncation_distance truncation distance for the tsdf to add to publication msg
 * @param voxel_size voxel size for tsdf to add to publication msg
 * @param publish_voxels_pos reference to array voxels to publish positions are copied to
 * @param publish_voxels_data reference to array voxels to publish sdf + weight data are copied to
 */
Publisher::Publisher(
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub, float & truncation_distance,
float & voxel_size, Vector3f * publish_voxels_pos, Voxel * publish_voxels_data){
  this->tsdf_pub = tsdf_pub;
  this->truncation_distance = truncation_distance;
  this->voxel_size = voxel_size;
  this->publish_voxels_pos = publish_voxels_pos;
  this->publish_voxels_data = publish_voxels_data;
}