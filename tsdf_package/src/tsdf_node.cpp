#include <iostream>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "tsdf_package_msgs/msg/tsdf.hpp"
#include "tsdf_package_msgs/msg/voxel.hpp"
#include "cuda/tsdf_handler.cuh"
#include "transformer.hpp"
#include "params.hpp"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

rclcpp::Clock::SharedPtr clock_;

TSDFHandler * tsdfHandler;
Transformer * transformer;

//keep track of origin positin transformed from lidar to world frame
Vector3f origin_transformed;
Vector3f * origin_transformed_h = &origin_transformed;

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub;
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub;

//keep track of average run time
float average, count = 0.0;

bool visualizePublishedVoxels;
float publishDistanceSquared;
float truncationDistance;

//arrays to hold occupied voxel data
Vector3f occupiedVoxels[OCCUPIED_VOXELS_SIZE];
Voxel sdfWeightVoxelVals[OCCUPIED_VOXELS_SIZE];

//determines if voxel is within publishing distance of drone - move this logic to tsdf handler
inline bool withinPublishDistance(Vector3f & a, Vector3f & b){
  Vector3f diff = b - a;
  float distanceSquared  = pow(diff(0), 2) + pow(diff(1), 2) + pow(diff(2), 2);
  if(distanceSquared <= publishDistanceSquared){
    return true;
  }
  return false;
}

void publishWithVisualization(int & numVoxels){
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
void publish(int & numVoxels){ 
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
    tsdf_pub->publish(message);
  }
}

//callback for point cloud subscriber
void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    auto start = std::chrono::high_resolution_clock::now();
    transformer->getOriginInPointCloudFrame(*msg, origin_transformed); // todo: if this throws errors then don't do the point cloud update

    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    transformer->convertPointCloud2ToPointCloudXYZ(msg, temp_cloud);
    printf("Point Cloud Size: %lu\n", temp_cloud->size());

    int * occupiedVoxelsIndex = new int(0);
    tsdfHandler->processPointCloudAndUpdateVoxels(temp_cloud, origin_transformed_h, occupiedVoxels, occupiedVoxelsIndex, sdfWeightVoxelVals);
    printf("Occupied Voxels: %d\n", *occupiedVoxelsIndex);

    //checks if occupied voxels count is larger than the amount to publish
    if(*occupiedVoxelsIndex > OCCUPIED_VOXELS_SIZE){
      *occupiedVoxelsIndex = OCCUPIED_VOXELS_SIZE;
    }
    publish(*occupiedVoxelsIndex);

    delete occupiedVoxelsIndex;

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Overall Duration: ";
    std::cout << duration.count() << std::endl;

    std::cout << "Average Duration: ";
    average += duration.count();
    count++;
    std::cout << average/count << std::endl; 
    std::cout << "---------------------------------------------------------------" << std::endl;
}

int main(int argc, char ** argv)
{

  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("tsdf_node");

  node->declare_parameter<float>("voxel_size", .5);
  node->declare_parameter<float>("truncation_distance", .1);
  node->declare_parameter<float>("max_weight", 10000.0);
  node->declare_parameter<float>("publish_distance_squared", 425);
  node->declare_parameter<bool>("visualize_published_voxels", false);
  float voxel_size, max_weight;
  node->get_parameter("voxel_size", voxel_size);
  node->get_parameter("truncation_distance", truncationDistance);
  node->get_parameter("max_weight", max_weight);
  node->get_parameter("publish_distance_squared", publishDistanceSquared);
  node->get_parameter("visualize_published_voxels", visualizePublishedVoxels);

  Params params(voxel_size, truncationDistance, max_weight, publishDistanceSquared);
  initializeGlobalVars(params);

  clock_ = node->get_clock();
  tsdfHandler = new TSDFHandler();
  //source frame set to lidar frame
  transformer = new Transformer("drone_1/LidarCustom", clock_);

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 500, callback
  ); 

  vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("occupiedVoxels", 100);

  tsdf_pub = node->create_publisher<tsdf_package_msgs::msg::Tsdf>("tsdf", 10);

  rclcpp::spin(node);

  rclcpp::shutdown();

  delete tsdfHandler;
  delete transformer;
  
  return 0;
}