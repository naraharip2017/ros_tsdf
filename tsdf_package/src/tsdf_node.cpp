#include <iostream>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "cuda/tsdf_container.cuh"
#include "cuda/tsdf_handler.cuh"
#include "transformer.hpp"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

extern void testVoxelBlockTraversal(TSDFContainer * tsdfContainer, Vector3f * occupiedVoxels_h, int * index_h);
extern void testVoxelTraversal();

rclcpp::Clock::SharedPtr clock_;
TSDFHandler * tsdfHandler;
Transformer * transformer;

Vector3f origin_transformed;
Vector3f * origin_transformed_h = &origin_transformed;

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub;

float average1, count = 0.0;

Vector3f occupiedVoxels[200000];
Voxel sdfWeightVoxelVals[200000];

void publishOccupiedVoxelsMarker(int & numVoxels){
    visualization_msgs::msg::MarkerArray markerArray;
    markerArray.markers.resize(2);
    visualization_msgs::msg::Marker markerGreen;
    markerGreen.type = visualization_msgs::msg::Marker::POINTS;
    markerGreen.action = visualization_msgs::msg::Marker::ADD;
    markerGreen.header.frame_id = "drone_1";
    markerGreen.ns = "occupied_voxels";
    markerGreen.id = 0;
    markerGreen.pose.orientation.w = 1.0;
    markerGreen.scale.x = .1;
    markerGreen.scale.y = .1;
    markerGreen.scale.z = .1;
    markerGreen.color.a = 1.0; // Don't forget to set the alpha!
    markerGreen.color.r = 0.0;
    markerGreen.color.g = 1.0;
    markerGreen.color.b = 0.0;

    visualization_msgs::msg::Marker markerRed;
    markerRed.type = visualization_msgs::msg::Marker::POINTS;
    markerRed.action = visualization_msgs::msg::Marker::ADD;
    markerRed.header.frame_id = "drone_1";
    markerRed.ns = "occupied_voxels";
    markerRed.id = 1;
    markerRed.pose.orientation.w = 1.0;
    markerRed.scale.x = .1;
    markerRed.scale.y = .1;
    markerRed.scale.z = .1;
    markerRed.color.a = 1.0; // Don't forget to set the alpha!
    markerRed.color.r = 1.0;
    markerRed.color.g = 0.0;
    markerRed.color.b = 0.0;

    for(int i=0;i<numVoxels;i++){
      Vector3f v = occupiedVoxels[i];
      Voxel voxel = sdfWeightVoxelVals[i];
      geometry_msgs::msg::Point p;
      //although in ned frame this swaps the points to enu for easier visualization in rviz
      p.x = v(1);
      p.y = v(0);
      p.z = v(2)*-1;
      if(voxel.sdf >= 0){
        markerGreen.points.push_back(p);
      }
      else{
        markerRed.points.push_back(p);
      }
    }

    markerGreen.header.stamp = clock_->now();
    markerArray.markers[0] = markerGreen;
    markerRed.header.stamp = clock_->now();
    markerArray.markers[1] = markerRed;
    vis_pub->publish(markerArray);
}

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

    publishOccupiedVoxelsMarker(*occupiedVoxelsIndex);

    free(occupiedVoxelsIndex);

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Overall Duration: ";
    std::cout << duration.count() << std::endl;

    std::cout << "Average Duration: ";
    average1 += duration.count();
    count++;
    std::cout << average1/count << std::endl; 
    std::cout << "---------------------------------------------------------------" << std::endl;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("my_subscriber");

  clock_ = node->get_clock();
  tsdfHandler = new TSDFHandler();
  transformer = new Transformer("drone_1/LidarCustom", clock_);

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 500, callback
  ); 

  vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("occupiedVoxels", 100);

  rclcpp::spin(node);

  rclcpp::shutdown();

  free(tsdfHandler);
  free(transformer);
  
  return 0;
}