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

void publishOccupiedVoxelsMarker(int & numVoxels){
    visualization_msgs::msg::MarkerArray markerArray;
    markerArray.markers.resize(1);
    visualization_msgs::msg::Marker marker;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.header.frame_id = "drone_1";
    marker.ns = "occupied_voxels";
    marker.id = 0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = .1;
    marker.scale.y = .1;
    marker.scale.z = .1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for(int i=0;i<numVoxels;i++){
      Vector3f v = occupiedVoxels[i];
      geometry_msgs::msg::Point p;
      //although in ned frame this swaps the points to enu for easier visualization in rviz
      p.x = v(1);
      p.y = v(0);
      p.z = v(2)*-1;
      marker.points.push_back(p);
    }

    marker.header.stamp = clock_->now();
    markerArray.markers[0] = marker;
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
    tsdfHandler->processPointCloudAndUpdateVoxels(temp_cloud, origin_transformed_h, occupiedVoxels, occupiedVoxelsIndex);
    printf("Occupied Voxels: %d\n", *occupiedVoxelsIndex);

    publishOccupiedVoxelsMarker(*occupiedVoxelsIndex);

    free(occupiedVoxelsIndex);

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Overall Duration: ";
    std::cout << duration.count() << std::endl;

    std::cout << "Cuda Average Duration: ";
    average1 += duration.count();
    count++;
    std::cout << average1/count << std::endl; 
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tsdfHandler = new TSDFHandler();
  transformer = new Transformer("drone_1/LidarCustom");

  auto node = rclcpp::Node::make_shared("my_subscriber");

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 100, callback
  ); 

  vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("occupiedVoxels", 500);

  rclcpp::spin(node);

  rclcpp::shutdown();

  free(tsdfHandler);
  free(transformer);
  
  return 0;
}