#include <iostream>

#include "cuda/tsdf_handler.cuh"
#include "transformer.hpp"
#include "params.hpp"
#include "publisher.hpp"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

rclcpp::Clock::SharedPtr clock_;

TSDFHandler * tsdfHandler;
Transformer * transformer;
Publisher * publisher;

//keep track of origin positin transformed from lidar to world frame
Vector3f origin_transformed;
Vector3f * origin_transformed_h = &origin_transformed;

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub;
rclcpp::Publisher<tsdf_package_msgs::msg::Tsdf>::SharedPtr tsdf_pub;

//keep track of average run time
double average, count = 0.0;

//arrays to hold occupied voxel data
Vector3f occupiedVoxels[OCCUPIED_VOXELS_SIZE];
Voxel sdfWeightVoxelVals[OCCUPIED_VOXELS_SIZE];

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
    publisher->publish(*occupiedVoxelsIndex);

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
  node->declare_parameter<float>("truncation_distance", 4.0);
  node->declare_parameter<float>("max_weight", 10000.0);
  node->declare_parameter<float>("publish_distance_squared", 2500.00);
  node->declare_parameter<float>("garbage_collect_distance_squared", 250000.0);
  node->declare_parameter<bool>("visualize_published_voxels", false);
  float voxel_size, max_weight, publish_distance_squared, truncation_distance, garbage_collect_distance_squared;
  bool visualize_published_voxels;
  node->get_parameter("voxel_size", voxel_size);
  node->get_parameter("truncation_distance", truncation_distance);
  node->get_parameter("max_weight", max_weight);
  node->get_parameter("publish_distance_squared", publish_distance_squared);
  node->get_parameter("visualize_published_voxels", visualize_published_voxels);
  node->get_parameter("garbage_collect_distance_squared", garbage_collect_distance_squared);

  Params params(voxel_size, truncation_distance, max_weight, publish_distance_squared, garbage_collect_distance_squared);
  initializeGlobalVars(params);

  clock_ = node->get_clock();
  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 500, callback
  ); 

  vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("occupiedVoxels", 100);

  tsdf_pub = node->create_publisher<tsdf_package_msgs::msg::Tsdf>("tsdf", 10);

  tsdfHandler = new TSDFHandler();
  //source frame set to lidar frame
  transformer = new Transformer("drone_1/LidarCustom", clock_);
  publisher = new Publisher(vis_pub, tsdf_pub, visualize_published_voxels, publish_distance_squared, truncation_distance, clock_, occupiedVoxels, sdfWeightVoxelVals);

  rclcpp::spin(node);

  rclcpp::shutdown();

  delete tsdfHandler;
  delete transformer;
  
  return 0;
}