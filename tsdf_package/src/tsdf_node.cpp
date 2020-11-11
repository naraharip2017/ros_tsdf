#include <memory>
#include <pcl_conversions/pcl_conversions.h>
#include <cstdio>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include "rclcpp/rclcpp.hpp"
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "std_msgs/msg/string.hpp"
#include "transformPC.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <iostream>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "cuda/tsdf_container.cuh"
#include "pointcloud.cuh"

typedef Eigen::Matrix<float, 3, 1> Vector3f;

extern void pointcloudMain( pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, TSDFContainer * tsdfContainer, Vector3f * occupiedVoxels_h, int * index_h);
extern void testVoxelBlockTraversal(TSDFContainer * tsdfContainer, Vector3f * occupiedVoxels_h, int * index_h);
extern void testVoxelTraversal();

const std::string source_frame = "drone_1/LidarCustom";
rclcpp::Clock::SharedPtr clock_;
tf2_ros::Buffer* tfBuffer;
tf2_ros::TransformListener* tfListener;
geometry_msgs::msg::PointStamped point_in;
Vector3f origin_transformed;
Vector3f * origin_transformed_h = &origin_transformed;
Handler * handler;

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub;

float average1, count = 0.0;

Vector3f occupiedVoxels[1000000];
 
// float FloorFun(float x, float scale){
//   return floor(x*scale) / scale;
// }

// void GetVoxelBlockCenterFromPoint(Vector3f point){
//   float voxelBlock_size = VOXEL_SIZE * VOXEL_PER_BLOCK;
//   float halfVoxelBlock_size = voxelBlock_size/2;
//   float scale = 1 / voxelBlock_size;
//   Vector3f blockCenter;
//   blockCenter(0) = FloorFun(point(0), scale) + halfVoxelBlock_size;
//   blockCenter(1) = FloorFun(point(1), scale) + halfVoxelBlock_size;
//   blockCenter(2) = FloorFun(point(2), scale) + halfVoxelBlock_size;
//   std::cout << blockCenter;
//   std::cout << "\n";
// }

void getOriginInPointCloudFrame(const sensor_msgs::msg::PointCloud2 & in, Vector3f & origin_transformed){
  // Get the TF transform
  geometry_msgs::msg::TransformStamped transform;
  geometry_msgs::msg::PointStamped point_out;
  point_in.header = in.header;
  try { //wait for a duration
  //transform from lidar frame to world frame

    auto header = in.header;
    auto frame_id = header.frame_id;
    auto stamp = header.stamp;
    auto start = std::chrono::high_resolution_clock::now();
    transform = tfBuffer->lookupTransform(frame_id, source_frame, tf2_ros::fromMsg(stamp), std::chrono::milliseconds(1000)); //trouble - continuously increasing in time to retrieve transformation
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "transform duration: ";
    std::cout << duration.count() << std::endl;
    tf2::doTransform(point_in, point_out, transform);
    auto point = point_out.point;
    origin_transformed(0) = point.x;
    origin_transformed(1) = point.y;
    origin_transformed(2) = point.z;
      
  } catch (tf2::LookupException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  } catch (tf2::ExtrapolationException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  }
  
}

void callback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    auto start = std::chrono::high_resolution_clock::now();
    getOriginInPointCloudFrame(*msg, origin_transformed); // if this throws errors then don't do the point cloud update
    
    // printf("(%f, %f, %f)\n", origin_transformed(0), origin_transformed(1), origin_transformed(2));
    // sensor_msgs::msg::PointCloud2::SharedPtr target_frame_msg;
    // transformPointCloud(target_frame, *msg.get(), *target_frame_msg.get(), *tfBuffer);
    // pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    // pcl::fromROSMsg(*msg, pointcloud_pcl);
    //target frame, target time, fixed frame, tf listener, cloud in/out
    //convert point cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr converted_temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PCLHeader header = temp_cloud->header;  
    // std::string frame_id = header.frame_id;
    // pcl::uint64_t stamp = header.stamp;

    //get sensor origin with PointCloud2 and tsdf_handler object pass that to pointcloudMain
    // tsdfHandler.

    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

    // auto start1 = std::chrono::high_resolution_clock::now();
    printf("point cloud size: %lu\n", temp_cloud->size());
    int * occupiedVoxelsIndex = new int(0);
    handler->processPointCloudAndUpdateVoxels(temp_cloud, origin_transformed_h, occupiedVoxels, occupiedVoxelsIndex);
    //pointcloudMain(temp_cloud, origin_transformed_h, tsdfHandler, occupiedVoxels, occupiedVoxelsIndex);
    // printf("occupied voxels: %d\n", *occupiedVoxelsIndex);

    visualization_msgs::msg::MarkerArray markerArray;
    markerArray.markers.resize(1);
    // for(int i=0;i<*occupiedVoxelsIndex;i++){
    //   Vector3f v = occupiedVoxels[i];
    //   visualization_msgs::msg::Marker marker;
    //   marker.header.frame_id = "drone_1";
    //   rclcpp::Time t(0);
    //   marker.header.stamp = clock_->now();
    //   marker.ns = "lidar";
    //   marker.id = i;
    //   marker.type = visualization_msgs::msg::Marker::CUBE;
    //   marker.action = visualization_msgs::msg::Marker::ADD;
    //   marker.pose.position.x = v(1);
    //   marker.pose.position.y = v(0);
    //   marker.pose.position.z = v(2)*-1;
    //   // marker.pose.position.x = v(0);
    //   // marker.pose.position.y = v(1);
    //   // marker.pose.position.z = v(2);
    //   marker.pose.orientation.x = 0.0;
    //   marker.pose.orientation.y = 0.0;
    //   marker.pose.orientation.z = 0.0;
    //   marker.pose.orientation.w = 1.0;
    //   marker.scale.x = .1;
    //   marker.scale.y = .1;
    //   marker.scale.z = .1;
    //   marker.color.a = 1.0; // Don't forget to set the alpha!
    //   marker.color.r = 0.0;
    //   marker.color.g = 1.0;
    //   marker.color.b = 0.0;
    //   markerArray.markers[i] = marker;
    // }
      visualization_msgs::msg::Marker marker;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.header.frame_id = "drone_1";
    marker.ns = "lidar";
    marker.id = 0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = .1;
    marker.scale.y = .1;
    marker.scale.z = .1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for(int i=0;i<*occupiedVoxelsIndex;i++){
      Vector3f v = occupiedVoxels[i];
      geometry_msgs::msg::Point p;
      p.x = v(1);
      p.y = v(0);
      p.z = v(2)*-1;
      marker.points.push_back(p);
      // visualization_msgs::msg::Marker marker;
      // markerArray.markers[i] = marker;
    }

    marker.header.stamp = clock_->now();
    markerArray.markers[0] = marker;
    vis_pub->publish(markerArray);


    free(occupiedVoxelsIndex);

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "overal duration: ";
    std::cout << duration.count() << std::endl;
    // auto stop1 = std::chrono::high_resolution_clock::now(); 
    // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1); 
    std::cout << "cuda average duration: ";
    average1 += duration.count();
    count++;
    std::cout << average1/count << std::endl; 
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Vector3f test(-.75,.6,.3);

  // std::cout << test;

  // std::cout << "\n";

  // GetVoxelBlockCenterFromPoint(test);

  clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tfBuffer = new tf2_ros::Buffer(clock_);
  tfListener = new tf2_ros::TransformListener(*tfBuffer);
  handler = new Handler();
  // testVoxelBlockTraversal(tsdfHandler, occupiedVoxels, occupiedVoxelsIndex);
  //      for(int i=0; i < 3; ++i){
  //   printf("occupied voxel: (%f, %f, %f)\n", occupiedVoxels[i](0), occupiedVoxels[i](1), occupiedVoxels[i](2));
  // }
    // printf("occupied voxels: %d\n", *occupiedVoxelsIndex);
      // origin_transformed(0) = 0;
      // origin_transformed(1) = 0;
      // origin_transformed(2) = 0;
      
  // // Vector3f point_h[1];
  // // Vector3f A(1,1,1);
  // // point_h[0] = A;
  // // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 1);

  // point_in.point.x = 0.0;
  // point_in.point.y = 0.0;
  // point_in.point.z = 0.0;
  // // Transformer *transformer = new Transformer();

// visualization_msgs::MarkerArray* marker_array
  auto node = rclcpp::Node::make_shared("my_subscriber");

  auto lidar_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/airsim_node/drone_1/lidar/LidarCustom", 100, callback
  ); //todo: should it be 1? might be .1 check publishing rate in airsim but the mapping pipeline runs at 2hz?
  vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("occupiedVoxels", 100);
  rclcpp::spin(node);
  // while(rclcpp::ok()){
  //   vis_pub->publish(markerArray);
  // }
  //   while(rclcpp::ok()){
  //   // visualization_msgs::msg::MarkerArray markerArray;
  //   // markerArray.markers.resize(3);
  //   visualization_msgs::msg::Marker marker;
  //   marker.type = visualization_msgs::msg::Marker::POINTS;
  //   marker.action = visualization_msgs::msg::Marker::ADD;
  //   marker.header.frame_id = "map";
  //   marker.ns = "lidar";
  //   marker.id = 0;
  //   marker.pose.orientation.w = 1.0;
  //   marker.scale.x = .1;
  //   marker.scale.y = .1;
  //   marker.scale.z = .1;
  //   marker.color.a = 1.0; // Don't forget to set the alpha!
  //   marker.color.r = 0.0;
  //   marker.color.g = 1.0;
  //   marker.color.b = 0.0;
  //   for(int i=0;i<3;i++){
  //     Vector3f v = occupiedVoxels[i];
  //     geometry_msgs::msg::Point p;
  //     p.x = v(0);
  //     p.y = v(1);
  //     p.z = v(2);
  //     marker.points.push_back(p);
  //     // visualization_msgs::msg::Marker marker;
  //     // markerArray.markers[i] = marker;
  //   }
  //   marker.header.stamp = clock_->now();
  //   vis_pub->publish(marker);
  // }
  rclcpp::shutdown();

  free(tfBuffer);
  free(tfListener);
  free(handler);
  
  // //create hash table and everything here which is defined and implemented in tsdf_node.cuh and tsdf.cu. Then pass the table to pointCloudMain where point clouds are handled. Inside the class we hold all variables

      // int size= 64;
      // Vector3f point_h[size];
      // Vector3f A(1,1,1);
      // Vector3f B(5,5,5);
      // Vector3f C(9,9,9);
      // Vector3f D(9,9,9);
      // point_h[0] = A;
      // point_h[1] = B;
      // point_h[2] = C;
      // point_h[3] = D;
      // point_h[4] = C;
      // point_h[5] = C;
      // point_h[6] = C;
      // point_h[7] = C;

    //  point_h[0] = A;
    

    // for(int i=1; i<=size; i++){
    //   Vector3f * p = new Vector3f(i,i,i);
    //   point_h[i-1] = *p;
    // }

    //     for(int i=1; i<=size; ++i){
    //   Point * p = new Point(i+4,i+4,i+4);
    //   point_h[i-1] = *p;
    // }

  // Point point_h[2];
  // Point * A = new Point(1,1,1);
  // Point * B = new Point(2,2,2);
  // point_h[0] = *A;
  // point_h[1] = *B;
    
  // testVoxelBlockTraversal(tsdfHandler);

  //addPoints
  
  //tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, size);
  // Vector3f * point_h2;
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  // point_h[0] = *C;
  // point_h[1] = *D;

  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h, 2);

  // tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);
  //tsdfHandler->integrateVoxelBlockPointsIntoHashTable(point_h2, 0);

  

  //add unallocated points first test with linked list etc
  //change lidar to 300000 points and see size of pc per frame
  return 0;
}