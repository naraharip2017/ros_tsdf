#include "transformer.hpp"

/**
 * Transformer constructor
 * @param lidar_source_frame frame of lidar - convert from this frame to that of the point cloud
 * @param clock_ used for initializing tf_buffer
 */ 
Transformer::Transformer(std::string lidar_source_frame, rclcpp::Clock::SharedPtr clock_){
    tf_buffer = new tf2_ros::Buffer(clock_); 
    tf_listener = new tf2_ros::TransformListener(*tf_buffer);
    this->lidar_source_frame = lidar_source_frame;
}

/**
* Method attempts to converts lidar position from lidar frame to point cloud frame
* @param point_cloud point cloud with frame_id to convert to from lidar_source_frame
* @param lidar_position_transformed Vector3f to store the lidar position in lidar_source_frame transformed to the point cloud's frame 
*/
void Transformer::getLidarPositionInPointCloudFrame(const sensor_msgs::msg::PointCloud2 & point_cloud, Vector3f & lidar_position_transformed){
  geometry_msgs::msg::TransformStamped transform; //used to keep track of transformation between lidar_source_frame and point cloud frame
  geometry_msgs::msg::PointStamped point_out; //used to store location of lidar position converted to point cloud frame
  geometry_msgs::msg::PointStamped lidar_position_end_frame; //used to set the header for the frame to convert the lidar position to
  lidar_position_end_frame.header = point_cloud.header; //set header to the lidar_position_end_frame pointstamped object
  try { 
    auto header = point_cloud.header;
    auto frame_id = header.frame_id;
    auto stamp = header.stamp;

    auto start = std::chrono::high_resolution_clock::now();

    //lookup transform from lidar_source_frame to frame_id. Spend at most 1s waiting for the transform
    transform = tf_buffer->lookupTransform(frame_id, lidar_source_frame, tf2_ros::fromMsg(stamp), std::chrono::milliseconds(1000));

    //print out duration of waiting for transform
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Transform Duration: ";
    std::cout << duration.count() << std::endl;

    //using the transform get the lidar position in the lidar frame to point cloud frame and store in point_out
    tf2::doTransform(lidar_position_end_frame, point_out, transform);
    auto point = point_out.point;
    lidar_position_transformed(0) = point.x;
    lidar_position_transformed(1) = point.y;
    lidar_position_transformed(2) = point.z;
      
  } catch (tf2::LookupException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  } catch (tf2::ExtrapolationException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
  }
}

/**
* Method converts PointCloud2 to PointCloudXYZ
* @param point_cloud_2: the sensor_msgs:msg:PointCloud2::SharedPtr object to convert to a pcl PointCloudXYZ
* @param point_cloud_xyz: the PointCloudXYZ that is used to store the converted point_cloud_2
*/
void Transformer::convertPointCloud2ToPointCloudXYZ(sensor_msgs::msg::PointCloud2::SharedPtr point_cloud_2, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_xyz){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*point_cloud_2,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*point_cloud_xyz);
}