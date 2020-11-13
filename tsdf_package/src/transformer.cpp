#include "transformer.hpp"

Transformer::Transformer(std::string source_frame_, rclcpp::Clock::SharedPtr clock_){
    tf_buffer = new tf2_ros::Buffer(clock_); 
    tf_listener = new tf2_ros::TransformListener(*tf_buffer);
    source_frame = source_frame_;
}

/*
* Method attempts to converts origin of lidar frame to point cloud frame
*/
void Transformer::getOriginInPointCloudFrame(const sensor_msgs::msg::PointCloud2 & in, Vector3f & origin_transformed){
  geometry_msgs::msg::TransformStamped transform;
  geometry_msgs::msg::PointStamped point_out;
  origin_in.header = in.header;
  try { 
    auto header = in.header;
    auto frame_id = header.frame_id;
    auto stamp = header.stamp;

    auto start = std::chrono::high_resolution_clock::now();

    transform = tf_buffer->lookupTransform(frame_id, source_frame, tf2_ros::fromMsg(stamp), std::chrono::milliseconds(1000));

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout << "Transform Duration: ";
    std::cout << duration.count() << std::endl;

    tf2::doTransform(origin_in, point_out, transform);
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

/*
* Method converts pointcloud2 to pointcloudXYZ
*/
void Transformer::convertPointCloud2ToPointCloudXYZ(sensor_msgs::msg::PointCloud2::SharedPtr msg, pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*msg,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);
}