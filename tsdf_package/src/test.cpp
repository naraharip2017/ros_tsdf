#include <iostream>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

#include "cuda/tsdf_handler.cuh"
#include "transformer.hpp"
#include "publisher.hpp"

rclcpp::Clock::SharedPtr ros_clock;

sensor_msgs::msg::PointCloud2::SharedPtr get_lidar(msr::airlib::MultirotorRpcLibClient* airsim_client) {
    auto lidar_data = airsim_client->getLidarData("LidarCustom", "drone_1"); // airsim api is imu_name, vehicle_name

    sensor_msgs::msg::PointCloud2 lidar_msg;
    lidar_msg.header.stamp = ros_clock->now();
    lidar_msg.header.frame_id = "world_ned"; // todo

    if (lidar_data.point_cloud.size() > 3)
    {
        lidar_msg.height = 1;
        lidar_msg.width = lidar_data.point_cloud.size() / 3;

        lidar_msg.fields.resize(3);
        lidar_msg.fields[0].name = "x"; 
        lidar_msg.fields[1].name = "y"; 
        lidar_msg.fields[2].name = "z";
        int offset = 0;

        for (size_t d = 0; d < lidar_msg.fields.size(); ++d, offset += 4)
        {
            lidar_msg.fields[d].offset = offset;
            lidar_msg.fields[d].datatype = sensor_msgs::msg::PointField::FLOAT32;
            lidar_msg.fields[d].count  = 1;
        }

        lidar_msg.is_bigendian = false;
        lidar_msg.point_step = offset; // 4 * num fields
        lidar_msg.row_step = lidar_msg.point_step * lidar_msg.width;

        lidar_msg.is_dense = true; // todo
        std::vector<float> data_std = lidar_data.point_cloud;

        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&data_std[0]);
        vector<unsigned char> lidar_msg_data(bytes, bytes + sizeof(float) * data_std.size());
        lidar_msg.data = std::move(lidar_msg_data);
    }

    return std::make_shared<sensor_msgs::msg::PointCloud2>(lidar_msg);
}

int main(int argc, char **argv) {
    msr::airlib::MultirotorRpcLibClient * airsim_client = new msr::airlib::MultirotorRpcLibClient("oren-mc1040");
	airsim_client->confirmConnection();
        
    ros_clock = std::make_shared<rclcpp::Clock>();

    tsdf::TSDFHandler * tsdf_handler;
    tsdf::Transformer * transformer;

    tsdf_handler = new tsdf::TSDFHandler();
    transformer = new tsdf::Transformer("drone_1/LidarCustom", ros_clock);

    Vector3f lidar_position_transformed;
    Vector3f publish_voxels_pos[PUBLISH_VOXELS_MAX_SIZE];
    tsdf::Voxel publish_voxels_data[PUBLISH_VOXELS_MAX_SIZE];
    for(int i = 0; i < 100; i++) {
        sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg = get_lidar(airsim_client);

        //convert lidar position coordinates to same frame as point cloud
        try{
            transformer->getLidarPositionInPointCloudFrame(*lidar_msg, lidar_position_transformed);
        } //if error converting lidar position don't incorporate the data into the TSDF
        catch(tf2::LookupException & e){
            return -1;
        }
        catch(tf2::ExtrapolationException & e){
            return -1;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        transformer->convertPointCloud2ToPointCloudXYZ(lidar_msg, point_cloud_xyz);
        printf("Point Cloud Size: %lu\n", point_cloud_xyz->size());

        //used to keep track of number of voxels to publish after integrating this lidar scan into the TSDF
        int publish_voxels_size = 0;

        tsdf_handler->processPointCloudAndUpdateVoxels(point_cloud_xyz, &lidar_position_transformed, publish_voxels_pos, &publish_voxels_size, publish_voxels_data);

        usleep(200000);
    }

    return 0;
}