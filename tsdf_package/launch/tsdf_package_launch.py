from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tsdf_package',
            node_executable='tsdf_node',
            node_name='tsdf_node',
            parameters=[
                {
                    "voxel_size" : .5,
                    "truncation_distance" : 4.0,
                    "max_weight" : 10000.0,
                    "visualize_published_voxels" : False,
                    "publish_distance_squared" : 2500.0,
                    "garbage_collect_distance_squared" : 250000.0
                }
            ]
        )
    ])