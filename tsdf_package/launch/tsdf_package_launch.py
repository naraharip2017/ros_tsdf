from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tsdf_package',
            node_executable='tsdf_node',
            node_name='tsdf_node'
        )
    ])