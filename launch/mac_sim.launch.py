#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression

def generate_launch_description():
    return LaunchDescription([
        # Launch argument declarations
        DeclareLaunchArgument(
            'start_pose',
            default_value='[0.0, 0.0, 0.0]',
            description="Initial pose of self agent"
        ),
        DeclareLaunchArgument(
            'start_yaw',
            default_value='0.0',
            description="Initial yaw of self agent"
        ),
        DeclareLaunchArgument(
            'all_agents',
            default_value="['NX01']",
            description="List of all agents in simulation"
        ),
        DeclareLaunchArgument(
            'self_agent',
            default_value='NX01',
            description="Namespace of self agent"
        ),
        DeclareLaunchArgument(
            'model_weight_file',
            default_value='long.pth',
            description="Name of model weight file"
        ),
        Node(
            package="macbf_torch",
            executable="mac_sim",
            parameters=[{
                'start_pose': PythonExpression(LaunchConfiguration('start_pose')),
                'start_yaw': LaunchConfiguration('start_yaw'),
                'all_agents': PythonExpression(LaunchConfiguration('all_agents')),
                'self_agent': LaunchConfiguration('self_agent'),
                'model_weight_file': LaunchConfiguration('model_weight_file')
            }]
        )
    ])

# ros2 launch macbf_torch mac_sim.launch.py \
#     start_yaw:=45.0 \
#     start_pose:="[1.0, 2.0, 3.0]" \
#     all_agents:="['NX01', 'NX02']" \
#     self_agent:=NX01 \
#     model_weight_file:=your_model.pth