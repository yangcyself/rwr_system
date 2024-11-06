import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""
Visualize the robot hand in rviz and show the GUI to adjust the joint angles.
Does not connect to real robot nor start simulation.
"""

def generate_launch_description():

    urdf = os.path.join(
        get_package_share_directory('viz'),
        "models",
        "faive_hand_p4",
        "urdf",
        "p4.urdf")

    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc,}],
            arguments=[urdf]),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'),
        Node(
            package="viz",
            executable="visualize_joints.py",
            name="visualize_joints",
            output='log', 
            parameters=[
                {"scheme_path": os.path.join(get_package_share_directory('viz'), "models", "faive_hand_p4/scheme_p4.yaml")}
            ]
        ),
        
    ])