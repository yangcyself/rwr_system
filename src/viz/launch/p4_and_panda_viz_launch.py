import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""
Visualize the robot hand in rviz and show the GUI to adjust the joint angles.
Does not connect to real robot nor start simulation.
"""

def generate_launch_description():

    p4_urdf = os.path.join(
        get_package_share_directory('viz'),
        "models",
        "faive_hand_p4",
        "urdf",
        "p4.urdf")

    with open(p4_urdf, 'r') as infp:
        p4_robot_desc = infp.read()

    panda_urdf = os.path.join(
        get_package_share_directory('viz'),
        "models",
        "franka_panda",
        "panda.urdf")

    with open(panda_urdf, 'r') as infp:
        panda_robot_desc = infp.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='hand',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': p4_robot_desc,}],
            remappings=[('/joint_states', '/hand/joint_states')]
            # arguments=[p4_urdf]
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='arm',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': panda_robot_desc,}],
            # arguments=[panda_urdf],
            remappings=[('/joint_states', '/arm/joint_states')]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.18', 
                       '0.7071', '0', '0', '0.7071', 'panda_link8', 'hand_root']
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'),

        Node(
            package="viz",
            executable="visualize_joints.py",
            name="visualize_joints",
            namespace='hand',
            output='log', # 'screen' for debugging
            remappings=[('/joint_states', '/hand/joint_states')]
        ),

        # Node(
        #     package="joint_state_publisher_gui",
        #     executable="joint_state_publisher_gui",
        #     namespace='hand',
        #     name="joint_state_publisher_gui",
        #     output="screen",
        # ),
        
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            namespace='arm',
            name="joint_state_publisher_gui",
            output="screen",
        ),

    ])


