from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, OpaqueFunction
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
import os
import xacro

def generate_launch_description():
    urdf = PathJoinSubstitution([
        FindPackageShare('robotic_arm_gazebo'),
        'urdf',
        'robotic_arm.urdf.xacro',
    ])
    # Use OpaqueFunction to evaluate xacro at runtime into a plain Python string
    def launch_setup(context, *args, **kwargs):
        pkg_share_path = FindPackageShare('robotic_arm_gazebo').find('robotic_arm_gazebo')
        urdf_path = os.path.join(pkg_share_path, 'urdf', 'robotic_arm.urdf.xacro')
        doc = xacro.process_file(urdf_path)
        robot_description_str = doc.toxml()

        rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_str}],
        )

        spawn = ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_sim', 'create',
                 '-name', 'robotic_arm',
                 '-topic', 'robot_description'],
            output='screen'
        )

        return [rsp, TimerAction(period=2.0, actions=[spawn])]

    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '-v', '3'],
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        OpaqueFunction(function=launch_setup),
    ])