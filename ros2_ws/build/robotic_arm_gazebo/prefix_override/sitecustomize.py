import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/fadi/Documents/RoboticArmProject/gesture-control-robot/ros2_ws/install/robotic_arm_gazebo'
