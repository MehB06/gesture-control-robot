import rclpy
from rclpy.node import Node

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

def main():
    rclpy.init()
    node = ControlNode()
    node.get_logger().info('control_node started')
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()