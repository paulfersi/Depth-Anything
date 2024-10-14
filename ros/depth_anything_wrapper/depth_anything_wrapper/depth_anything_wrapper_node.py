import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class DepthAnythingWrapperNode(Node):

    def __init__(self):
        super().__init__('depth_anything_wrapper_node')
        self.image_subscription = self.create_subscription(String,'topic',self.image_callback,10)

        self.depth_publisher = self.create_publisher(String,'topic',10)

    def image_callback(self,msg):
        pass

    def compute_depth(self):
        pass

    def publish_depth(self):
        pass


def main(args=None):
    rclpy.init(args=args)

    depth_anything_wrapper_node = DepthAnythingWrapperNode()

    rclpy.spin(depth_anything_wrapper_node)

    depth_anything_wrapper_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    