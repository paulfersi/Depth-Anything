import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from std_msgs.msg import String


class DepthAnythingWrapperNode(Node):

    def __init__(self):
        super().__init__('depth_anything_wrapper_node')

        # load_parameters()

        self.init_depth_anything()

        self.image_subscription = self.create_subscription(Image,'/image_raw',self.image_callback,10)
        self.depth_publisher = self.create_publisher(Image,'/depth',10)

    def init_depth_anything(self):
        encoder = 'vits'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(self.device).eval()

        # Define image transformation
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])


    def image_callback(self,msg):
        # ROS Image msg to OpenCv format
        cv_image = self.convert_ros_to_cv(msg)

        depth_image = self.compute_depth(cv_image)

    def convert_ros_to_cv(self, ros_image):
        # Convert a ROS Image message to a NumPy array (OpenCV format)
        np_arr = np.frombuffer(ros_image.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv_image


    def compute_depth(self,raw_image):
        # publish_depth()
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

    