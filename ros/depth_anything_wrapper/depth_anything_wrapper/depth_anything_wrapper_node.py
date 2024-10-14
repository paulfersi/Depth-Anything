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

        self.publish_depth(depth_image,msg.header)

    def convert_ros_to_cv(self, ros_image):
        # Convert a ROS Image message to a NumPy array (OpenCV format)
        np_arr = np.frombuffer(ros_image.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv_image


    def compute_depth(self,raw_image):
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        # Compute depth
        with torch.no_grad():
            depth = self.depth_anything(image)

        # Resize depth to original image size
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        return depth

    def publish_depth(self,depth_image,header):
        depth_msg = Image()
        depth_msg.header = header 
        depth_msg.height,depth_msg.width = depth_image.shape 
        depth_msg.encoding = 'mono8'  # Single channel (depth)
        depth_msg.is_bigendian = 0
        depth_msg.step = depth_msg.width
        depth_msg.data = depth_image.tobytes()
        self.depth_publisher.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingWrapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    