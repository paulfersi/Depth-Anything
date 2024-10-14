import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Configuration
encoder = 'vits' # s for small, also base and large
image_path = 'original_frame.jpg'  # Image captured by the webcam
depth_output_path = 'depth_frame.jpg'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

total_params = sum(param.numel() for param in depth_anything.parameters())
print(f'Total parameters: {total_params / 1e6:.2f}M')

transform = Compose([
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

# Read image
raw_image = cv2.imread(image_path)

if raw_image is None:
    raise FileNotFoundError(f'Image not found: {image_path}')

# Original image size
original_height, original_width = raw_image.shape[:2]

# Resize image for processing
image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
image = cv2.resize(image, (518, 518))  # Resize to match the input size expected by the model

image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    depth = depth_anything(image)

depth = F.interpolate(depth[None], (518, 518), mode='bilinear', align_corners=False)[0, 0]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

depth = depth.cpu().numpy().astype(np.uint8)

# Apply color map to depth image
depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

# Resize depth map to match the original image size
depth_color = cv2.resize(depth_color, (original_width, original_height))

# Save the depth color image
cv2.imwrite(depth_output_path, depth_color)

print(f'Depth image saved to {depth_output_path}')
