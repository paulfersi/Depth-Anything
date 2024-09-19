import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import os
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Config
save_in_color = False
encoder = 'vits'
image_folder = '/Users/paolofersino/Downloads/left'  # Folder containing the images

if save_in_color:
    depth_output_folder = './depth_maps_folder'  # Folder where depth maps will be saved
else:
    depth_output_folder = './depth_maps_folder_gray'  # Folder where depth maps will be saved

os.makedirs(depth_output_folder, exist_ok=True)  # Create the output folder if it doesn't exist


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

# Get a list of all image files in the folder
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # Add any image formats you expect
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

if not image_files:
    raise FileNotFoundError(f'No image files found in the folder: {image_folder}')

# Sort the images by timestamp (using the numeric part of the filename)
image_files.sort(key=lambda x: float(os.path.splitext(x)[0]))  # Sort by the numeric timestamp part

# Loop through each image and generate the depth map
for idx, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, image_file)
    
    # Read image
    raw_image = cv2.imread(image_path)
    
    if raw_image is None:
        print(f'Image not found or could not be read: {image_path}')
        continue

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

    if save_in_color:
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        # Resize depth map to match the original image size
        depth_color = cv2.resize(depth_color, (original_width, original_height))

        # Create a filename for the depth map
        depth_output_path = os.path.join(depth_output_folder, f'depth_{os.path.splitext(image_file)[0]}.jpg')

        # Save the depth color image
        cv2.imwrite(depth_output_path, depth_color)

        print(f'[{idx}] Depth image saved to {depth_output_path}')
    else:
        depth_resized = cv2.resize(depth, (original_width, original_height))

        # Create a filename for the depth map
        depth_output_path = os.path.join(depth_output_folder, f'depth_{os.path.splitext(image_file)[0]}.jpg')

        # Save the depth color image
        cv2.imwrite(depth_output_path, depth_resized)

        print(f'[{idx}] Grayscale depth image saved to {depth_output_path}')

