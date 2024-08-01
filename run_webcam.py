import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

#encoders = ['vits', 'vitb', 'vitl']
encoder = 'vits'   #the small one
video_path = 0


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec to encode video output
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640,480))

#parser = argparse.ArgumentParser()
#parser.add_argument('--img-path', type=str)
#parser.add_argument('--outdir', type=str, default='./vis_depth')
#parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

#parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
#parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

#args = parser.parse_args()

margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

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

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, raw_image = cap.read()

    if not ret:
        break

    raw_image = cv2.resize(raw_image, (640,480))

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    
    depth_color = cv2.applyColorMap(depth,cv2.COLORMAP_INFERNO)

    # Create split region
    split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255

    # Apply color map to depth image and resize to match raw_image size
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(depth_color, (raw_image.shape[1], raw_image.shape[0]))

    # Concatenate images horizontally
    combined_results = cv2.hconcat([raw_image, split_region, depth_color])

    # Create caption space
    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
    captions = ['Raw image', 'Depth Anything']
    segment_width = w + margin_width

    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

    # Final result with caption
    final_result = cv2.vconcat([caption_space, combined_results])
    
    out_video.write(depth_color)
    cv2.imshow('Depth Anything',final_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out_video.release()
cv2.destroyAllWindows()