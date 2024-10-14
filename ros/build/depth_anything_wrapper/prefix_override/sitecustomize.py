import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sonia/Depth-Anything/ros/install/depth_anything_wrapper'
