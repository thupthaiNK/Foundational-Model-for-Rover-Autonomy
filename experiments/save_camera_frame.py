"""Save one frame from /exomy/camera/image_raw to Desktop for visual inspection."""
import rclpy
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PIL

class FrameSaver(Node):
    def __init__(self):
        super().__init__('frame_saver')
        self.done = False
        self.create_subscription(Image, '/exomy/camera/image_raw', self.cb, 1)

    def cb(self, msg):
        if self.done:
            return
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if 'bgr' in msg.encoding:
            arr = arr[:, :, ::-1].copy()
        out = '/mnt/c/Users/DELL/Desktop/camera_view.png'
        PIL.fromarray(arr.copy()).save(out)
        print(f'Saved! {msg.width}x{msg.height} {msg.encoding} → {out}')
        self.done = True

rclpy.init()
n = FrameSaver()
t = time.time()
while not n.done and time.time() - t < 10:
    rclpy.spin_once(n, timeout_sec=0.1)
if not n.done:
    print('Timeout — no image from /exomy/camera/image_raw')
n.destroy_node()
rclpy.shutdown()
