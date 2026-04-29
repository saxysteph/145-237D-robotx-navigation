#!/usr/bin/env python3
"""Record ROS2 camera topic frames to disk for CV dataset generation."""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageRecorder(Node):
    def __init__(self, topic: str, out_dir: str, save_every: int) -> None:
        super().__init__("robotx_uav_dataset_recorder")
        self.out_dir = out_dir
        self.save_every = max(1, save_every)
        self.counter = 0
        self.saved = 0
        os.makedirs(self.out_dir, exist_ok=True)
        self.sub = self.create_subscription(Image, topic, self.on_image, 10)
        self.get_logger().info(f"Recording from {topic} into {self.out_dir}")

    def on_image(self, msg: Image) -> None:
        self.counter += 1
        if self.counter % self.save_every != 0:
            return

        frame = self.to_bgr(msg)
        if frame is None:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
            return

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns == 0:
            stamp_ns = int(datetime.utcnow().timestamp() * 1e9)
        path = os.path.join(self.out_dir, f"uav_{stamp_ns}.jpg")
        ok = cv2.imwrite(path, frame)
        if ok:
            self.saved += 1
            if self.saved % 25 == 0:
                self.get_logger().info(f"Saved {self.saved} frames")

    def to_bgr(self, msg: Image):
        h, w = msg.height, msg.width
        if h <= 0 or w <= 0:
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        encoding = msg.encoding.lower()

        if encoding == "rgb8":
            rgb = arr.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if encoding == "bgr8":
            return arr.reshape((h, w, 3))
        if encoding == "rgba8":
            rgba = arr.reshape((h, w, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            bgra = arr.reshape((h, w, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/robotx/uav/camera/image_raw")
    parser.add_argument("--out-dir", default="captures/gazebo_uav")
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth frame")
    args = parser.parse_args()

    rclpy.init()
    node = ImageRecorder(args.topic, args.out_dir, args.save_every)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Finished. Saved {node.saved} frames.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
