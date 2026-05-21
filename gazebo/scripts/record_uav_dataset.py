#!/usr/bin/env python3
"""Record ROS2 camera frames (+ optional pose) for CV dataset generation."""

import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Image


class ImageRecorder(Node):
    def __init__(
        self,
        image_topic: str,
        out_dir: str,
        save_every: int,
        pose_topic: str,
        odom_topic: str,
        navsat_topic: str,
    ) -> None:
        super().__init__("robotx_uav_dataset_recorder")
        self.out_dir = out_dir
        self.save_every = max(1, save_every)
        self.counter = 0
        self.saved = 0
        self.latest_pose = None  # (stamp_ns, x, y, z, yaw_deg)
        self.latest_navsat = None  # (stamp_ns, lat, lon, alt)
        os.makedirs(self.out_dir, exist_ok=True)
        self.pose_csv_path = os.path.join(self.out_dir, "uav_pose_log.csv")
        self.pose_csv = open(self.pose_csv_path, "a", newline="", encoding="utf-8")
        self.pose_writer = csv.writer(self.pose_csv)
        if self.pose_csv.tell() == 0:
            self.pose_writer.writerow(
                [
                    "image_name",
                    "image_stamp_ns",
                    "pose_stamp_ns",
                    "drone_x_m",
                    "drone_y_m",
                    "drone_z_m",
                    "drone_yaw_deg",
                    "navsat_stamp_ns",
                    "drone_lat_deg",
                    "drone_lon_deg",
                    "drone_alt_m",
                ]
            )

        self.sub_image = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.sub_pose = None
        self.sub_odom = None
        self.sub_navsat = None
        if pose_topic:
            self.sub_pose = self.create_subscription(PoseStamped, pose_topic, self.on_pose, 20)
        if odom_topic:
            self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        if navsat_topic:
            self.sub_navsat = self.create_subscription(NavSatFix, navsat_topic, self.on_navsat, 20)

        self.get_logger().info(f"Recording from {image_topic} into {self.out_dir}")
        if pose_topic:
            self.get_logger().info(f"Pose topic: {pose_topic}")
        if odom_topic:
            self.get_logger().info(f"Odom topic: {odom_topic}")
        if navsat_topic:
            self.get_logger().info(f"NavSat topic: {navsat_topic}")

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
        image_name = f"uav_{stamp_ns}.jpg"
        path = os.path.join(self.out_dir, image_name)
        ok = cv2.imwrite(path, frame)
        if ok:
            self.saved += 1
            self.write_pose_row(image_name, stamp_ns)
            if self.saved % 25 == 0:
                self.get_logger().info(f"Saved {self.saved} frames")

    def write_pose_row(self, image_name: str, image_stamp_ns: int) -> None:
        pose_stamp = ""
        x = ""
        y = ""
        z = ""
        yaw = ""
        if self.latest_pose is not None:
            pose_stamp, x, y, z, yaw = self.latest_pose
        nav_stamp = ""
        lat = ""
        lon = ""
        alt = ""
        if self.latest_navsat is not None:
            nav_stamp, lat, lon, alt = self.latest_navsat
        self.pose_writer.writerow([image_name, image_stamp_ns, pose_stamp, x, y, z, yaw, nav_stamp, lat, lon, alt])
        self.pose_csv.flush()

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

    @staticmethod
    def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        # ZYX yaw from quaternion.
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))

    def on_pose(self, msg: PoseStamped) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)
        self.latest_pose = (stamp_ns, p.x, p.y, p.z, yaw)

    def on_odom(self, msg: Odometry) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)
        self.latest_pose = (stamp_ns, p.x, p.y, p.z, yaw)

    def on_navsat(self, msg: NavSatFix) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latest_navsat = (stamp_ns, msg.latitude, msg.longitude, msg.altitude)

    def close(self) -> None:
        try:
            self.pose_csv.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/robotx/uav/camera/image_raw")
    parser.add_argument("--out-dir", default="captures/gazebo_uav")
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--pose-topic", default="", help="Optional geometry_msgs/PoseStamped topic.")
    parser.add_argument("--odom-topic", default="", help="Optional nav_msgs/Odometry topic.")
    parser.add_argument("--navsat-topic", default="", help="Optional sensor_msgs/NavSatFix topic.")
    args = parser.parse_args()

    rclpy.init()
    node = ImageRecorder(
        image_topic=args.topic,
        out_dir=args.out_dir,
        save_every=args.save_every,
        pose_topic=args.pose_topic,
        odom_topic=args.odom_topic,
        navsat_topic=args.navsat_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Finished. Saved {node.saved} frames.")
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
