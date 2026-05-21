import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge


class BBoxDepthXYZNode(Node):

    def __init__(self):

        super().__init__('bbox_depth_xyz_node')

        self.bridge = CvBridge()

        self.latest_depth = None
        self.camera_info = None

        self.depth_sub = self.create_subscription(
            Image,
            '/oak/stereo/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            String,
            '/vlm/qwen7b_detections',
            self.detection_callback,
            10
        )

        self.output_pub = self.create_publisher(
            String,
            '/vlm/object_depth_xyz',
            10
        )

        self.get_logger().info(
            'Qwen7B depth fusion node started.'
        )

    def depth_callback(self, msg):

        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

        except Exception as e:
            self.get_logger().error(
                f'Depth conversion error: {e}'
            )

    def camera_info_callback(self, msg):

        self.camera_info = msg

    def scale_bbox(
        self,
        bbox,
        src_w,
        src_h,
        dst_w,
        dst_h
    ):

        x1, y1, x2, y2 = bbox

        sx = dst_w / float(src_w)
        sy = dst_h / float(src_h)

        x1 = int(x1 * sx)
        x2 = int(x2 * sx)

        y1 = int(y1 * sy)
        y2 = int(y2 * sy)

        x1 = max(0, min(dst_w - 1, x1))
        x2 = max(0, min(dst_w - 1, x2))

        y1 = max(0, min(dst_h - 1, y1))
        y2 = max(0, min(dst_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def get_depth(self, depth_image, bbox):

        x1, y1, x2, y2 = bbox

        region = depth_image[y1:y2, x1:x2]

        valid = region[np.isfinite(region)]
        valid = valid[valid > 0]

        if valid.size == 0:
            return None

        depth = float(np.median(valid))

        # mm → metres
        if depth > 20:
            depth = depth / 1000.0

        return depth

    def calculate_xyz(self, bbox, depth):

        if self.camera_info is None:
            return None

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]

        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        x1, y1, x2, y2 = bbox

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return [
            round(float(X), 3),
            round(float(Y), 3),
            round(float(Z), 3)
        ]

    def detection_callback(self, msg):

        if self.latest_depth is None:
            self.get_logger().warn(
                'Waiting for depth image...'
            )
            return

        if self.camera_info is None:
            self.get_logger().warn(
                'Waiting for camera info...'
            )
            return

        try:

            detections = json.loads(msg.data)

            image_w = detections.get("image_width")
            image_h = detections.get("image_height")

            depth_h, depth_w = self.latest_depth.shape[:2]

            output_objects = []

            for obj in detections.get("objects", []):

                label = obj.get("label", "object")
                bbox = obj.get("bbox", [])

                if len(bbox) != 4:
                    continue

                depth_bbox = self.scale_bbox(
                    bbox,
                    image_w,
                    image_h,
                    depth_w,
                    depth_h
                )

                if depth_bbox is None:
                    continue

                depth = self.get_depth(
                    self.latest_depth,
                    depth_bbox
                )

                if depth is None:
                    continue

                xyz = self.calculate_xyz(
                    depth_bbox,
                    depth
                )

                output_objects.append({
                    "label": label,
                    "bbox": bbox,
                    "depth_bbox": depth_bbox,
                    "depth_m": round(depth, 3),
                    "xyz_m": xyz
                })

            output = {
                "objects": output_objects,
                "scene_reasoning":
                    detections.get("scene_reasoning", ""),
                "suggested_action":
                    detections.get("suggested_action", "")
            }

            out_msg = String()
            out_msg.data = json.dumps(output)

            self.output_pub.publish(out_msg)

            self.get_logger().info(out_msg.data)

        except Exception as e:

            self.get_logger().error(
                f'XYZ fusion error: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = BBoxDepthXYZNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()