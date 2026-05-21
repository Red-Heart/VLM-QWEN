import json
import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class QwenBBoxVisualizer(Node):
    def __init__(self):
        super().__init__('qwen_bbox_visualizer')

        self.bridge = CvBridge()

        self.latest_detection = None

        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            String,
            '/vlm/qwen7b_detections',
            self.detection_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/vlm/qwen7b_bbox_image',
            10
        )

        self.get_logger().info("Qwen7B bbox visualizer started.")

    def detection_callback(self, msg):
        try:
            self.latest_detection = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Detection parse error: {e}")

    def draw_text_block(self, image, text, y_start, color=(0, 255, 255)):
        lines = []
        max_len = 70

        words = text.split()

        current = ""

        for word in words:
            if len(current + word) < max_len:
                current += word + " "
            else:
                lines.append(current)
                current = word + " "

        lines.append(current)

        y = y_start

        for line in lines:
            cv2.putText(
                image,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            y += 30

        return y

    def image_callback(self, msg):

        if self.latest_detection is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            objects = self.latest_detection.get("objects", [])

            for obj in objects:

                label = obj.get("label", "object")
                bbox = obj.get("bbox", [])

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            reasoning = self.latest_detection.get(
                "scene_reasoning",
                ""
            )

            action = self.latest_detection.get(
                "suggested_action",
                ""
            )

            inference_time = self.latest_detection.get(
                "inference_time_sec",
                ""
            )

            y = 30

            cv2.putText(
                frame,
                f"Inference: {inference_time} sec",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            y += 40

            y = self.draw_text_block(
                frame,
                f"Reasoning: {reasoning}",
                y,
                (255, 255, 0)
            )

            y += 20

            self.draw_text_block(
                frame,
                f"Action: {action}",
                y,
                (0, 255, 255)
            )

            out_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            self.image_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Visualizer error: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = QwenBBoxVisualizer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()