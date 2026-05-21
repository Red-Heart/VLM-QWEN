import json
import re
import time

import cv2
import torch
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenVLMNode(Node):
    def __init__(self):
        super().__init__('qwen_vlm_node')

        self.bridge = CvBridge()
        self.latest_image = None
        self.is_running = False

        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        self.get_logger().info("Loading Qwen processor...")

        min_pixels = 256 * 28 * 28
        max_pixels = 384 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )

        self.get_logger().info("Loading Qwen model...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

        self.get_logger().info("Qwen model loaded.")

        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            String,
            '/vlm/qwen_detections',
            10
        )

        # Run every 3 seconds first. Later we can reduce if stable.
        self.timer = self.create_timer(3.0, self.run_inference)

        self.get_logger().info("Qwen VLM node started.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def clean_and_extract_json(self, text):
        try:
            cleaned = text.strip()
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")

            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_text = cleaned[start:end + 1]
                parsed = json.loads(json_text)

                if "objects" not in parsed:
                    parsed["objects"] = []

                return parsed

        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}")

        return {
            "objects": [],
            "scene_reasoning": text,
            "suggested_action": "JSON parsing failed."
        }

    def clamp_bbox(self, bbox, width=640, height=480):
        if len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = bbox

            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(0, min(width - 1, int(x2)))
            y2 = max(0, min(height - 1, int(y2)))

            if x2 <= x1 or y2 <= y1:
                return None

            return [x1, y1, x2, y2]

        except Exception:
            return None

    def clean_objects(self, objects):
        cleaned_objects = []

        for obj in objects:
            label = obj.get("label", "object")
            bbox = obj.get("bbox", [])

            fixed_bbox = self.clamp_bbox(bbox)

            if fixed_bbox is None:
                continue

            cleaned_objects.append({
                "label": label,
                "bbox": fixed_bbox
            })

        return cleaned_objects

    def run_inference(self):
        if self.is_running:
            return

        if self.latest_image is None:
            self.get_logger().warn("Waiting for RGB image...")
            return

        self.is_running = True

        try:
            start_time = time.time()

            frame = cv2.resize(self.latest_image, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_frame)

            prompt = """
Detect up to 2 navigation obstacles.

Return ONLY JSON. No markdown.

Format:
{
  "objects": [
    {"label": "object", "bbox": [x1, y1, x2, y2]}
  ],
  "scene_reasoning": "medium length text reasoning about the scene and any uncertainties",
  "suggested_action": "short"
}

Rules:
- Bounding boxes must tightly enclose the object.
- Do not crop partial objects.
- Only detect objects that are clearly visible and identifiable.
- Do not hallucinate objects that are not clearly present.
- Use absolute pixel coordinates only.
- x must be 0 to 640.
- y must be 0 to 480.
- Do not detect ceiling, air vent, or flag.
- Focus on floor-level robot obstacles: person, chair, table, box, door, wall.
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to("cuda")

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            parsed = self.clean_and_extract_json(output_text)
            parsed["objects"] = self.clean_objects(parsed.get("objects", []))
            parsed["image_width"] = 640
            parsed["image_height"] = 480
            parsed["inference_time_sec"] = round(time.time() - start_time, 2)

            msg = String()
            msg.data = json.dumps(parsed)
            self.detection_pub.publish(msg)

            self.get_logger().info(msg.data)

        except Exception as e:
            self.get_logger().error(f"Qwen inference error: {e}")

        finally:
            self.is_running = False


def main(args=None):
    rclpy.init(args=args)
    node = QwenVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()