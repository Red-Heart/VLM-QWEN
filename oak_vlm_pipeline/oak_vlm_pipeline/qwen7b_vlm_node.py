import json
import time
import cv2
import torch
import rclpy

from datetime import datetime
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen7BVLMNode(Node):
    def __init__(self):
        super().__init__('qwen7b_vlm_node')

        self.bridge = CvBridge()
        self.latest_image = None
        self.is_running = False

        self.tracked_objects = {}
        self.next_object_id = 1

        self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

        self.get_logger().info("Loading Qwen2.5-VL-7B processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.get_logger().info("Loading Qwen2.5-VL-7B model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        self.get_logger().info("Qwen2.5-VL-7B loaded successfully.")
        self.get_logger().info(f"CUDA available: {torch.cuda.is_available()}")
        self.get_logger().info(f"GPU: {torch.cuda.get_device_name(0)}")

        self.image_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            String,
            '/vlm/qwen7b_detections',
            10
        )

        self.timer = self.create_timer(8.0, self.run_inference)

        self.get_logger().info("Qwen7B VLM ROS node started.")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def extract_json(self, text):
        try:
            cleaned = text.replace("```json", "").replace("```", "").strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1 and end > start:
                return json.loads(cleaned[start:end + 1])

        except Exception as e:
            self.get_logger().error(f"JSON parse error: {e}")

        return {
            "analysis_time": datetime.now().isoformat(),
            "scene_summary": "JSON parsing failed.",
            "objects": [],
            "scene_reasoning": {
                "environment_type": "unknown",
                "navigation_analysis": text,
                "obstacle_analysis": "",
                "spatial_analysis": "",
                "risk_assessment": ""
            },
            "suggested_robot_action": {
                "primary_action": "stop",
                "action_reason": "Model output could not be parsed safely.",
                "navigation_priority": "high"
            }
        }

    def clamp_bbox(self, bbox, width, height):
        if not isinstance(bbox, list) or len(bbox) != 4:
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

    def assign_tracking_ids(self, objects):
        updated_objects = []

        for obj in objects:
            label = obj.get("label", "object")
            bbox = obj.get("bbox")

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            assigned_id = None
            best_distance = float("inf")

            for object_id, tracked in self.tracked_objects.items():
                if tracked["label"] != label:
                    continue

                tracked_x, tracked_y = tracked["center"]
                distance = ((center_x - tracked_x) ** 2 + (center_y - tracked_y) ** 2) ** 0.5

                if distance < 120 and distance < best_distance:
                    best_distance = distance
                    assigned_id = object_id

            if assigned_id is None:
                safe_label = label.replace(" ", "_").lower()
                assigned_id = f"{safe_label}_{self.next_object_id}"
                self.next_object_id += 1

            self.tracked_objects[assigned_id] = {
                "label": label,
                "center": (center_x, center_y),
                "last_seen": datetime.now().isoformat()
            }

            obj["tracking_id"] = assigned_id
            obj["bbox_center_px"] = [round(center_x, 1), round(center_y, 1)]

            updated_objects.append(obj)

        return updated_objects

    def run_inference(self):
        if self.is_running:
            return

        if self.latest_image is None:
            self.get_logger().warn("Waiting for RGB image...")
            return

        self.is_running = True

        try:
            start_time = time.time()
            analysis_time = datetime.now().isoformat()

            frame = self.latest_image.copy()
            image_h, image_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_frame)

            prompt = f"""
You are a robotic vision-language perception system.

Current analysis timestamp:
{analysis_time}

Analyze the image and return ONLY valid JSON.

Format:

{{
  "analysis_time": "{analysis_time}",

  "scene_summary": "short scene description",

  "objects": [
    {{
      "label": "object name",
      "bbox": [x1, y1, x2, y2],
      "importance": "low | medium | high",
      "relative_position": "left | center | right",
      "relative_distance": "near | medium | far"
    }}
  ],

  "scene_reasoning": {{
    "environment_type": "workspace | office | lab | room",
    "navigation_analysis": "short navigation analysis",
    "risk_assessment": "short risk analysis"
  }},

  "suggested_robot_action": {{
    "primary_action": "forward | stop | turn_left | turn_right | avoid_obstacle",
    "action_reason": "short explanation"
  }}
}}

Rules:
- Return ONLY JSON
- No markdown
- Maximum 5 objects
- Tight bounding boxes
- Keep reasoning concise
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

            inputs = self.processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
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

            parsed = self.extract_json(output_text)

            cleaned_objects = []

            for obj in parsed.get("objects", []):
                bbox = self.clamp_bbox(
                    obj.get("bbox", []),
                    image_w,
                    image_h
                )

                if bbox is None:
                    continue

                cleaned_objects.append({
                    "label": obj.get("label", "object"),
                    "bbox": bbox,
                    "importance": obj.get("importance", "medium"),
                    "relative_position": obj.get("relative_position", "unknown"),
                    "relative_distance": obj.get("relative_distance", "unknown"),
                    "navigation_effect": obj.get("navigation_effect", "")
                })

            tracked_objects = self.assign_tracking_ids(cleaned_objects)

            parsed["objects"] = tracked_objects
            parsed["analysis_time"] = parsed.get("analysis_time", analysis_time)
            parsed["ros_timestamp"] = datetime.now().isoformat()
            parsed["image_width"] = image_w
            parsed["image_height"] = image_h
            parsed["performance_metrics"] = {
                "inference_time_sec": round(time.time() - start_time, 2),
                "num_detected_objects": len(tracked_objects),
                "model": self.model_id,
                "image_width": image_w,
                "image_height": image_h,
                "cuda_available": torch.cuda.is_available()
            }

            msg = String()
            msg.data = json.dumps(parsed, separators=(',', ':'))
            self.detection_pub.publish(msg)

            pretty_output = json.dumps(parsed, indent=2)
            self.get_logger().info("\n" + pretty_output)

        except Exception as e:
            self.get_logger().error(f"Qwen7B inference error: {e}")

        finally:
            self.is_running = False


def main(args=None):
    rclpy.init(args=args)
    node = Qwen7BVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()