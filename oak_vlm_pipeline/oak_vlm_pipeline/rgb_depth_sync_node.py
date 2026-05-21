import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RGBDepthSyncNode(Node):
    def __init__(self):
        super().__init__('rgb_depth_sync_node')

        self.bridge = CvBridge()
        self.camera_info = None

        self.rgb_sub = Subscriber(self, Image, '/oak/rgb/image_raw')
        self.depth_sub = Subscriber(self, Image, '/oak/stereo/image_raw')

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info('RGB + depth sync node started.')

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def synced_callback(self, rgb_msg, depth_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            h, w = rgb_image.shape[:2]
            dh, dw = depth_image.shape[:2]

            if self.camera_info:
                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx = self.camera_info.k[2]
                cy = self.camera_info.k[5]

                self.get_logger().info(
                    f'Synced RGB {w}x{h}, Depth {dw}x{dh}, '
                    f'fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}'
                )
            else:
                self.get_logger().info(
                    f'Synced RGB {w}x{h}, Depth {dw}x{dh}, waiting for camera_info...'
                )

        except Exception as e:
            self.get_logger().error(f'Error processing synced frames: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RGBDepthSyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
