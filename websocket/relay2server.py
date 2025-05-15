import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import roslibpy
import base64
import numpy as np
import cv2
import struct
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import copy

class MultiTopicBridge(Node):
    def __init__(self):
        super().__init__('rosbridge_multi_topic_relay')
        self.color_frame_count = 0
        self.depth_frame_count = 0
        self.IF_OPENCV_SHOW = False
        self.bridge = CvBridge()
        if self.IF_OPENCV_SHOW:
            cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL)
            self.depth_img = None
            self.timer = self.create_timer(0.08, self.timer_callback)

        # publishers
        self.pub_color_compressed = self.create_publisher(
            CompressedImage, '/relay/rgb/compressed', 1)
        self.pub_depth_compressed = self.create_publisher(
            CompressedImage, '/relay/depth/compressed', 1)
        self.pub_depth_raw = self.create_publisher( 
            Image, '/relay/depth', 1)
        self.pub_odom = self.create_publisher(
            Odometry, '/relay/pose/odom', 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        # rosbridge websocket connect
        self.ros_client = roslibpy.Ros(host='35.3.201.75', port=9090)
        self.ros_client.run()

        # subscribe rgb image
        self.topic_color = roslibpy.Topic(
            self.ros_client,
            '/camera/color/image_raw/compressed',
            'sensor_msgs/CompressedImage',
            queue_length=1,
            throttle_rate=100)
        self.topic_color.subscribe(self.cb_color)

        # subscribe depth (compressedDepth)
        self.topic_depth = roslibpy.Topic(
            self.ros_client,
            '/camera/depth/image_rect_raw/compressedDepth',
            'sensor_msgs/CompressedImage',
            queue_length=1,
            throttle_rate=100)
        self.topic_depth.subscribe(self.cb_depth)

        # subscribe odometry
        self.topic_odom = roslibpy.Topic(
            self.ros_client,
            '/camera/pose/sample',
            'nav_msgs/Odometry',
            queue_length=1,
            throttle_rate=20)
        self.topic_odom.subscribe(self.cb_odom)

    # ---------- callbacks ----------
    def cb_color(self, msg):
        self.pub_color_compressed.publish(
            self._to_compressed_image(msg, 'camera_color_frame'))

    def cb_depth(self, msg):
        # compress depth
        self.pub_depth_compressed.publish(
            self._to_compressed_image(msg, 'camera_depth_frame'))

        # decompress raw depth
        raw_depth_msg = self._compressed_depth_to_image(
            msg, 'camera_depth_frame')
        if raw_depth_msg:
            self.pub_depth_raw.publish(raw_depth_msg)

        self.depth_img = copy.deepcopy(raw_depth_msg)


    def timer_callback(self):
        if self.IF_OPENCV_SHOW:
            bridge = CvBridge()
            try:
                depth_image = bridge.imgmsg_to_cv2(self.depth_img, desired_encoding="passthrough")  # 16UC1
            except Exception as e:
                self.get_logger().error(f"CVBridge conversion error: {e}")
                return

            # Normalize for visualization
            depth_image_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_vis = np.uint8(depth_image_vis)
            depth_colormap = cv2.applyColorMap(depth_image_vis, cv2.COLORMAP_JET)

            # Display
            cv2.imshow("Depth Stream", depth_colormap)
            cv2.waitKey(1)
        else:
            pass


    def cb_odom(self, msg):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = msg['header'].get('frame_id', 'camera_frame')
        odom_msg.child_frame_id = msg.get('child_frame_id', '')
        pose = msg['pose']['pose']
        odom_msg.pose.pose.position.x = pose['position']['x']
        odom_msg.pose.pose.position.y = pose['position']['y']
        odom_msg.pose.pose.position.z = pose['position']['z']
        odom_msg.pose.pose.orientation.x = pose['orientation']['x']
        odom_msg.pose.pose.orientation.y = pose['orientation']['y']
        odom_msg.pose.pose.orientation.z = pose['orientation']['z']
        odom_msg.pose.pose.orientation.w = pose['orientation']['w']
        twist = msg['twist']['twist']
        odom_msg.twist.twist.linear.x = twist['linear']['x']
        odom_msg.twist.twist.linear.y = twist['linear']['y']
        odom_msg.twist.twist.linear.z = twist['linear']['z']
        odom_msg.twist.twist.angular.x = twist['angular']['x']
        odom_msg.twist.twist.angular.y = twist['angular']['y']
        odom_msg.twist.twist.angular.z = twist['angular']['z']
        self.pub_odom.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = odom_msg.header.frame_id
        t.child_frame_id = odom_msg.child_frame_id or "camera_link"
        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

    # ---------- helpers ----------
    def _to_compressed_image(self, msg, frame_id='camera_frame'):
        image_msg = CompressedImage()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = frame_id
        image_msg.format = msg['format']
        image_msg.data = base64.b64decode(msg['data'])
        return image_msg

    def _compressed_depth_to_image(self, msg, frame_id='camera_depth_frame'):
        """
        decode `compressedDepth` → `sensor_msgs/Image` (16UC1).
        """
        raw_bytes = base64.b64decode(msg['data'])

        png_start = raw_bytes.find(b'\x89PNG')
        if png_start == -1:
            self.get_logger().warn('PNG header not found in compressedDepth')
            return None

        png_bytes = raw_bytes[png_start:]
        png_arr = np.frombuffer(png_bytes, dtype=np.uint8)
        depth_image = cv2.imdecode(png_arr, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            self.get_logger().warn('Failed to decode compressedDepth PNG')
            return None

        image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = frame_id
        return image_msg

    # ---------- teardown ----------
    def destroy_node(self):
        self.topic_color.unsubscribe()
        self.topic_depth.unsubscribe()
        self.topic_odom.unsubscribe()
        self.ros_client.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MultiTopicBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
