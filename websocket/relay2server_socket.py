import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import socket
import pickle
import threading
import struct
import base64
# --- Configuration ---
SOCKET_HOST = '0.0.0.0'  # Listen on all available interfaces
SOCKET_PORT = 12345
MAX_BUFFER_SIZE = 4096 # For receiving request from client, and sending pickled data length

class SensorDataManager:
    """Holds the latest sensor data and provides thread-safe access."""
    def __init__(self, logger):
        self.latest_rgb_cv_image = None
        self.latest_depth_cv_image = None
        self.latest_pose_dict = None
        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()
        self.logger = logger
        self.data_ready = False # Flag to indicate if all data has been received at least once

    def rgb_callback(self, msg: CompressedImage):
        try:
            cv_img = self.cv_bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_rgb_cv_image = cv_img
            self.check_data_ready()
        except Exception as e:
            self.logger.error(f'RGB callback error: {e}')

    def depth_callback(self, msg: CompressedImage):
        try:
            png_bytes = msg.data[12:]
            depth      = cv2.imdecode(
                        np.frombuffer(png_bytes, np.uint8),
                        cv2.IMREAD_UNCHANGED)  
            # depth = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.latest_depth_cv_image = depth
            self.check_data_ready()
        except Exception as e:
            self.logger.error(f'Depth callback error: {e}')


    def pose_callback(self, msg: Odometry):
        with self.lock:
            self.latest_pose_dict = {
                "header": {
                    "stamp_sec": msg.header.stamp.sec,
                    "stamp_nanosec": msg.header.stamp.nanosec,
                    "frame_id": msg.header.frame_id
                },
                "pose": {
                    "position": {"x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y, "z": msg.pose.pose.position.z},
                    "orientation": {"x": msg.pose.pose.orientation.x, "y": msg.pose.pose.orientation.y,
                                    "z": msg.pose.pose.orientation.z, "w": msg.pose.pose.orientation.w}
                }
            }
        # self.logger.info('Received Pose', throttle_duration_sec=5)
        self.check_data_ready()

    def check_data_ready(self):
        if not self.data_ready:
            with self.lock:
                if self.latest_rgb_cv_image is not None and \
                   self.latest_depth_cv_image is not None and \
                   self.latest_pose_dict is not None:
                    self.data_ready = True
                    self.logger.info("Initial set of sensor data received. Server is ready for client requests.")

    def get_latest_data(self):
        with self.lock:
            if not self.data_ready:
                return None
            # Return copies to avoid issues if data is updated while pickling
            return {
                "rgb_image": self.latest_rgb_cv_image.copy() if self.latest_rgb_cv_image is not None else None,
                "depth_image": self.latest_depth_cv_image.copy() if self.latest_depth_cv_image is not None else None,
                "pose": self.latest_pose_dict.copy() if self.latest_pose_dict is not None else None,
                "timestamp_server_ns": time.time_ns()
            }

    def _compressed_depth_to_image(self, msg, frame_id='camera_depth_frame'):
        """
        decode `compressedDepth` → `sensor_msgs/Image` (16UC1).
        """
        raw_bytes = base64.b64decode(msg.data)

        png_start = raw_bytes.find(b'\x89PNG')
        # if png_start == -1:
        #     self.get_logger().warn('PNG header not found in compressedDepth')
        #     return None

    def _to_compressed_image(self, msg, frame_id='camera_frame'):
        image_msg = CompressedImage()
        image_msg.header.stamp = msg.header.stamp
        image_msg.header.frame_id = frame_id
        image_msg.format = msg.format
        image_msg.data = base64.b64decode(msg.data)
        return image_msg

class SensorServerNode(Node):
    def __init__(self, data_manager: SensorDataManager):
        super().__init__('sensor_socket_server_node')
        self.data_manager = data_manager
        self.get_logger().info("Sensor Socket Server ROS Node Started")

        # --- Subscribers to sensor topics ---
        self.rgb_subscriber = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed', self.data_manager.rgb_callback, 10)
        self.depth_subscriber = self.create_subscription(
            CompressedImage, '/camera/aligned_depth_to_color/image_raw/compressedDepth', self.data_manager.depth_callback, 10)
        self.pose_subscriber = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.data_manager.pose_callback,
            qos_profile_sensor_data) 

        # --- Dummy Publishers (to simulate sensor data for this example) ---
        # self.dummy_rgb_pub = self.create_publisher(CompressedImage, 'sensor/rgb_image', 10)
        # self.dummy_depth_pub = self.create_publisher(CompressedImage, 'sensor/depth_image', 10)
        # self.dummy_pose_pub = self.create_publisher(Odometry, 'sensor/pose', 10)
        # self.timer = self.create_timer(0.1, self.publish_dummy_data) # Publish dummy data at 10Hz
        self.frame_count = 0
        self.get_logger().info("Dummy sensor publishers started.")

    def publish_dummy_data(self):
        now = self.get_clock().now().to_msg()

        rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb_arr, f"RGB ROS Frame: {self.frame_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rgb_msg = self.data_manager.cv_bridge.cv2_to_imgmsg(rgb_arr, encoding="bgr8")
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_rgb_optical_frame"
        self.dummy_rgb_pub.publish(rgb_msg)

        depth_arr = np.full((480, 640), 1500, dtype=np.uint16) # 1.5 meters in mm
        # Add some pattern to depth
        cv2.circle(depth_arr, (320, 240), 50, int(1000 + (self.frame_count % 10) * 50) , -1) # Varying depth circle
        depth_msg = self.data_manager.cv_bridge.cv2_to_imgmsg(depth_arr, encoding="16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        self.dummy_depth_pub.publish(depth_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position = Point(x=1.0 + self.frame_count * 0.01, y=2.0, z=0.0)
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.dummy_pose_pub.publish(pose_msg)

        self.frame_count += 1

def compress_payload(payload_dict):
    """
    Compresses 'rgb_image' and 'depth_image' in the payload dictionary
    using lossless PNG encoding. Other items are left as is.
    Args:
        payload_dict (dict): The dictionary containing sensor data.
                             Expected to have 'rgb_image' and/or 'depth_image'
                             as NumPy arrays.
    Returns:
        dict: A new dictionary with images replaced by their PNG-compressed bytes.
              Original metadata like 'shape' and 'dtype' for images are stored
              to aid in perfect reconstruction if needed, though cv2.imdecode
              with IMREAD_UNCHANGED often handles this for PNG.
    """
    compressed_dict = payload_dict.copy() # Work on a copy
    # Compress RGB Image
    if 'rgb_image' in compressed_dict and isinstance(compressed_dict['rgb_image'], np.ndarray):
        rgb_image_np = compressed_dict['rgb_image']
        success, encoded_image = cv2.imencode('.png', rgb_image_np)
        if success:
            compressed_dict['rgb_image'] = encoded_image.tobytes() # Store as bytes
            # Store metadata for potential precise reconstruction if imdecode isn't enough
            # (though for PNG and typical image types, it usually is)
            compressed_dict['rgb_image_shape'] = rgb_image_np.shape
            compressed_dict['rgb_image_dtype'] = str(rgb_image_np.dtype)
            compressed_dict['rgb_image_compressed_format'] = 'png'
        else:
            print("Warning: RGB image PNG encoding failed.")
            # Optionally, remove the key or send uncompressed with a flag
            compressed_dict['rgb_image'] = None # Or handle error appropriately
    # Compress Depth Image
    if 'depth_image' in compressed_dict and isinstance(compressed_dict['depth_image'], np.ndarray):
        depth_image_np = compressed_dict['depth_image']
        # PNG supports 8-bit and 16-bit grayscale.
        # If depth_image_np is float32, PNG won't directly store it losslessly as float.
        # It would typically be converted to uint16 or uint8.
        # For this example, we assume depth_image_np is uint8 or uint16.
        if depth_image_np.dtype not in [np.uint8, np.uint16]:
            print(f"Warning: Depth image dtype {depth_image_np.dtype} might not be perfectly preserved by PNG. "
                  "Consider converting to uint16 if precision loss is acceptable, or use a different compression.")
        success, encoded_image = cv2.imencode('.png', depth_image_np)
        if success:
            compressed_dict['depth_image'] = encoded_image.tobytes() # Store as bytes
            compressed_dict['depth_image_shape'] = depth_image_np.shape
            compressed_dict['depth_image_dtype'] = str(depth_image_np.dtype)
            compressed_dict['depth_image_compressed_format'] = 'png'
        else:
            print("Warning: Depth image PNG encoding failed.")
            compressed_dict['depth_image'] = None
    return compressed_dict


def socket_server_thread(data_manager: SensorDataManager, logger):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow address reuse
    try:
        server_socket.bind((SOCKET_HOST, SOCKET_PORT))
        server_socket.listen(1) # Listen for one connection at a time
        logger.info(f"Socket server listening on {SOCKET_HOST}:{SOCKET_PORT}")

        while rclpy.ok(): # Keep running as long as ROS is okay
            try:
                conn, addr = server_socket.accept()
                logger.info(f"Socket connection from {addr}")
                with conn:
                    # 1. Wait for a request from the client
                    request = conn.recv(1024) # Expecting a small request string
                    if not request:
                        logger.warn("Client disconnected before sending request.")
                        continue
                    
                    request_str = request.decode().strip()
                    logger.info(f"Received request: '{request_str}'")

                    if request_str == "GET_SENSOR_DATA":
                        sensor_data = data_manager.get_latest_data()


                        if sensor_data and \
                           sensor_data["rgb_image"] is not None and \
                           sensor_data["depth_image"] is not None and \
                           sensor_data["pose"] is not None:
                            
                            # Add a success flag to the payload
                            sensor_data["success"] = True
                            sensor_data["message"] = "Data acquired successfully."

                            sensor_data = compress_payload(sensor_data)
                            pickled_data = pickle.dumps(sensor_data)
                            data_len = len(pickled_data)

                            # Send the length of the pickled data first (unsigned long long - 8 bytes)
                            conn.sendall(struct.pack('>Q', data_len))
                            # Send the pickled data
                            conn.sendall(pickled_data)
                            logger.info(f"Sent {data_len} bytes of sensor data to {addr}.")
                        else:
                            error_payload = {
                                "success": False,
                                "message": "Sensor data not yet fully available on server.",
                                "rgb_image": None, "depth_image": None, "pose": None,
                                "timestamp_server_ns": time.time_ns()
                            }
                            pickled_error = pickle.dumps(error_payload)
                            error_len = len(pickled_error)
                            conn.sendall(struct.pack('>Q', error_len))
                            conn.sendall(pickled_error)
                            logger.warn(f"Sent error (data not ready) to {addr}.")
                    else:
                        logger.warn(f"Unknown request '{request_str}' from {addr}. Closing connection.")
                        # Optionally send an error message back
                        error_payload = {
                            "success": False, "message": f"Unknown request: {request_str}",
                             "rgb_image": None, "depth_image": None, "pose": None,
                             "timestamp_server_ns": time.time_ns()
                        }
                        pickled_error = pickle.dumps(error_payload)
                        error_len = len(pickled_error)
                        conn.sendall(struct.pack('>Q', error_len))
                        conn.sendall(pickled_error)

            except ConnectionResetError:
                logger.warn(f"Client {addr} reset the connection.")
            except socket.timeout:
                logger.warn("Socket accept timed out (should not happen with blocking accept).")
            except Exception as e:
                logger.error(f"Socket server per-client error: {e}", exc_info=True)
            finally:
                if 'conn' in locals() and conn: # Make sure conn was defined
                    try:
                        conn.close()
                    except Exception as e_close:
                        logger.error(f"Error closing client connection: {e_close}")
                logger.info(f"Connection with {addr if 'addr' in locals() else 'unknown'} closed. Waiting for new connection.")

    except Exception as e:
        logger.error(f"Socket server main loop error: {e}", exc_info=True)
    finally:
        logger.info("Socket server thread shutting down.")
        server_socket.close()


def main(args=None):
    rclpy.init(args=args)
    
    # Using a temporary node to get a logger for SensorDataManager
    temp_node_for_logger = rclpy.create_node('temp_logger_node_sensor_server')
    data_manager = SensorDataManager(temp_node_for_logger.get_logger())
    temp_node_for_logger.destroy_node() # Don't need it anymore

    sensor_server_ros_node = SensorServerNode(data_manager)
    
    # Start socket server in a separate thread
    # Pass the logger from the ROS node to the socket thread for consistent logging
    sock_thread = threading.Thread(target=socket_server_thread, args=(data_manager, sensor_server_ros_node.get_logger()), daemon=True)
    sock_thread.start()

    try:
        rclpy.spin(sensor_server_ros_node)
    except KeyboardInterrupt:
        sensor_server_ros_node.get_logger().info("Keyboard interrupt received, shutting down ROS node.")
    except Exception as e:
        sensor_server_ros_node.get_logger().error(f"Exception in rclpy.spin: {e}", exc_info=True)
    finally:
        sensor_server_ros_node.get_logger().info("Shutting down sensor server ROS node...")
        sensor_server_ros_node.destroy_node()
        rclpy.shutdown()
        # The socket thread is a daemon, so it will exit when the main thread (rclpy.spin) exits.
        # Or, if rclpy.ok() is used in its loop, it will exit when rclpy is shutdown.
        # We can also add a more explicit shutdown mechanism if needed (e.g., using an event).
        if sock_thread.is_alive():
            # Give it a moment to close gracefully based on rclpy.ok()
            sock_thread.join(timeout=2.0) 
            if sock_thread.is_alive():
                print("Socket thread did not shut down gracefully via rclpy.ok(). It will be terminated as daemon.")
        print("Sensor server fully shut down.")


if __name__ == '__main__':
    main()