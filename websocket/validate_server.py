import socket
import pickle
import threading
import struct
import time
import numpy as np
import cv2 # For dummy image generation

# --- Configuration ---
SOCKET_HOST = '0.0.0.0'  # Listen on all available interfaces
SOCKET_PORT = 12345      # Same port as your client expects
SERVER_NAME = "StandaloneDummySensorServer"

# --- Global frame counter for dummy data ---
# Use a list to pass by reference to threads, or a Lock with a simple int
frame_counter_lock = threading.Lock()
_frame_count = 0

def get_and_increment_frame_count():
    global _frame_count
    with frame_counter_lock:
        current_count = _frame_count
        _frame_count += 1
    return current_count

def generate_dummy_data():
    """Generates a set of dummy sensor data."""
    frame_id = get_and_increment_frame_count()
    timestamp_ns = time.time_ns()

    # Dummy RGB Image
    rgb_arr = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(rgb_arr, f"Standalone RGB Frame: {frame_id}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)
    cv2.rectangle(rgb_arr, (100 + frame_id % 200, 100), (200 + frame_id % 200, 200), (0,0,255), 3)


    # Dummy Depth Image (16-bit grayscale, representing millimeters)
    depth_arr = np.full((480, 640), 3000, dtype=np.uint16) # Base depth 3 meters
    cv2.circle(depth_arr, (320, 240), 50 + (frame_id % 50), int(1000 + (frame_id % 10) * 100) , -1)
    cv2.putText(depth_arr, f"D:{frame_id}", (30, 450), # Will be noisy on actual depth display
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5000), 1)


    # Dummy Pose
    pose_dict = {
        "header": {
            "stamp_sec": int(timestamp_ns / 1_000_000_000),
            "stamp_nanosec": int(timestamp_ns % 1_000_000_000),
            "frame_id": "dummy_odom"
        },
        "pose": {
            "position": {"x": 1.0 + frame_id * 0.02, "y": 0.5 - frame_id * 0.01, "z": 0.1},
            "orientation": {"x": 0.0, "y": 0.0, "z": np.sin(frame_id * 0.05 / 2.0), "w": np.cos(frame_id * 0.05 / 2.0)}
        }
    }

    return {
        "rgb_image": rgb_arr,
        "depth_image": depth_arr,
        "pose": pose_dict,
        "timestamp_server_ns": timestamp_ns, # Server-side timestamp when data was packed
        "success": True,
        "message": f"Dummy data generated successfully by {SERVER_NAME}."
    }

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


def handle_client_connection(client_socket, client_address):
    """Handles a single client connection."""
    print(f"[{time.strftime('%H:%M:%S')}] Accepted connection from {client_address}")
    try:
        # 1. Wait for a request from the client (e.g., "GET_SENSOR_DATA")
        request = client_socket.recv(1024) # Expecting a small request string
        if not request:
            print(f"[{time.strftime('%H:%M:%S')}] Client {client_address} disconnected before sending request.")
            return

        request_str = request.decode().strip()
        print(f"[{time.strftime('%H:%M:%S')}] Received request: '{request_str}' from {client_address}")

        if request_str == "GET_SENSOR_DATA":
            sensor_data_payload = generate_dummy_data()
            sensor_data_payload = compress_payload(sensor_data_payload)

            pickled_payload = pickle.dumps(sensor_data_payload)
            payload_len = len(pickled_payload)

            # Send the length of the pickled data first (unsigned long long - 8 bytes, network byte order)
            client_socket.sendall(struct.pack('>Q', payload_len))
            # Send the pickled data
            client_socket.sendall(pickled_payload)
            print(f"[{time.strftime('%H:%M:%S')}] Sent {payload_len} bytes of sensor data to {client_address}.")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Unknown request '{request_str}' from {client_address}. Sending error.")
            error_payload = {
                "success": False,
                "message": f"Unknown request: '{request_str}' received by {SERVER_NAME}.",
                "rgb_image": None, "depth_image": None, "pose": None,
                "timestamp_server_ns": time.time_ns()
            }
            pickled_error = pickle.dumps(error_payload)
            error_len = len(pickled_error)
            client_socket.sendall(struct.pack('>Q', error_len))
            client_socket.sendall(pickled_error)

    except ConnectionResetError:
        print(f"[{time.strftime('%H:%M:%S')}] Client {client_address} reset the connection.")
    except socket.timeout:
        print(f"[{time.strftime('%H:%M:%S')}] Socket timeout for {client_address}.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Error handling client {client_address}: {e}", exc_info=True)
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] Closing connection with {client_address}")
        client_socket.close()

def run_server():
    """Main server loop to listen for and handle connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow address reuse immediately after server closes
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((SOCKET_HOST, SOCKET_PORT))
        server_socket.listen(5) # Allow up to 5 queued connections
        print(f"{SERVER_NAME} listening on {SOCKET_HOST}:{SOCKET_PORT}...")

        while True:
            try:
                client_socket, client_address = server_socket.accept()
                # For simplicity, handling client sequentially. 
                # For concurrent clients, start a new thread:
                # client_thread = threading.Thread(target=handle_client_connection, args=(client_socket, client_address))
                # client_thread.daemon = True # So threads exit when main program exits
                # client_thread.start()
                handle_client_connection(client_socket, client_address) # Sequential handling

            except socket.timeout: # server_socket.accept() can timeout if set
                continue 
            except KeyboardInterrupt:
                print(f"\n{SERVER_NAME} received KeyboardInterrupt. Shutting down.")
                break
            except Exception as e:
                print(f"Error in server accept loop: {e}", exc_info=True)
                break # Or continue, depending on desired robustness

    finally:
        print(f"{SERVER_NAME} is shutting down.")
        server_socket.close()

if __name__ == "__main__":
    # You might need to install OpenCV and NumPy if you haven't:
    # pip install opencv-python numpy
    run_server()
