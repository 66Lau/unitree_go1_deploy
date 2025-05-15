#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import lcm

from rc_command_lcmt_relay import rc_command_lcmt_relay 

LCM_URL     = "udpm://239.255.76.67:7667?ttl=255"
LCM_CHANNEL = "rc_command_relay"
SPEED_TTL   = 1.0

class VelLCMBridge(Node):
    def __init__(self):
        super().__init__('vel_lcm_bridge')

        # ---------------- LCM ----------------
        self.lc = lcm.LCM(LCM_URL)

        self.last_cmd_time = self.get_clock().now()
        self.cmd_lin_x = 0.0
        self.cmd_lin_y = 0.0
        self.cmd_yaw   = 0.0

        # ---------------- ROS ----------------
        self.create_subscription(Twist, '/cmd_vel', self.cb_vel, 10)

        # 10 Hz publish to LCM
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('VelLCMBridge started – forwarding /cmd_vel → LCM')

    def cb_vel(self, msg: Twist):
        self.cmd_lin_x = float(msg.linear.x)
        self.cmd_lin_y = float(msg.linear.y)
        self.cmd_yaw   = float(msg.angular.y)

        self.last_cmd_time = self.get_clock().now()

    def timer_callback(self):
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if elapsed > SPEED_TTL:
            lin_x = lin_y = yaw = 0.0 
        else:
            lin_x, lin_y, yaw = self.cmd_lin_x, self.cmd_lin_y, self.cmd_yaw

        print(f"lin_x: {lin_x:.2f}, lin_y: {lin_y:.2f}, yaw: {yaw:.2f}")

        lcm_msg = rc_command_lcmt_relay()
        lcm_msg.mode = 0
        lcm_msg.left_stick  = [lin_y, lin_x]
        lcm_msg.right_stick = [yaw, 0.0]
        lcm_msg.knobs       = [0.0, 0.0]

        lcm_msg.left_upper_switch = \
        lcm_msg.left_lower_left_switch = \
        lcm_msg.left_lower_right_switch = \
        lcm_msg.right_upper_switch = \
        lcm_msg.right_lower_left_switch = \
        lcm_msg.right_lower_right_switch = 0

        self.lc.publish(LCM_CHANNEL, lcm_msg.encode())

    def destroy_node(self):
        super().destroy_node()

# ---------------- main ----------------
def main(args=None):
    rclpy.init(args=args)
    node = VelLCMBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
