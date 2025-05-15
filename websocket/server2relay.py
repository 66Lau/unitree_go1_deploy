import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelBridge(Node):
    def __init__(self):
        super().__init__('rosbridge_vel_relay')

        # Create a publisher for /cmd_vel
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer to publish periodically
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.5      # forward speed (m/s)
        msg.linear.y = 0.0
        msg.angular.z = 0.1     # yaw rate (rad/s)

        self.pub_vel.publish(msg)
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}, angular.z={msg.angular.z}')


def main(args=None):
    rclpy.init(args=args)
    node = VelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
