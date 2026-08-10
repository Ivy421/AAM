import time
import math
import csv
import os
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


CSV_PATH = '/home/smmg/AAM/config/calibration/mark1/odom.csv'


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class Mark1BaseController(Node):
    def __init__(self):
        super().__init__('mark1_base_controller')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.x = None
        self.y = None
        self.yaw = None

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def wait_for_odom(self, timeout_sec=5.0):
        start_time = time.time()

        while rclpy.ok() and self.x is None:
            time.sleep(0.05)

            if time.time() - start_time > timeout_sec:
                raise TimeoutError("Waiting for /odom timed out. Check Mark1 bringup and base connection.")

    def send_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        msg = Twist()

        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = wz

        self.cmd_pub.publish(msg)

    def stop(self, repeat=10):
        for _ in range(repeat):
            self.send_cmd(0.0, 0.0, 0.0)
            time.sleep(0.05)

    def move_for_velocity(self, vx=0.0, vy=0.0, wz=0.0, duration=2.0, rate_hz=20):
        if duration <= 0:
            self.stop()
            return

        dt = 1.0 / rate_hz
        start_time = time.time()

        try:
            while rclpy.ok() and (time.time() - start_time) < duration:
                self.send_cmd(vx=vx, vy=vy, wz=wz)
                time.sleep(dt)

        finally:
            self.stop()

    def move_distance(self, distance_m, speed_mps=0.08, axis='x', rate_hz=20):

        if axis not in ['x', 'y']:
            raise ValueError("axis not in x or y")

        self.wait_for_odom()

        start_x = self.x
        start_y = self.y

        direction = 1.0 if distance_m >= 0 else -1.0
        speed = abs(speed_mps) * direction

        dt = 1.0 / rate_hz

        try:
            while rclpy.ok():
                moved = math.sqrt(
                    (self.x - start_x) ** 2 +
                    (self.y - start_y) ** 2
                )

                if moved >= abs(distance_m):
                    break

                if axis == 'x':
                    self.send_cmd(vx=speed, vy=0.0, wz=0.0)
                else:
                    self.send_cmd(vx=0.0, vy=speed, wz=0.0)

                time.sleep(dt)

        finally:
            self.stop()

    def move_x(self, distance_m, speed_mps=0.08, rate_hz=20):
        self.move_distance(
            distance_m=distance_m,
            speed_mps=speed_mps,
            axis='x',
            rate_hz=rate_hz
        )

    def move_y(self, distance_m, speed_mps=0.06, rate_hz=20):
        self.move_distance(
            distance_m=distance_m,
            speed_mps=speed_mps,
            axis='y',
            rate_hz=rate_hz
        )

    def turn_rad(self, angle_rad, angular_speed=0.3, rate_hz=20):
        self.wait_for_odom()

        direction = 1.0 if angle_rad >= 0 else -1.0
        wz = abs(angular_speed) * direction

        accumulated_angle = 0.0
        last_yaw = self.yaw

        dt = 1.0 / rate_hz

        try:
            while rclpy.ok():
                current_yaw = self.yaw
                delta_yaw = normalize_angle(current_yaw - last_yaw)

                accumulated_angle += delta_yaw
                last_yaw = current_yaw

                if abs(accumulated_angle) >= abs(angle_rad):
                    break

                self.send_cmd(vx=0.0, vy=0.0, wz=wz)
                time.sleep(dt)

        finally:
            self.stop()

    def turn_deg(self, angle_deg, angular_speed=0.3, rate_hz=20):
        self.turn_rad(
            angle_rad=math.radians(angle_deg),
            angular_speed=angular_speed,
            rate_hz=rate_hz
        )


class OdomSaver(Node):
    def __init__(self, csv_path=CSV_PATH):
        super().__init__('mark1_odom_saver')

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.csv_path = csv_path
        self.header = [
            'sample_id',
            'saved_wall_time',
            'timestamp',
            'x',
            'y',
            'yaw',
            'vx',
            'vy',
            'wz'
        ]

        file_is_empty = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)

        if file_is_empty:
            self.writer.writerow(self.header)
            self.csv_file.flush()

        self.next_sample_index = self._get_next_sample_index()
        self.latest_odom = None

        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info(f'Ready to append selected /odom to {self.csv_path}')

    def _get_next_sample_index(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            return 1

        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

            valid_rows = [row for row in rows[1:] if len(row) > 0]
            return len(valid_rows) + 1

        except Exception:
            return 1

    def odom_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        self.latest_odom = {
            'timestamp': stamp,
            'x': x,
            'y': y,
            'yaw': yaw,
            'vx': vx,
            'vy': vy,
            'wz': wz
        }

    def wait_for_latest_odom(self, timeout_sec=5.0):
        start_time = time.time()

        while rclpy.ok() and self.latest_odom is None:
            time.sleep(0.05)

            if time.time() - start_time > timeout_sec:
                raise TimeoutError("Waiting for OdomSaver /odom message timed out.")

    def save_once(self, sample_id=None):
        self.wait_for_latest_odom()

        if sample_id is None:
            sample_id = f'sample_{self.next_sample_index:06d}'
            self.next_sample_index += 1

        saved_wall_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        odom = self.latest_odom

        self.writer.writerow([
            sample_id,
            saved_wall_time,
            odom['timestamp'],
            odom['x'],
            odom['y'],
            odom['yaw'],
            odom['vx'],
            odom['vy'],
            odom['wz']
        ])

        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())

        self.get_logger().info(
            f"Appended odom [{sample_id}]: "
            f"x={odom['x']:.4f}, y={odom['y']:.4f}, yaw={odom['yaw']:.4f}, "
            f"vx={odom['vx']:.4f}, vy={odom['vy']:.4f}, wz={odom['wz']:.4f}"
        )

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(f'Odom csv closed: {self.csv_path}')


def mark1_ctrl(vx=0.0, vy=0.0, wz=0.0, duration=2.0, rate_hz=20):
    """
    Simple API for Mark1 velocity control.

    vx: forward velocity, m/s
    vy: left velocity, m/s
    wz: yaw angular velocity, rad/s
    duration: motion duration, s
    rate_hz: command publish frequency
    """
    rclpy.init()

    base = Mark1BaseController()
    executor = MultiThreadedExecutor()
    executor.add_node(base)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        base.wait_for_odom()
        base.move_for_velocity(vx=vx, vy=vy, wz=wz, duration=duration, rate_hz=rate_hz)

    finally:
        base.stop()
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        base.destroy_node()
        rclpy.shutdown()


def main():
    mark1_ctrl(vx=0.0, vy=0.0, wz=0.0, duration=1, rate_hz=20)


if __name__ == '__main__':
    main()
