import time
import math
import csv
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


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
                raise TimeoutError("等待 /odom 超时，请检查 bringup 是否启动、底盘是否连接。")

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
        """
        按速度和时间控制底盘运动。

        vx: x方向速度，单位 m/s
        vy: y方向速度，单位 m/s，麦轮可用
        wz: z轴角速度，单位 rad/s
        duration: 持续时间，单位 s
        """

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
        """
        按距离运动。

        axis='x'：前进/后退
        axis='y'：左移/右移，麦轮可用
        """

        if axis not in ['x', 'y']:
            raise ValueError("axis 只能是 'x' 或 'y'")

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
    def __init__(self, csv_path='/home/smmg/AAM/config/calibration/mark1/odom.csv'):
        super().__init__('mark1_odom_saver')

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.csv_path = csv_path
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'timestamp',
            'x',
            'y',
            'yaw',
            'vx',
            'vy',
            'wz'
        ])

        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info(f'Start saving /odom to {self.csv_path}')

    def odom_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        self.writer.writerow([
            stamp,
            x,
            y,
            yaw,
            vx,
            vy,
            wz
        ])

        self.csv_file.flush()

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(f'Odom csv saved to {self.csv_path}')


def main():
    rclpy.init()

    base = Mark1BaseController()
    odom_saver = OdomSaver(
        csv_path='/home/smmg/AAM/config/calibration/mark1/odom.csv'
    )

    executor = MultiThreadedExecutor()
    executor.add_node(base)
    executor.add_node(odom_saver)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        base.wait_for_odom()
        print("成功接收到 /odom，开始运动并保存里程计数据。")

        time.sleep(1.0)

        # 1. x方向运动：前进 2 秒
        base.move_for_velocity(
            vx=0.05,
            vy=0.0,
            wz=0.0,
            duration=2.0,
            rate_hz=20
        )

        time.sleep(1.0)

        # 2. y方向运动：横移 2 秒，麦轮底盘可用
        base.move_for_velocity(
            vx=0.0,
            vy=-0.05,
            wz=0.0,
            duration=2.0,
            rate_hz=20
        )

        time.sleep(1.0)

        # 3. wz旋转：顺时针旋转 2 秒
        base.move_for_velocity(
            vx=0.0,
            vy=0.0,
            wz=-0.3,
            duration=2.0,
            rate_hz=20
        )

        time.sleep(1.0)

        # 最后停车
        base.stop()

        print("运动结束，odom 数据已保存。")

    except KeyboardInterrupt:
        print("用户中断，正在停车。")

    finally:
        base.stop()

        time.sleep(1)

        executor.shutdown()
        spin_thread.join(timeout=1.0)

        odom_saver.close()

        base.destroy_node()
        odom_saver.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()