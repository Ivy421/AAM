import math
import platform
import time
from dataclasses import dataclass

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


SOCKETCAN_CHANNELS = {
    "r_piper": "r_piper",
    "l_piper": "l_piper",
}


def _default_interface():
    return "agx_cando" if platform.system() == "Windows" else "socketcan"


def _default_channel(arm_name, interface):
    return SOCKETCAN_CHANNELS.get(arm_name, arm_name)


def _deg_to_rad(values):
    return [math.radians(v) for v in values]


def _endpose_to_sdk(pose):
    x, y, z, rx, ry, rz = pose
    return [x / 1000.0, y / 1000.0, z / 1000.0, *_deg_to_rad([rx, ry, rz])]


def _endpose_from_sdk(pose):
    x, y, z, rx, ry, rz = pose
    return [x * 1000.0, y * 1000.0, z * 1000.0, *[math.degrees(v) for v in [rx, ry, rz]]]


def _msg_value(ret):
    return None if ret is None else ret.msg


@dataclass
class PiperCtrl:
    name: str
    robot: object
    gripper: object | None = None

    def enable(self, joint_index=255):
        return self.robot.enable(joint_index)

    def disable(self, joint_index=255):
        return self.robot.disable(joint_index)

    def reset(self):
        return self.robot.reset()

    def stop(self):
        return self.robot.electronic_emergency_stop()

    def disconnect(self, join_timeout=1.0):
        return self.robot.disconnect(join_timeout=join_timeout)

    def is_ok(self):
        return self.robot.is_ok()

    def get_status(self):
        return _msg_value(self.robot.get_arm_status())

    def get_joint(self):
        joints = _msg_value(self.robot.get_joint_angles())
        return None if joints is None else [math.degrees(v) for v in joints]

    def get_endpose(self):
        pose = _msg_value(self.robot.get_flange_pose())
        return None if pose is None else _endpose_from_sdk(pose)

    def get_tcp_pose(self):
        pose = _msg_value(self.robot.get_tcp_pose())
        return None if pose is None else _endpose_from_sdk(pose)

    def set_speed(self, percent=100):
        return self.robot.set_speed_percent(percent)

    def set_tcp_offset(self, x, y, z, rx, ry, rz):
        return self.robot.set_tcp_offset(_endpose_to_sdk([x, y, z, rx, ry, rz]))

    def move_joint(self, j1, j2, j3, j4, j5, j6):
        return self.robot.move_j(_deg_to_rad([j1, j2, j3, j4, j5, j6]))

    def move_js(self, j1, j2, j3, j4, j5, j6):
        return self.robot.move_js(_deg_to_rad([j1, j2, j3, j4, j5, j6]))

    def move_endpose(self, x, y, z, rx, ry, rz):
        return self.robot.move_p(_endpose_to_sdk([x, y, z, rx, ry, rz]))

    def move_line(self, x, y, z, rx, ry, rz):
        return self.robot.move_l(_endpose_to_sdk([x, y, z, rx, ry, rz]))

    def move_arc(self, start_pose, mid_pose, end_pose):
        return self.robot.move_c(
            _endpose_to_sdk(start_pose),
            _endpose_to_sdk(mid_pose),
            _endpose_to_sdk(end_pose),
        )

    def move_gripper(self, width, force=1.0):
        if self.gripper is None:
            raise RuntimeError("gripper is not initialized")
        return self.gripper.move_gripper_m(value=width / 1000.0, force=force)

    def get_gripper(self):
        if self.gripper is None:
            return None
        return _msg_value(self.gripper.get_gripper_status())

    def disable_gripper(self):
        if self.gripper is None:
            return False
        return self.gripper.disable_gripper()

    def calibrate_gripper(self, timeout=1.0):
        if self.gripper is None:
            return False
        return self.gripper.calibrate_gripper(timeout=timeout)

    def clear_error(self, clear_gripper=True):
        ok = self.robot.clear_joint_error(255)
        if clear_gripper and self.gripper is not None:
            self.gripper.calibrate_gripper(timeout=1.0)
        return ok


def connect_piper(
    arm_name,
    channel=None,
    interface=None,
    firmware=PiperFW.DEFAULT,
    with_gripper=True,
    start_read_thread=True,
    **kwargs,
):
    interface = _default_interface() if interface is None else interface
    channel = _default_channel(arm_name, interface) if channel is None else channel
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=firmware,
        channel=channel,
        interface=interface,
        **kwargs,
    )
    robot = AgxArmFactory.create_arm(cfg)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER) if with_gripper else None
    robot.connect(start_read_thread=start_read_thread)
    time.sleep(0.1)
    return PiperCtrl(arm_name, robot, gripper)


def connect_right(**kwargs):
    return connect_piper("r_piper", **kwargs)


def connect_left(**kwargs):
    return connect_piper("l_piper", **kwargs)


if __name__ == "__main__":
    r_piper = connect_right()
    #r_piper.disable()
    
    r_piper.clear_error()
    r_piper.enable()
    r_piper.set_speed(5)
    r_piper.move_joint(  
0, 20, -30, 0,45, 0 
    )



