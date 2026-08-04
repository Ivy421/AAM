import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="r_piper")
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()

# 张开到 5cm，力 1N
end_effector.move_gripper_m(value=0.04, force=1.0)
time.sleep(1.0)

# 闭合（行程 0）
end_effector.move_gripper_m(value=0.0, force=1.0)
time.sleep(1.0)
end_effector.move_gripper_m(value=0.04, force=1.0)
time.sleep(1.0)

# 闭合（行程 0）
end_effector.move_gripper_m(value=0.0, force=1.0)