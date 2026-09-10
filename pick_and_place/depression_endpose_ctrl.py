"""Execute repair-block pick and fix after glue-brush task."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from Piper.piper_ctrl import connect_right
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from Piper.endpose_reachability_safe import reachability_test, load_arm_model, frame_pose, get_safe_bounds, DEFAULT_EE_FRAME


TRAVEL_SPEED = 15
APPROACH_SPEED = 8
GRAB_SPEED = 5
LEAVE_SPEED = 8
FIX_SPEED = 5
RETREAT_SPEED = 8

GRIPPER_CLOSE_MM = -6
GRIPPER_OPEN_MM = 30
GRIPPER_FORCE = 1.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path): return json.loads(path.read_text(encoding="utf-8"))


def move_joint(piper, joints, speed):
    piper.set_speed(speed)
    piper.move_joint(*joints)

def move_line(piper, endpose, speed):
    piper.set_speed(speed)
    piper.move_line(*endpose)

def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    pickplace_dir = run_dir / "pickplace"

    glue_pick = load_json(pickplace_dir / "glue_brush_pick_endpose.json")
    repair = load_json(pickplace_dir / "pick_place_endpose.json")

    glue_prepick = glue_pick["prepick_joint_degrees"]
    grab_joint = repair["grab_joint_degrees"]
    grab_endpose = repair['grab_endpose']
    leave_endpose = grab_endpose
    leave_endpose[2] +=20
    #pre_grab_joint = repair["pre_grab_joint_degrees"]
    leave_clearance = repair["leave_clearance_joint_degrees"]
    pre_fix_joint = repair["pre_fix_joint_degrees"]
    fix_joint = repair["fix_joint_degrees"]

    if not grab_joint or not fix_joint: raise RuntimeError("Grab or fix joint degrees are invalid.")
    if not pre_fix_joint: raise RuntimeError("pre_fix_joint_degrees is invalid.")

    printer_clearance = list(grab_joint)
    printer_clearance[0] = 110.0

    brush_clearance = list(printer_clearance)
    brush_clearance[0] = 85.0
    brush_clearance[1] = 70.0

    fix_clearance = list(pre_fix_joint)
    fix_clearance[1] = 80.0
    fix_clearance[2] = -40.0

    piper = connect_right()

    try:
        piper.clear_error(clear_gripper=False)
        piper.enable()

        ## 这一段计划移动到 glue task中的脚本
        print("Glue pick -> glue prepick")
        piper.move_gripper(GRIPPER_OPEN_MM, force=GRIPPER_FORCE)
        move_joint(piper, glue_prepick, TRAVEL_SPEED)
        
        time.sleep(3)
        ##

        print("Glue prepick -> printer clearance")
        move_joint(piper, printer_clearance, TRAVEL_SPEED)
        time.sleep(2)

        #if pre_grab_joint:
        #    print("Printer clearance -> pre-grab")
        #    move_joint(piper, pre_grab_joint, APPROACH_SPEED)
        #else:
        #    print("Pre-grab unavailable -> directly approach grab")

        print("Move to grab")
        move_joint(piper, grab_joint, GRAB_SPEED)
        time.sleep(5)

        print("夹住 repair model")
        piper.move_gripper(GRIPPER_CLOSE_MM, force=GRIPPER_FORCE)
        time.sleep(1)

        #if pre_grab_joint:
        #    print("Leave printer through pre-grab")
        #    move_joint(piper, pre_grab_joint, LEAVE_SPEED)
        #else:
        #    print("Search leave clearance")
        #    leave_clearance = find_leave_clearance(piper)
        #    move_joint(piper, leave_clearance, LEAVE_SPEED)
        print('pull up model')
        move_line(piper, leave_endpose, TRAVEL_SPEED)
        time.sleep(4)

        print("Move to brush clearance")
        move_joint(piper, brush_clearance, TRAVEL_SPEED)
        time.sleep(4)

        print("Move to fix clearance")
        move_joint(piper, fix_clearance, TRAVEL_SPEED)
        time.sleep(4)

        print("Move to pre-fix")
        move_joint(piper, pre_fix_joint, APPROACH_SPEED)
        time.sleep(4)

        print("Install repair model")
        move_joint(piper, fix_joint, FIX_SPEED)
        time.sleep(5)

        print("Release repair model")
        piper.move_gripper(GRIPPER_OPEN_MM, force=GRIPPER_FORCE)
        time.sleep(2)

        print("Retreat to pre-fix")
        move_joint(piper, pre_fix_joint, RETREAT_SPEED)
        time.sleep(1)

        print("Retreat to fix clearance")
        move_joint(piper, fix_clearance, TRAVEL_SPEED)
        time.sleep(1)

        print("Repair pick-and-fix complete")

    finally:
        piper.disconnect()


if __name__ == "__main__":
    main()
