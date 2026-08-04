"""Estimate the AprilTag pose and Piper grasp pose for the glue brush."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glue import glue_servopick_endpose as base


BRUSH_GRASP_POINT_TAG_MM = np.array([-50, 0 , -20], dtype=float)      
BRUSH_TAG_FAMILY = "tag36h11"
BRUSH_TAG_X_FROM_DETECTED_AXIS = "+y"  # 为什么这里是+y?
DEFAULT_OUTPUT_NAME = "glue_brush_pick_endpose.json"


_base_parse_args = base.parse_args


def parse_args():
    args = _base_parse_args()
    args.tag_family = BRUSH_TAG_FAMILY
    args.needle_axis = BRUSH_TAG_X_FROM_DETECTED_AXIS
    if args.run_dir is not None and args.output is None:
        args.output = args.run_dir.expanduser().resolve() / "pickplace" / DEFAULT_OUTPUT_NAME
    return args


def main():
    base.GRASP_POINT_TAG_MM = BRUSH_GRASP_POINT_TAG_MM
    base.parse_args = parse_args
    base.main()


if __name__ == "__main__":
    main()
