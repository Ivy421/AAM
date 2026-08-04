from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation



import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from Piper import endpose_reachability_safe as ik

DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config/calibration/right_camera/camera_config.npy"

def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = np.load(path, allow_pickle=True).item()
    intrinsic = config["color_intrinsic"]
    camera_matrix = np.array(
        [
            [intrinsic["fx"], 0.0, intrinsic["ppx"]],
            [0.0, intrinsic["fy"], intrinsic["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(intrinsic.get("coeffs", []), dtype=np.float64).reshape(-1)
    if len(coefficients) not in (4, 5, 8, 12, 14):
        coefficients = np.zeros(5, dtype=np.float64)
    return camera_matrix, coefficients, config

camera_matrix, coefficients, config = load_camera(DEFAULT_CAMERA_CONFIG)
print(camera_matrix,coefficients )