import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =========================
# User parameters
# =========================

INPUT_PATH = Path("E:/HKUSTGZ/AAM/construction/data/reachable_scanpose.json")

OUTPUT_PATH = Path("E:/HKUSTGZ/AAM/construction/data/target_scanpose.json")
TARGET_NUM = 20

# If same/similar radius-yaw-pitch has many roll values,
# keep only the one whose roll is closest to 0 deg.
RADIUS_GROUP_TOL = 1e-3   # meter
YAW_GROUP_TOL = 1e-3      # degree
PITCH_GROUP_TOL = 1e-3    # degree

# Distance weights for farthest-point sampling.
# yaw/pitch are more important for view diversity; roll is less important.
W_RADIUS = 1.0
W_YAW = 1.3
W_PITCH = 1.3
W_ROLL = 0.5


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input json must be a list of scan-pose dictionaries.")
    return data


def save_records(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees, considering wrap-around."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def group_key(record: Dict[str, Any]) -> Tuple[int, int, int]:
    """Group by similar radius/yaw/pitch. Roll is not included because roll-only variants are often redundant."""
    radius = float(record["radius"])
    yaw = float(record["yaw"])
    pitch = float(record["pitch"])
    return (
        round(radius / RADIUS_GROUP_TOL),
        round(yaw / YAW_GROUP_TOL),
        round(pitch / PITCH_GROUP_TOL),
    )


def roll_abs_score(record: Dict[str, Any]) -> float:
    """Prefer roll closer to 0 deg when radius/yaw/pitch are duplicated."""
    return angle_diff_deg(float(record["roll"]), 0.0)


def deduplicate_by_radius_yaw_pitch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_group: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for rec in records:
        key = group_key(rec)
        if key not in best_by_group:
            best_by_group[key] = rec
        else:
            # Keep the roll closest to 0 deg.
            if roll_abs_score(rec) < roll_abs_score(best_by_group[key]):
                best_by_group[key] = rec
    return list(best_by_group.values())


def param_distance(a: Dict[str, Any], b: Dict[str, Any], ranges: Dict[str, float]) -> float:
    """
    Distance in normalized radius/yaw/pitch/roll parameter space.
    This is not Cartesian distance. It measures scan-view diversity.
    """
    dr = abs(float(a["radius"]) - float(b["radius"])) / ranges["radius"]
    dyaw = angle_diff_deg(float(a["yaw"]), float(b["yaw"])) / ranges["yaw"]
    dpitch = abs(float(a["pitch"]) - float(b["pitch"])) / ranges["pitch"]
    droll = angle_diff_deg(float(a["roll"]), float(b["roll"])) / ranges["roll"]

    return math.sqrt(
        W_RADIUS * dr * dr
        + W_YAW * dyaw * dyaw
        + W_PITCH * dpitch * dpitch
        + W_ROLL * droll * droll
    )


def compute_ranges(records: List[Dict[str, Any]]) -> Dict[str, float]:
    radii = [float(r["radius"]) for r in records]
    yaws = [float(r["yaw"]) for r in records]
    pitches = [float(r["pitch"]) for r in records]

    return {
        "radius": max(max(radii) - min(radii), 1e-6),
        "yaw": max(max(yaws) - min(yaws), 1e-6),
        "pitch": max(max(pitches) - min(pitches), 1e-6),
        # roll is circular; 180 deg is enough as normalization baseline.
        "roll": 180.0,
    }


def center_preference_score(record: Dict[str, Any]) -> float:
    """Choose a stable first point close to the original/front view."""
    return (
        1.0 * abs(float(record["yaw"]))
        + 1.0 * abs(float(record["pitch"]))
        + 0.25 * angle_diff_deg(float(record["roll"]), 0.0)
    )


def farthest_point_sampling(records: List[Dict[str, Any]], target_num: int) -> List[Dict[str, Any]]:
    if len(records) <= target_num:
        return list(records)

    ranges = compute_ranges(records)
    remaining = list(records)

    # Start from a central and usually stable view.
    first = min(remaining, key=center_preference_score)
    selected = [first]
    remaining.remove(first)

    while remaining and len(selected) < target_num:
        best_rec = None
        best_score = -1.0

        for rec in remaining:
            # The candidate's diversity score is its distance to the nearest selected point.
            min_dist_to_selected = min(param_distance(rec, s, ranges) for s in selected)

            # Tie-breaker: prefer roll closer to 0 deg.
            tie_break = -1e-4 * roll_abs_score(rec)
            score = min_dist_to_selected + tie_break

            if score > best_score:
                best_score = score
                best_rec = rec

        selected.append(best_rec)
        remaining.remove(best_rec)

    return selected


def validate_records(records: List[Dict[str, Any]]) -> None:
    required = {"idx", "radius", "yaw", "pitch", "roll", "endpose"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} is missing keys: {sorted(missing)}")
        if not isinstance(rec["endpose"], list) or len(rec["endpose"]) != 6:
            raise ValueError(f"Record {i} has invalid endpose. Expected 6 numbers.")


def main() -> None:
    input_path = INPUT_PATH

    records = load_records(input_path)
    validate_records(records)

    deduped = deduplicate_by_radius_yaw_pitch(records)
    selected = farthest_point_sampling(deduped, TARGET_NUM)

    # Keep all original keys and values. Do not reassign idx.
    save_records(OUTPUT_PATH, selected)

    print(f"Input file: {input_path}")
    print(f"Original reachable poses: {len(records)}")
    print(f"After radius/yaw/pitch deduplication: {len(deduped)}")
    print(f"Saved selected poses: {len(selected)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
