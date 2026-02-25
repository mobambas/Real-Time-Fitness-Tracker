from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np


class ExerciseType(str, Enum):
    SQUATS = "squats"
    PUSH_UPS = "push-ups"
    SIT_UPS = "sit-ups"
    LUNGES = "lunges"
    BICEP_CURLS = "bicep curls"
    LAT_PULLDOWN = "lat pulldown"
    PULL_UPS = "pull-ups"


@dataclass
class Landmark:
    """Minimal landmark representation compatible with MediaPipe Pose output."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


PoseFrame = Dict[str, Landmark]


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Returns the angle ABC (with B as the vertex) in degrees.

    This is the same convention used across the exercise scripts (e.g. `squats.py`,
    `push-up.py`, `bicep_curls.py`), and is intentionally simple so it can be
    ported directly to Dart/Kotlin for the mobile app.
    """
    a_np = np.array(a)
    b_np = np.array(b)
    c_np = np.array(c)
    radians = np.arctan2(c_np[1] - b_np[1], c_np[0] - b_np[0]) - np.arctan2(
        a_np[1] - b_np[1], a_np[0] - b_np[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return float(angle if angle <= 180.0 else 360.0 - angle)


def angle_from_frame(frame: PoseFrame, a: str, b: str, c: str) -> float:
    """
    Convenience helper: compute angle between three named landmarks in a pose frame.

    Example:
        angle = angle_from_frame(frame, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")
    """
    la, lb, lc = frame[a], frame[b], frame[c]
    return calculate_angle((la.x, la.y), (lb.x, lb.y), (lc.x, lc.y))


