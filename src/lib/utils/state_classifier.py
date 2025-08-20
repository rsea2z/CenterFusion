from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def compute_yaw_rate(history: List[Tuple[float, np.ndarray, float, np.ndarray, float]]) -> float:
    """Estimate yaw rate (rad/s) from last two observations."""
    if len(history) < 2:
        return 0.0
    t2, _, yaw2, _, _ = history[-1]
    t1, _, yaw1, _, _ = history[-2]
    dt = max(1e-6, t2 - t1)
    # wrap to [-pi,pi]
    dyaw = float(yaw2 - yaw1)
    while dyaw > np.pi:
        dyaw -= 2 * np.pi
    while dyaw < -np.pi:
        dyaw += 2 * np.pi
    return dyaw / dt


class RuleBasedStateClassifier:
    """
    Simple rule-based vehicle state classifier with short-time smoothing.
    States: going_straight | lane_change | turning | parking
    """

    def __init__(
        self,
        speed_thresh_move: float = 1.0,   # m/s
        lat_speed_thresh: float = 0.6,    # m/s
        yaw_rate_turn: float = np.deg2rad(10.0),  # rad/s
        yaw_rate_straight: float = np.deg2rad(5.0),
        stop_speed: float = 0.2,
        min_duration_s: float = 0.5,
    ):
        self.speed_thresh_move = speed_thresh_move
        self.lat_speed_thresh = lat_speed_thresh
        self.yaw_rate_turn = yaw_rate_turn
        self.yaw_rate_straight = yaw_rate_straight
        self.stop_speed = stop_speed
        self.min_duration_s = min_duration_s

    def classify_one(self, history: List[Tuple[float, np.ndarray, float, np.ndarray, float]]) -> str:
        if not history:
            return "parking"
        t2, _, yaw2, vel2, _ = history[-1]
        # velocity in camera coords: vel2 = [vx, vy, vz]; forward≈z, lateral≈x
        v_forward = float(vel2[2])
        v_lateral = float(vel2[0])
        speed = float(np.linalg.norm([v_forward, v_lateral]))
        yaw_rate = compute_yaw_rate(history)

        # parking/stop
        if speed < self.stop_speed:
            return "parking"

        # turning
        if abs(yaw_rate) >= self.yaw_rate_turn:
            return "turning"

        # lane change (lateral motion but heading change small)
        if abs(v_lateral) >= self.lat_speed_thresh and abs(yaw_rate) < self.yaw_rate_straight:
            return "lane_change"

        # default moving straight
        if speed >= self.speed_thresh_move and abs(yaw_rate) < self.yaw_rate_straight:
            return "going_straight"

        # fallback
        return "going_straight"

    def batch_classify(self, track_histories: Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray, float]]]) -> Dict[int, str]:
        return {tid: self.classify_one(hist) for tid, hist in track_histories.items()}
