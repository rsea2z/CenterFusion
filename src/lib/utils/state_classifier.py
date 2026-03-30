"""
Simple rule-based vehicle state classifier with short-time smoothing.
States: going_straight | going_left | going_right | parking
"""
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
    Simple rule-based vehicle state classifier.
    States: going_straight | going_left | going_right | parking
    """

    def __init__(
        self,
        speed_thresh_move: float = 1.0,   # m/s
        lat_speed_thresh: float = 0.6,    # m/s
        yaw_rate_straight: float = np.deg2rad(5.0),
        stop_speed: float = 0.2,
        min_duration_s: float = 0.5,
    ):
        self.speed_thresh_move = speed_thresh_move
        self.lat_speed_thresh = lat_speed_thresh
        self.yaw_rate_straight = yaw_rate_straight
        self.stop_speed = stop_speed
        self.min_duration_s = min_duration_s

    def classify_one(
        self,
        history: List[Tuple[float, np.ndarray, float, np.ndarray, float]],
        radar_stats: list | None = None,
    ) -> str:
        if not history:
            return "parking"
        # Use mean lateral/forward velocity from history for stability
        vels = [h[3] for h in history]
        vl_mean = float(np.mean([v[0] for v in vels]))
        vf_mean = float(np.mean([v[2] for v in vels]))
        speed = float(np.linalg.norm([vf_mean, vl_mean]))
        yaw_rate = compute_yaw_rate(history)

        # dyn_prop parking: if most radar points say stationary, classify as parking
        if radar_stats:
            stationary_ratios = [
                rs.get('dyn_stationary', 0.0)
                for rs in radar_stats[-5:]  # recent 5 frames
                if rs and rs.get('count', 0) > 0
            ]
            if stationary_ratios:
                avg_stationary = float(np.mean(stationary_ratios))
                if avg_stationary > 0.5:
                    return "parking"

        # parking/stop
        if speed < self.stop_speed:
            return "parking"

        # going_right: positive lateral speed (vehicle moving to its right in camera frame)
        if vl_mean > self.lat_speed_thresh and abs(yaw_rate) < self.yaw_rate_straight:
            return "going_right"

        # going_left: negative lateral speed (vehicle moving to its left in camera frame)
        if vl_mean < -self.lat_speed_thresh and abs(yaw_rate) < self.yaw_rate_straight:
            return "going_left"

        # default: going straight
        if speed >= self.speed_thresh_move:
            return "going_straight"

        # fallback
        return "going_straight"

    def batch_classify(self, track_histories: Dict[int, List[Tuple[float, np.ndarray, float, np.ndarray, float]]]) -> Dict[int, str]:
        return {tid: self.classify_one(hist) for tid, hist in track_histories.items()}
