from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrackState:
    track_id: int
    cls: int
    last_time: float
    # history: list of (t, location(x,y,z), yaw, velocity(vx,vy,vz), score)
    history: List[Tuple[float, np.ndarray, float, np.ndarray, float]] = field(default_factory=list)
    # radar_stats parallel list (dict per frame or None): {'count':int,'vx_mean':..,'vy_mean':..,'vr_mean':..,'rcs_mean':.., ...}
    radar_stats: List[dict | None] = field(default_factory=list)
    active: bool = True
    age: int = 0
    hits: int = 0
    misses: int = 0

    def add(self, t: float, location: np.ndarray, yaw: float, velocity: np.ndarray, score: float, radar: dict | None = None):
        self.history.append((t, location.astype(np.float32), float(yaw), velocity.astype(np.float32), float(score)))
        self.radar_stats.append(radar)
        self.last_time = t
        self.hits += 1
        self.misses = 0
        self.age += 1

    def last(self):
        return self.history[-1] if self.history else None


class SimpleTracker:
    """
    Minimal 3D distance + class-consistent tracker for per-camera streaming.
    - Associate detections to existing tracks by nearest 3D center (x,z mainly) within threshold.
    - No motion model; simple life-time with miss tolerance.
    """

    def __init__(
        self,
        dist_thresh: float = 2.5,
        max_misses: int = 5,
        id_start: int = 1,
    ):
        self.dist_thresh = float(dist_thresh)
        self.max_misses = int(max_misses)
        self.next_id = int(id_start)
        self.tracks: Dict[int, TrackState] = {}

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # Use horizontal plane distance (x,z). Inputs are (x,y,z)
        return float(np.linalg.norm([a[0] - b[0], a[2] - b[2]]))

    def update(self, detections: List[dict], t: float) -> Dict[int, dict]:
        """
        Args:
            detections: list of dict with keys: class, score, location(np.array(3)), yaw(float), velocity(np.array(3))
            t: current timestamp (seconds)

        Returns:
            mapping: track_id -> detection dict (augmented with track_id)
        """
        mapping: Dict[int, dict] = {}

        # Prepare candidates: per class existing tracks last positions
        by_class: Dict[int, List[Tuple[int, TrackState]]] = {}
        for tid, tr in self.tracks.items():
            if not tr.active:
                continue
            by_class.setdefault(tr.cls, []).append((tid, tr))

        # Greedy matching by nearest-neighbor per detection
        used_tracks: set = set()
        for det in detections:
            cls = int(det.get("class", 0))
            loc = det.get("location")
            if hasattr(loc, "detach"):
                loc = loc.detach().cpu().numpy()
            loc = np.array(loc).reshape(3)

            yaw = float(det.get("yaw", 0.0))
            vel = det.get("velocity", np.zeros(3, np.float32))
            if hasattr(vel, "detach"):
                vel = vel.detach().cpu().numpy()
            vel = np.array(vel).reshape(-1)
            if vel.size < 3:
                vel = np.pad(vel, (0, 3 - vel.size), constant_values=0.0)
            vel = vel[:3]

            score = float(det.get("score", 0.0))

            # find best track of same class
            best_tid = -1
            best_dist = 1e9
            if cls in by_class:
                for tid, tr in by_class[cls]:
                    if tid in used_tracks:
                        continue
                    last = tr.last()
                    if last is None:
                        continue
                    dist = self._distance(last[1], loc)
                    if dist < best_dist and dist <= self.dist_thresh:
                        best_dist = dist
                        best_tid = tid

            radar_feat = det.get('radar_feat')  # optional radar stats precomputed
            if best_tid < 0:
                # create new track
                tid = self.next_id
                self.next_id += 1
                tr = TrackState(track_id=tid, cls=cls, last_time=t)
                tr.add(t, loc, yaw, vel, score, radar=radar_feat)
                self.tracks[tid] = tr
                mapping[tid] = {**det, "track_id": tid}
                used_tracks.add(tid)
            else:
                # update existing track
                tr = self.tracks[best_tid]
                tr.add(t, loc, yaw, vel, score, radar=radar_feat)
                mapping[best_tid] = {**det, "track_id": best_tid}
                used_tracks.add(best_tid)

        # Aging and deactivate missing tracks
        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks:
                tr.misses += 1
                tr.age += 1
                if tr.misses > self.max_misses:
                    tr.active = False

        return mapping

    def get_active_tracks(self) -> Dict[int, TrackState]:
        return {tid: tr for tid, tr in self.tracks.items() if tr.active}

    def dump_tracks(self) -> Dict[int, dict]:
        # For serialization
        out: Dict[int, dict] = {}
        for tid, tr in self.tracks.items():
            out[tid] = {
                "track_id": tid,
                "class": tr.cls,
                "history": [
                    {
                        "t": float(t),
                        "location": loc.tolist(),
                        "yaw": float(yaw),
                        "velocity": vel.tolist(),
                        "score": float(score),
                        "radar": tr.radar_stats[i] if i < len(tr.radar_stats) else None,
                    }
                    for i, (t, loc, yaw, vel, score) in enumerate(tr.history)
                ],
                "active": bool(tr.active),
                "age": tr.age,
                "hits": tr.hits,
                "misses": tr.misses,
            }
        return out
