# track.py
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import config

@dataclass
class Segment:
    kind: str
    length: float
    angle: float = 0.0
    radius: float = 0.0
    s_start: float = 0.0

    @property
    def s_end(self) -> float:
        return self.s_start + self.length

    @property
    def curvature(self) -> float:
        if self.kind == "straight" or self.radius == 0:
            return 0.0

        sign = 1.0 if self.angle > 0 else -1.0
        return sign / self.radius

# Track Layout

_TRACK_RAW = [
    ("straight", 100, 0),
    ("corner",   300, +30),
    ("straight", 100, 0),
    ("corner",   500, +15),
    ("straight", 200, 0),
    ("corner",   80,  +85),
    ("straight", 60,  0),
    ("corner",   35,  +160),
    ("straight", 120, 0),
    ("corner",   120, -30),
    ("straight", 730, 0),
    ("corner",   80,  -100),
    ("straight", 30,  0),
    ("corner",   90,  +100),
    ("straight", 80,  0),
    ("corner",   300, +25),
    ("straight", 600, 0),
    ("corner",   200, +70),
    ("straight", 180, 0),
    ("corner",   300, -30),
    ("straight", 100, 0),
    ("corner",   150, +55),
    ("straight", 80,  0),
    ("corner",   350, -25),
    ("straight", 770, 0),
    ("corner",   180, +80),
    ("straight", 100, 0),
    ("corner",   70,  -50),
    ("straight", 50,  0),
    ("corner",   100, +35),
    ("straight", 30,  0),
    ("corner",   200, -20),
    ("straight", 30,  0),
    ("corner",   180, -25),
    ("straight", 30,  0),
    ("corner",   500, -15),
    ("straight", 155.3, 0),
]

def _build_segments(raw = _TRACK_RAW) -> List[Segment]:
    segments: List[Segment] = []
    s = 0.0
    for entry in raw:
        kind = entry[0]
        if kind == "straight":
            length = entry[1]
            seg = Segment(kind="straight", length=length, s_start=s)
        else:
            radius = entry[1]
            angle_rad = math.radians(entry[2])
            arc_length = abs(angle_rad) * radius
            seg = Segment(
                kind ="corner",
                length =arc_length,
                angle = angle_rad,
                radius = radius,
                s_start = s,
            )
        segments.append(seg)
        s+=seg.length
    return segments

# Track Class

class Track:
    def __init__(self):
        self.segments: List[Segment] = _build_segments()
        self.length: float = sum(seg.length for seg in self.segments)
        self.width: float = config.TRACK_WIDTH
        self.gravel_width: float = config.GRAVEL_WIDTH
        self.severe_offtrack_distance: float = config.SEVERE_OFFTRACK_DISTANCE
        self._world_points: np.ndarray = self._compute_world_points()

    def segment_at(self, s: float):
        s = s % self.length
        for seg in self.segments:
            if seg.s_start <= s < seg.s_end:
                return seg
        return self.segments[-1]

    def curvature_at(self, s: float) -> float:
        return self.segment_at(s).curvature

    def lookahead_curvatures(self, s: float, current_speed: float, times: Optional[List[float]] = None,) -> List[float]:
        if times is None:
            times = config.RL_LOOKAHEAD_TIMES
        speed = max(current_speed, 1.0)
        return [self.curvature_at(s + t * speed) for t in times]

    def distance_to_edges(self, lateral: float) -> Tuple[float, float]:
        half = self.width / 2.0
        return (half + lateral, half - lateral)

    def is_off_track(self, lateral: float) -> bool:
        return abs(lateral) > self.width / 2.0

    def is_in_gravel(self, lateral: float) -> bool:
        half = self.width / 2.0
        return half < abs(lateral) <= half + self.gravel_width

    def is_severe_offtrack(self, lateral: float) -> bool:
        half = self.width / 2.0
        boundary = half + self.gravel_width + self.severe_offtrack_distance
        return abs(lateral) > boundary

    def max_safe_speed(self, s: float, grip: float) -> float:
        seg = self.segment_at(s)
        if seg.kind == "straight" or seg.radius == 0:
            return config.MAX_SPEED
        v_max = math.sqrt(
            grip * config.GRAVITY * seg.radius * config.CORNERING_G_FACTOR
        )

        return max(config.MIN_CORNER_SPEED, min(v_max, config.MAX_SPEED))

    def racing_line_lateral(self, s: float) -> float:

        s = s % self.length
        seg = self.segment_at(s)
        frac = (s - seg.s_start) / seg.length if seg.length > 0 else 0.0

        # Maximum lateral excursion: stay 1.5m from the edge for safety margin
        W = self.width / 2.0 - 1.5

        # ---- Corner: outside → inside (at apex) → outside via cosine ----
        if seg.kind == "corner":
            sign = 1.0 if seg.curvature > 0 else -1.0

            return -sign * W * math.cos(frac * 2.0 * math.pi)

        # ---- Straight: blend between previous-corner-exit and next-corner-entry ----
        idx = self.segments.index(seg)
        n = len(self.segments)

        # Find the previous corner (wrap around if needed)
        prev_outside = 0.0
        for i in range(1, n):
            prev_seg = self.segments[(idx - i) % n]
            if prev_seg.kind == "corner":
                prev_sign = 1.0 if prev_seg.curvature > 0 else -1.0
                prev_outside = -prev_sign * W
                break

        # Find the next corner
        next_outside = 0.0
        for i in range(1, n):
            next_seg = self.segments[(idx + i) % n]
            if next_seg.kind == "corner":
                next_sign = 1.0 if next_seg.curvature > 0 else -1.0
                next_outside = -next_sign * W
                break

        return prev_outside + (next_outside - prev_outside) * frac

    def _compute_world_points(self, n_points: int = 2000) -> np.ndarray:

        # First pass: compute (x, y, heading) at every segment boundary
        boundaries = []
        x, y, heading = 0.0, 0.0, 0.0
        for seg in self.segments:
            boundaries.append((x, y, heading))
            if seg.kind == "straight":
                x += math.cos(heading) * seg.length
                y += math.sin(heading) * seg.length
            else:
                sign = 1.0 if seg.angle > 0 else -1.0
                cx = x - sign * seg.radius * math.sin(heading)
                cy = y + sign * seg.radius * math.cos(heading)
                start_angle = math.atan2(y - cy, x - cx)
                end_angle = start_angle + sign * abs(seg.angle)
                x = cx + seg.radius * math.cos(end_angle)
                y = cy + seg.radius * math.sin(end_angle)
                heading += seg.angle

        # Second pass: sample uniformly along arc length
        points = []
        for i in range(n_points):
            s = i * self.length / n_points
            # Find segment containing s
            seg_idx = 0
            for j, seg in enumerate(self.segments):
                if seg.s_start <= s < seg.s_end:
                    seg_idx = j
                    break
            else:
                seg_idx = len(self.segments) - 1
            seg = self.segments[seg_idx]
            local_s = s - seg.s_start
            bx, by, bh = boundaries[seg_idx]

            if seg.kind == "straight":
                px = bx + math.cos(bh) * local_s
                py = by + math.sin(bh) * local_s
                ph = bh
            else:
                sign = 1.0 if seg.angle > 0 else -1.0
                cx = bx - sign * seg.radius * math.sin(bh)
                cy = by + sign * seg.radius * math.cos(bh)
                start_angle = math.atan2(by - cy, bx - cx)
                t = local_s / seg.length
                a = start_angle + sign * abs(seg.angle) * t
                px = cx + seg.radius * math.cos(a)
                py = cy + seg.radius * math.sin(a)
                ph = bh + sign * abs(seg.angle) * t
            points.append((px, py, ph))
        return np.array(points)

    @property
    def world_points(self) -> np.ndarray:
        return self._world_points

    def position_at(self, s:float, lateral: float = 0.0) -> Tuple[float, float, float]:
        s = s % self.length
        idx = int(s / self.length * len(self._world_points)) % len(self._world_points)
        x,y,heading = self.world_points[idx]
        wx = x + lateral * math.sin(heading)
        wy = y - lateral * math.cos(heading)
        return wx, wy, heading

# Sanity Check

if __name__ == "__main__":
    t = Track()
    print(f"Total track length: {t.length:.1f} units (target ~5891)")
    print(f"Number of segments: {len(t.segments)}")
    print(f"World points sampled: {len(t.world_points)} (uniform arc-length)")
    net = math.degrees(sum(s.angle for s in t.segments))
    print(f"Net turn angle: {net:.1f}° (target 360°)")
    print()
    for i, seg in enumerate(t.segments):
        if seg.kind == "straight":
            print(f"  S{i:02d}: straight  len={seg.length:6.1f}")
        else:
            print(f"  S{i:02d}: corner    len={seg.length:6.1f}  "
                  f"r={seg.radius:5.1f}  angle={math.degrees(seg.angle):+6.1f}°")
    print()
    print(f"  curvature at s=600:  {t.curvature_at(600):+.4f}")