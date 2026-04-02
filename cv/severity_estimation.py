import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# -----------------------------
# Config
# -----------------------------

TYPE_WEIGHTS = {
    "scratch": 0.4,
    "rub": 0.5,
    "dent": 1.0,
    "crack": 1.1,
    "lamp broken": 1.2,
    "tire flat": 1.2,
    "dislocated part": 1.3,
    "no part": 1.4,
    "glass shatter": 1.4,
    "crash": 1.5,
}

CLASS_MIN_CONF = {
    "crash": 0.60,
    "no part": 0.60,
    "glass shatter": 0.50,
    "dislocated part": 0.45,
    "lamp broken": 0.45,
    "tire flat": 0.45,
}

CRIT_CLASSES = {"no part", "crash", "glass shatter", "dislocated part", "tire flat", "lamp broken"}

SUPPORT_CLASSES_WHEN_ANCHOR = {"lamp broken", "dislocated part"}

@dataclass
class ScoringConfig:
    # base filtering
    base_conf: float = 0.30
    # proxy floor (adaptive; min value used when weak evidence)
    proxy_floor_weak: float = 0.04
    proxy_floor_strong: float = 0.10

    # extent clipping
    extent_clip: float = 0.95

    # how much "type" matters
    tw_mean_mix: float = 0.60   # mean weight share
    tw_max_mix: float = 0.40    # max weight share

    # count impact (smaller than before to reduce overboost)
    count_alpha: float = 0.02
    count_cap: int = 8

    # soft saturation: raw -> raw/(raw+k)
    sat_k: float = 0.35

    # confidence-to-boost mapping
    boost_min_conf: float = 0.40
    boost_full_conf: float = 0.75

    # boosts (scaled by confidence)
    boost_crash: float = 0.18
    boost_no_part: float = 0.22
    boost_glass: float = 0.12
    boost_tire_combo: float = 0.18

    # optional “combo minimum” without hard 0.70
    combo_no_part_lamp_min_raw: float = 0.55  # ~3.2
    combo_no_part_lamp_requires_extent: float = 0.12

    # severity output
    half_steps: bool = True  # set False when you compute MAE/RMSE!
    severity_min: float = 1.0
    severity_max: float = 5.0

    # aggregation
    use_max_when_anchor: bool = True
    no_anchor_quantile: float = 0.75  # instead of pure max
    # if view max is much higher than rest and not anchored, damp it
    outlier_gap: float = 1.5
    outlier_penalty: float = 0.5


# -----------------------------
# Geometry
# -----------------------------

def compute_union_area_raster(boxes: List[Tuple[float, float, float, float]]) -> float:
    """
    Union area (px) via rasterization inside local bounding rect.
    Keeps your original idea, but with extra guards.
    """
    if not boxes:
        return 0.0

    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]

    min_x, max_x = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    min_y, max_y = int(np.floor(min(ys))), int(np.ceil(max(ys)))

    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return 0.0

    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        # normalize inverted boxes
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1i = int(np.floor(x1 - min_x)); x2i = int(np.ceil(x2 - min_x))
        y1i = int(np.floor(y1 - min_y)); y2i = int(np.ceil(y2 - min_y))

        x1i = max(0, min(w, x1i)); x2i = max(0, min(w, x2i))
        y1i = max(0, min(h, y1i)); y2i = max(0, min(h, y2i))

        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = 1

    return float(mask.sum())


def proxy_area_from_boxes(boxes: List[Tuple[float, float, float, float]]) -> float:
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    w = (max(xs) - min(xs))
    h = (max(ys) - min(ys))
    if w <= 0 or h <= 0:
        return 0.0
    return float(w * h)


# -----------------------------
# Helpers
# -----------------------------

def conf_scale(conf: float, c0: float, c1: float) -> float:
    """Map conf to [0..1] between c0..c1."""
    if c1 <= c0:
        return 1.0 if conf >= c0 else 0.0
    return float(np.clip((conf - c0) / (c1 - c0), 0.0, 1.0))


def soft_saturate(raw: float, k: float) -> float:
    """raw -> raw/(raw+k) to avoid hard clipping bias on highs."""
    raw = max(0.0, raw)
    return float(raw / (raw + k)) if (raw + k) > 0 else 0.0


def round_severity(sev: float, half_steps: bool) -> float:
    if half_steps:
        return float(round(sev * 2.0) / 2.0)
    return float(round(sev, 3))


# -----------------------------
# View scoring (rewritten)
# -----------------------------

def score_view(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    cfg: ScoringConfig = ScoringConfig(),
) -> Dict[str, Any]:
    """
    Key changes vs your v1:
    - adaptive proxy floor (smaller for weak evidence -> less low-severity inflation)
    - confidence-scaled boosts instead of fixed constants
    - no hard "raw >= 0.70" rule; replaced with softer minimum
    - soft saturation instead of hard clip-driven compression
    - keep None when no valid detections
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError("img_w/img_h must be positive.")

    if not detections:
        return {
            "severity": None,
            "has_damage": False,
            "anchor": False,
            "crit_count": 0,
            "max_conf": 0.0,
            "extent": 0.0,
            "raw": 0.0,
            "classes": [],
            "n_boxes": 0,
        }

    # 1) class-specific filtering
    filtered = []
    for d in detections:
        c = d.get("class")
        sc = float(d.get("conf", 0.0))
        min_c = CLASS_MIN_CONF.get(c, cfg.base_conf)
        if sc >= min_c:
            filtered.append(d)

    # 2) anchor detection on ORIGINAL detections (not only filtered)
    has_strong_anchor = (
        any(d.get("class") == "no part" and float(d.get("conf", 0.0)) >= 0.70 for d in detections) or
        any(d.get("class") == "crash" and float(d.get("conf", 0.0)) >= 0.60 for d in detections)
    )

    # 3) If anchored, allow support classes at lower conf
    if has_strong_anchor:
        for d in detections:
            c = d.get("class")
            sc = float(d.get("conf", 0.0))
            if c in SUPPORT_CLASSES_WHEN_ANCHOR and sc >= 0.25 and d not in filtered:
                filtered.append(d)

    # 4) fallback: base threshold
    if not filtered:
        filtered = [d for d in detections if float(d.get("conf", 0.0)) >= cfg.base_conf]

    if not filtered:
        return {
            "severity": None,
            "has_damage": False,
            "anchor": False,
            "crit_count": 0,
            "max_conf": 0.0,
            "extent": 0.0,
            "raw": 0.0,
            "classes": [],
            "n_boxes": 0,
        }

    boxes = [tuple(d["bbox"]) for d in filtered]
    classes = [d["class"] for d in filtered]
    confs = [float(d["conf"]) for d in filtered]
    max_conf = float(max(confs)) if confs else 0.0

    image_area = float(img_w * img_h)

    # 5) union + proxy
    union_area = compute_union_area_raster(boxes)
    proxy_area = proxy_area_from_boxes(boxes)
    if proxy_area <= 0.0 or union_area <= 0.0:
        return {
            "severity": None,
            "has_damage": False,
            "anchor": has_strong_anchor,
            "crit_count": 0,
            "max_conf": max_conf,
            "extent": 0.0,
            "raw": 0.0,
            "classes": classes,
            "n_boxes": len(boxes),
        }

    # adaptive proxy floor: less inflation when weak evidence
    proxy_floor_frac = cfg.proxy_floor_strong if (has_strong_anchor or max_conf >= 0.60) else cfg.proxy_floor_weak
    proxy_area = max(proxy_area, proxy_floor_frac * image_area)

    extent = float(np.clip(union_area / proxy_area, 0.0, cfg.extent_clip))

    # 6) type weight mix(mean,max)
    tw = np.array([TYPE_WEIGHTS.get(c, 1.0) for c in classes], dtype=np.float32)
    mean_tw = float(cfg.tw_mean_mix * tw.mean() + cfg.tw_max_mix * tw.max())

    # 7) count factor (weaker)
    n_boxes = len(boxes)
    count_factor = 1.0 + cfg.count_alpha * min(n_boxes, cfg.count_cap)

    raw = extent * mean_tw * count_factor

    # 8) confidence-scaled boosts
    s = conf_scale(max_conf, cfg.boost_min_conf, cfg.boost_full_conf)

    if "crash" in classes:
        raw += cfg.boost_crash * s

    if "no part" in classes:
        # only boost if extent isn't tiny OR anchor is present
        if extent > 0.10 or has_strong_anchor:
            raw += cfg.boost_no_part * s

    if "glass shatter" in classes:
        raw += cfg.boost_glass * s

    if ("tire flat" in classes) and any(c in classes for c in ["dent", "dislocated part", "crash"]):
        raw += cfg.boost_tire_combo * s

    # softer version of: no part + lamp broken
    if ("no part" in classes) and ("lamp broken" in classes) and (extent >= cfg.combo_no_part_lamp_requires_extent):
        raw = max(raw, cfg.combo_no_part_lamp_min_raw)

    # 9) soft saturation instead of hard clip-to-1 that compresses highs
    raw = soft_saturate(raw, cfg.sat_k)

    # 10) map to severity
    sev = cfg.severity_min + (cfg.severity_max - cfg.severity_min) * float(np.clip(raw, 0.0, 1.0))
    sev = round_severity(sev, cfg.half_steps)
    sev = float(np.clip(sev, cfg.severity_min, cfg.severity_max))

    crit_count = sum(1 for (c, cf) in zip(classes, confs) if (c in CRIT_CLASSES and cf >= 0.40))

    return {
        "severity": sev,
        "has_damage": True,
        "anchor": bool(has_strong_anchor),
        "crit_count": int(crit_count),
        "max_conf": max_conf,
        "extent": extent,
        "raw": float(raw),
        "classes": classes,
        "n_boxes": int(n_boxes),
    }


# -----------------------------
# Aggregation across views (rewritten)
# -----------------------------

def aggregate_vehicle_views_v2(
    view_results: List[Dict[str, Any]],
    cfg: ScoringConfig = ScoringConfig(),
) -> Dict[str, Any]:
    """
    Key changes vs your v1:
    - if there's anchor on the top view -> allow max (auction-style)
    - if no anchor -> use quantile (default 0.75) instead of max (reduces inflation)
    - keep your outlier damp, but applied on quantile mode too
    """
    scores = [vr.get("severity") for vr in view_results if vr.get("severity") is not None]
    if not scores:
        return {"vehicle_severity": 1.0, "used_scores": [], "coverage": 0}

    scores = [float(s) for s in scores]
    scores_sorted = sorted(scores)
    mx = max(scores_sorted)
    mn = min(scores_sorted)
    gap = mx - mn

    # anchor on max view?
    max_views = [vr for vr in view_results if vr.get("severity") == mx]
    has_anchor_on_max = any(bool(vr.get("anchor", False)) for vr in max_views)

    if cfg.use_max_when_anchor and has_anchor_on_max:
        vehicle_sev = mx
    else:
        vehicle_sev = float(np.quantile(scores_sorted, cfg.no_anchor_quantile))

    # outlier damp (no-anchor especially)
    if (gap >= cfg.outlier_gap) and (not has_anchor_on_max):
        vehicle_sev = max(vehicle_sev - cfg.outlier_penalty, float(np.median(scores_sorted)))

    vehicle_sev = round_severity(vehicle_sev, cfg.half_steps)
    vehicle_sev = float(np.clip(vehicle_sev, cfg.severity_min, cfg.severity_max))

    return {"vehicle_severity": vehicle_sev, "used_scores": scores_sorted, "coverage": len(scores_sorted)}