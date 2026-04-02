"""
evaluate_primary_secondary.py

Logika:
- yolov8m30.pt vertina tik primary_damage
- yolov8m120.pt vertina tik secondary_damage
- kiekvienam modeliui galima nustatyti skirtingus parametrus
- išvestis: folder, damage_primary_severity, damage_secondary_severity

Pavyzdys:
python evaluate_primary_secondary.py \
  --cars_root ../cars_test \
  --meta_csv ../cars_test_meta.csv \
  --best_weights yolov8m30.pt \
  --yolo_weights yolov8m120.pt \
  --out_csv primary_secondary_results.csv \
  --best_imgsz 640 \
  --best_conf 0.50 \
  --best_iou_tile 0.45 \
  --best_iou_merge 0.25 \
  --best_overlap 0.10 \
  --yolo_imgsz 1024 \
  --yolo_conf 0.20 \
  --yolo_iou_tile 0.60 \
  --yolo_iou_merge 0.40 \
  --yolo_overlap 0.20
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from torchvision.ops import nms

from cv.severity_estimation import score_view


# -----------------------------
# 1) Damage normalization / visual filter
# -----------------------------
NON_VISUAL = {
    "BIOHAZARD/CHEMICAL",
    "DAMAGE HISTORY",
    "ELECTRICAL",
    "ENGINE DAMAGE",
    "FRAME DAMAGE",
    "MECHANICAL",
    "MISSING/ALTERED VIN",
    "NORMAL WEAR",
    "REPOSSESSION",
    "SUSPENSION",
    "THEFT",
    "TRANSMISSION DAMAGE",
    "UNKNOWN",
    "WATER/FLOOD",
}


def normalize_damage_label(dmg: Optional[str]) -> str:
    if dmg is None or pd.isna(dmg):
        return "NONE"

    d = str(dmg).strip().upper()
    if d in {"", "NONE", "N/A", "NA", "NULL", "NAN"}:
        return "NONE"
    return d


def is_visual_damage(dmg: Optional[str]) -> bool:
    d = normalize_damage_label(dmg)
    if d == "NONE":
        return False
    return d not in NON_VISUAL


def visual_or_none(dmg: Optional[str]) -> Optional[str]:
    return dmg if is_visual_damage(dmg) else None


def damage_to_key(dmg: Optional[str]) -> str:
    d = normalize_damage_label(dmg)
    if d == "NONE":
        return "NONE"

    if d == "FRONT END":
        return "FRONT_END"
    if d == "REAR END":
        return "REAR_END"
    if d == "SIDE":
        return "SIDE"
    if d == "FRONT & REAR":
        return "FRONT_REAR"
    if d == "ALL OVER":
        return "ALL_OVER"
    if d == "TOP/ROOF":
        return "TOP_ROOF"
    if d == "UNDERCARRIAGE":
        return "UNDERCARRIAGE"
    if d == "MINOR DENT/SCRATCHES":
        return "MINOR"

    if d in {"HAIL", "STORM DAMAGE", "ROLLOVER", "STRIP", "VANDALISM", "BURN", "PARTIAL REPAIR"}:
        return "ALL_OVER"

    if d in {"BURN - ENGINE", "BURN - INTERIOR"}:
        return "ALL_OVER"

    return "OTHER"


# -----------------------------
# 2) View mapping
# -----------------------------
COPART_6 = {
    1: "front_left",
    2: "rear_left",
    3: "rear_right",
    4: "front_right",
    5: "front_center",
    6: "rear_center",
}

IAAI_4 = {
    1: "front_right",
    2: "front_left",
    3: "rear_left",
    4: "rear_right",
}


def get_view_map(n_imgs: int) -> Dict[int, str]:
    if n_imgs == 6:
        return COPART_6
    if n_imgs == 4:
        return IAAI_4
    raise ValueError(f"Unsupported number of images: {n_imgs} (expected 4 or 6)")


# -----------------------------
# 3) Region -> indices
# -----------------------------
def select_indices_for_damage_key(damage_key: str, n_imgs: int) -> List[int]:
    if n_imgs == 6:
        if damage_key == "FRONT_END":
            return [1, 4, 5]
        if damage_key == "REAR_END":
            return [2, 3, 6]
        if damage_key == "SIDE":
            return [1, 2, 3, 4]
        if damage_key in {"FRONT_REAR", "ALL_OVER", "MINOR", "OTHER", "TOP_ROOF", "UNDERCARRIAGE"}:
            return [1, 2, 3, 4, 5, 6]
        return [1, 2, 3, 4, 5, 6]

    if n_imgs == 4:
        if damage_key == "FRONT_END":
            return [1, 2]
        if damage_key == "REAR_END":
            return [3, 4]
        if damage_key == "SIDE":
            return [1, 2, 3, 4]
        if damage_key in {"FRONT_REAR", "ALL_OVER", "MINOR", "OTHER", "TOP_ROOF", "UNDERCARRIAGE"}:
            return [1, 2, 3, 4]
        return [1, 2, 3, 4]

    return list(range(1, n_imgs + 1))


def select_primary_secondary_indices(
    primary_key: str,
    secondary_key: str,
    n_imgs: int,
) -> Tuple[List[int], List[int]]:
    p_idx = select_indices_for_damage_key(primary_key, n_imgs) if primary_key != "NONE" else []
    s_idx = select_indices_for_damage_key(secondary_key, n_imgs) if secondary_key != "NONE" else []
    return p_idx, s_idx


def max_severity_for_indices(view_sev: Dict[int, Optional[float]], indices: List[int]) -> Optional[float]:
    vals = [view_sev.get(i) for i in indices]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# -----------------------------
# 4) Tiled YOLO predict
# -----------------------------
def tile_windows(W, H, grid=2, overlap=0.2):
    assert 0 <= overlap < 0.5
    tile_w = int(np.ceil(W / grid))
    tile_h = int(np.ceil(H / grid))

    stride_w = int(tile_w * (1 - overlap))
    stride_h = int(tile_h * (1 - overlap))
    stride_w = max(1, stride_w)
    stride_h = max(1, stride_h)

    tiles = []
    ys = list(range(0, max(1, H - tile_h + 1), stride_h))
    xs = list(range(0, max(1, W - tile_w + 1), stride_w))

    if len(xs) == 0 or xs[-1] != W - tile_w:
        xs.append(max(0, W - tile_w))
    if len(ys) == 0 or ys[-1] != H - tile_h:
        ys.append(max(0, H - tile_h))

    for y in ys:
        for x in xs:
            x1, y1 = x, y
            x2, y2 = min(W, x + tile_w), min(H, y + tile_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


def class_aware_nms(boxes_xyxy, scores, class_ids, iou_thres=0.6):
    keep_all = []
    unique_classes = torch.unique(class_ids)
    for c in unique_classes:
        idx = torch.where(class_ids == c)[0]
        if idx.numel() == 0:
            continue
        keep = nms(boxes_xyxy[idx], scores[idx], iou_thres)
        keep_all.append(idx[keep])
    if not keep_all:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)
    return torch.cat(keep_all, dim=0)


def yolo_tiled_predict(
    model: YOLO,
    image_bgr: np.ndarray,
    imgsz=1024,
    conf=0.2,
    iou_tile=0.6,
    iou_merge=0.4,
    grid=2,
    overlap=0.2,
    device=None,
):
    H, W = image_bgr.shape[:2]
    tiles = tile_windows(W, H, grid=grid, overlap=overlap)

    all_boxes = []
    all_scores = []
    all_classes = []

    with torch.inference_mode():
        for (x1, y1, x2, y2) in tiles:
            tile = image_bgr[y1:y2, x1:x2]

            results = model.predict(
                source=tile,
                imgsz=imgsz,
                conf=conf,
                iou=iou_tile,
                device=device,
                verbose=False,
            )

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            b = r.boxes.xyxy
            s = r.boxes.conf
            c = r.boxes.cls.to(torch.int64)

            if device is not None and b.device.type == "cpu":
                b = b.to("cuda")
                s = s.to("cuda")
                c = c.to("cuda")

            b = b.clone()
            b[:, [0, 2]] += float(x1)
            b[:, [1, 3]] += float(y1)

            all_boxes.append(b)
            all_scores.append(s)
            all_classes.append(c)

        if not all_boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes_t = torch.cat(all_boxes, dim=0)
        scores_t = torch.cat(all_scores, dim=0)
        classes_t = torch.cat(all_classes, dim=0)

        keep = class_aware_nms(boxes_t, scores_t, classes_t, iou_thres=iou_merge)

        boxes = boxes_t[keep].detach().cpu().numpy().astype(np.float32)
        scores = scores_t[keep].detach().cpu().numpy().astype(np.float32)
        classes = classes_t[keep].detach().cpu().numpy().astype(np.int32)

        return boxes, scores, classes


def to_detection_dicts(boxes, scores, classes, names_dict=None):
    dets = []
    for b, sc, cl in zip(boxes, scores, classes):
        cls_name = names_dict[int(cl)] if names_dict is not None else str(int(cl))
        dets.append({
            "class": cls_name,
            "conf": float(sc),
            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
        })
    return dets


# -----------------------------
# 5) IO helpers
# -----------------------------
def natural_sort_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_images(folder: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png")
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs, key=lambda p: natural_sort_key(p.name))


def infer_n_imgs(image_paths: List[Path]) -> int:
    n = len(image_paths)
    if n in (4, 6):
        return n
    if n > 6:
        return 6
    return n


def folder_id_int(folder: str) -> int:
    s = folder.split("_", 1)[0]
    try:
        return int(s)
    except ValueError:
        return 10**18


# -----------------------------
# 6) Score views for one car / one model
# -----------------------------
def score_views_for_car(
    model: YOLO,
    car_dir: Path,
    device=None,
    imgsz=1024,
    conf=0.2,
    iou_tile=0.6,
    iou_merge=0.4,
    overlap=0.2,
) -> Tuple[Dict[int, Optional[float]], int]:
    image_paths = list_images(car_dir)
    if not image_paths:
        return {}, 0

    n_in_folder = len(image_paths)
    n_imgs = infer_n_imgs(image_paths)

    if n_imgs not in (4, 6):
        return {}, n_in_folder

    image_paths = image_paths[:n_imgs]

    names = model.names if hasattr(model, "names") else None
    names_dict = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)} if names else None

    view_sev: Dict[int, Optional[float]] = {}
    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            view_sev[idx] = None
            continue

        H, W = img.shape[:2]
        boxes, scores, classes = yolo_tiled_predict(
            model=model,
            image_bgr=img,
            imgsz=imgsz,
            conf=conf,
            iou_tile=iou_tile,
            iou_merge=iou_merge,
            overlap=overlap,
            device=device,
        )
        dets = to_detection_dicts(boxes, scores, classes, names_dict=names_dict)

        vr = score_view(dets, img_w=W, img_h=H)
        view_sev[idx] = vr.get("severity", None)

    return view_sev, n_imgs


# -----------------------------
# 7) Only primary from best / only secondary from yolo
# -----------------------------
def compute_primary_only(
    view_sev: Dict[int, Optional[float]],
    n_imgs: int,
    primary_damage: Optional[str],
    secondary_damage: Optional[str],
) -> Optional[float]:
    p_key = damage_to_key(primary_damage)
    s_key = damage_to_key(secondary_damage)
    p_idx, _ = select_primary_secondary_indices(p_key, s_key, n_imgs)
    return max_severity_for_indices(view_sev, p_idx) if p_idx else None


def compute_secondary_only(
    view_sev: Dict[int, Optional[float]],
    n_imgs: int,
    primary_damage: Optional[str],
    secondary_damage: Optional[str],
) -> Optional[float]:
    p_key = damage_to_key(primary_damage)
    s_key = damage_to_key(secondary_damage)
    _, s_idx = select_primary_secondary_indices(p_key, s_key, n_imgs)
    return max_severity_for_indices(view_sev, s_idx) if s_idx else None


# -----------------------------
# 8) Main
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cars_root", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--yolov8m30_weights", type=str, default="yolov8m30.pt")
    parser.add_argument("--yolov8m120_weights", type=str, default="yolov8m120.pt")

    # yolov8m30.pt params -> primary
    parser.add_argument("--yolov8m30_imgsz", type=int, default=640)
    parser.add_argument("--yolov8m30_conf", type=float, default=0.50)
    parser.add_argument("--yolov8m30_iou_tile", type=float, default=0.45)
    parser.add_argument("--yolov8m30_iou_merge", type=float, default=0.25)
    parser.add_argument("--yolov8m30_overlap", type=float, default=0.10)

    # yolov8m120.pt params -> secondary
    parser.add_argument("--yolov8m120_imgsz", type=int, default=1024)
    parser.add_argument("--yolov8m120_conf", type=float, default=0.25)
    parser.add_argument("--yolov8m120_iou_tile", type=float, default=0.60)
    parser.add_argument("--yolov8m120_iou_merge", type=float, default=0.40)
    parser.add_argument("--yolov8m120_overlap", type=float, default=0.20)

    return parser.parse_args()


def main():
    args = parse_args()

    cars_root = args.cars_root
    meta_csv = args.meta_csv
    out_csv = args.out_csv

    meta = pd.read_csv(meta_csv)

    if {"damage_primary", "damage_secondary"}.issubset(meta.columns):
        primary_col = "damage_primary"
        secondary_col = "damage_secondary"
    elif {"primary_damage", "secondary_damage"}.issubset(meta.columns):
        primary_col = "primary_damage"
        secondary_col = "secondary_damage"
    else:
        raise ValueError(
            "meta_csv must contain either "
            "['damage_primary', 'damage_secondary'] "
            "or ['primary_damage', 'secondary_damage']"
        )

    if "id" not in meta.columns:
        raise ValueError("meta_csv must contain column 'id'")

    device = 0 if torch.cuda.is_available() else None

    model_yolov8m30 = YOLO(args.yolov8m30_weights)      # primary
    model_yolov8m120 = YOLO(args.yolov8m120_weights)    # secondary

    meta_by_id = meta.copy()
    meta_by_id["id"] = meta_by_id["id"].astype(str)
    meta_by_id = meta_by_id.set_index("id")

    rows = []

    car_folders = [f for f in os.listdir(cars_root) if (cars_root / f).is_dir()]
    car_folders = sorted(car_folders, key=folder_id_int)

    for folder in car_folders:
        car_dir = cars_root / folder
        folder_id = folder.split("_", 1)[0]

        if folder_id not in meta_by_id.index:
            rows.append({
                "folder": folder,
                "damage_primary_severity": None,
                "damage_secondary_severity": None,
            })
            continue

        row = meta_by_id.loc[folder_id]

        primary_damage_raw = row.get(primary_col, None)
        secondary_damage_raw = row.get(secondary_col, None)

        primary_damage = visual_or_none(primary_damage_raw)
        secondary_damage = visual_or_none(secondary_damage_raw)

        image_paths = list_images(car_dir)
        n_imgs = infer_n_imgs(image_paths)

        if not car_dir.exists() or not car_dir.is_dir() or n_imgs not in (4, 6):
            rows.append({
                "folder": folder,
                "damage_primary_severity": None,
                "damage_secondary_severity": None,
            })
            continue

        primary_sev = None
        if primary_damage is not None:
            view_yolov8m30, _ = score_views_for_car(
                model=model_yolov8m30,
                car_dir=car_dir,
                device=device,
                imgsz=args.yolov8m30_imgsz,
                conf=args.yolov8m30_conf,
                iou_tile=args.yolov8m30_iou_tile,
                iou_merge=args.yolov8m30_iou_merge,
                overlap=args.yolov8m30_overlap,
            )
            primary_sev = compute_primary_only(
                view_sev=view_yolov8m30,
                n_imgs=n_imgs,
                primary_damage=primary_damage,
                secondary_damage=secondary_damage,
            )

        secondary_sev = None
        if secondary_damage is not None:
            view_yolov8m120, _ = score_views_for_car(
                model=model_yolov8m120,
                car_dir=car_dir,
                device=device,
                imgsz=args.yolov8m120_imgsz,
                conf=args.yolov8m120_conf,
                iou_tile=args.yolov8m120_iou_tile,
                iou_merge=args.yolov8m120_iou_merge,
                overlap=args.yolov8m120_overlap,
            )
            secondary_sev = compute_secondary_only(
                view_sev=view_yolov8m120,
                n_imgs=n_imgs,
                primary_damage=primary_damage,
                secondary_damage=secondary_damage,
            )

        rows.append({
            "folder": folder,
            "damage_primary_severity": primary_sev,
            "damage_secondary_severity": secondary_sev,
        })

        print(
            f"{folder}: "
            f"damage_primary_severity={primary_sev}, "
            f"damage_secondary_severity={secondary_sev}"
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "folder",
            "damage_primary_severity",
            "damage_secondary_severity",
        ],
    )
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
