import argparse
import ast
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from ultralytics import YOLO
from torchvision.ops import nms

from cv.severity_estimation import score_view


# =========================================================
# 1) Damage normalization
# =========================================================

NON_VISUAL = {
    "BIOHAZARD",
    "DAMAGE HISTORY",
    "ELECTRICAL",
    "ENGINE DAMAGE",
    "FRAME DAMAGE",
    "MECHANICAL",
    "MISSING/ALTERED VIN",
    "NORMAL WEAR & TEAR",
    "CASH FOR CLUNKERS",
    "REPOSSESSION",
    "SUSPENSION",
    "THEFT",
    "MINOR",
    "TRANSMISSION DAMAGE",
    "UNKNOWN",
    "WATER/FLOOD",
    "REPLACED VIN",
    "UNDERCARRIAGE",
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
    if d in {"HAIL", "STORM DAMAGE", "ROLLOVER", "STRIPPED", "VANDALISM", "BURN", "PARTIAL REPAIR"}:
        return "ALL_OVER"
    if d in {"BURN - ENGINE", "BURN - INTERIOR"}:
        return "ALL_OVER"
    return "OTHER"


# =========================================================
# 2) View mapping / region indices
# =========================================================

def select_indices_for_damage_key(damage_key: str, n_imgs: int) -> List[int]:
    if n_imgs == 6:
        if damage_key == "FRONT_END":
            return [1, 4, 5]
        if damage_key == "REAR_END":
            return [2, 3, 6]
        if damage_key == "SIDE":
            return [1, 2, 3, 4]
        return [1, 2, 3, 4, 5, 6]

    if n_imgs == 4:
        if damage_key == "FRONT_END":
            return [1, 2]
        if damage_key == "REAR_END":
            return [3, 4]
        if damage_key == "SIDE":
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


# =========================================================
# 3) Thread-local state
# =========================================================

_THREAD_LOCAL = threading.local()


def get_thread_session() -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://autohelperbot.com/",
        })
        _THREAD_LOCAL.session = sess
    return sess


def get_thread_models(
    yolov8m30_weights: str,
    yolov8m120_weights: str,
) -> Tuple[YOLO, YOLO]:
    cache_key = f"{yolov8m30_weights}||{yolov8m120_weights}"
    models_cache = getattr(_THREAD_LOCAL, "models_cache", None)
    if models_cache is None:
        models_cache = {}
        _THREAD_LOCAL.models_cache = models_cache

    if cache_key not in models_cache:
        models_cache[cache_key] = (
            YOLO(yolov8m30_weights),
            YOLO(yolov8m120_weights),
        )

    return models_cache[cache_key]


# =========================================================
# 4) URL / dataset helpers
# =========================================================

def normalize_url(url: str) -> str:
    if url is None:
        return ""
    return str(url).strip()


def load_image_from_url(url: str, timeout: int = 20, retries: int = 3) -> np.ndarray:
    session = get_thread_session()
    last_err = None

    for _ in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            return img
        except Exception as e:
            last_err = e

    raise RuntimeError(f"failed to load image: {url} | error={last_err}")


def parse_photo_urls_json(value) -> List[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [normalize_url(x) for x in value if normalize_url(x)]

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [normalize_url(x) for x in parsed if normalize_url(x)]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [normalize_url(x) for x in parsed if normalize_url(x)]
    except Exception:
        pass

    return []


def expected_n_imgs_from_auction(auction_value: Optional[str]) -> Optional[int]:
    if auction_value is None or pd.isna(auction_value):
        return None
    a = str(auction_value).strip().lower()
    if a == "copart":
        return 6
    if a == "iaai":
        return 4
    return None


def get_ordered_image_urls(row: pd.Series) -> Tuple[List[str], int]:
    all_urls = parse_photo_urls_json(row.get("photo_urls_json", None))
    n_expected = expected_n_imgs_from_auction(row.get("auction", None))

    if n_expected is None:
        raise ValueError(f"Unsupported auction: {row.get('auction', None)}")

    if len(all_urls) < n_expected:
        return all_urls, len(all_urls)

    return all_urls[:n_expected], n_expected


# =========================================================
# 5) Tiled YOLO predict
# =========================================================

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


def load_images_parallel(
    image_urls: List[str],
    max_workers: int,
) -> Tuple[Dict[int, Optional[np.ndarray]], List[Dict[str, str]]]:
    images: Dict[int, Optional[np.ndarray]] = {}
    errors: List[Dict[str, str]] = []

    workers = max(1, min(max_workers, len(image_urls))) if image_urls else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_meta = {
            ex.submit(load_image_from_url, url): (idx, url)
            for idx, url in enumerate(image_urls, start=1)
        }

        for fut in as_completed(future_to_meta):
            idx, url = future_to_meta[fut]
            try:
                images[idx] = fut.result()
            except Exception as e:
                images[idx] = None
                errors.append({
                    "stage": f"download_view_{idx}",
                    "error": str(e),
                    "url": url,
                })

    return images, errors


def score_views_for_images(
    model: YOLO,
    images_by_idx: Dict[int, Optional[np.ndarray]],
    n_imgs: int,
    device=None,
    imgsz=1024,
    conf=0.2,
    iou_tile=0.6,
    iou_merge=0.4,
    overlap=0.2,
) -> Dict[int, Optional[float]]:
    names = model.names if hasattr(model, "names") else None
    names_dict = names if isinstance(names, dict) else ({i: n for i, n in enumerate(names)} if names else None)

    view_sev: Dict[int, Optional[float]] = {}

    for idx in range(1, n_imgs + 1):
        img = images_by_idx.get(idx)
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

    return view_sev


# =========================================================
# 6) Primary / secondary severity
# =========================================================

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


# =========================================================
# 7) Resume / save helpers
# =========================================================

def dedupe_keep_last(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if df.empty or key_col not in df.columns:
        return df
    return df.drop_duplicates(subset=[key_col], keep="last")


def load_existing_checkpoint(checkpoint_csv: Path) -> pd.DataFrame:
    if checkpoint_csv.exists():
        try:
            df = pd.read_csv(checkpoint_csv)
            if "vin" in df.columns:
                df["vin"] = df["vin"].astype(str)
            return dedupe_keep_last(df, "vin")
        except Exception as e:
            print(f"[WARN] failed to read checkpoint: {checkpoint_csv} | error={e}")
    return pd.DataFrame(columns=["vin", "auction", "damage_primary_severity", "damage_secondary_severity"])


def load_existing_errors(errors_csv: Path) -> pd.DataFrame:
    if errors_csv.exists():
        try:
            return pd.read_csv(errors_csv)
        except Exception as e:
            print(f"[WARN] failed to read errors file: {errors_csv} | error={e}")
    return pd.DataFrame(columns=["vin", "stage", "error", "url"])


def save_tables(rows: List[Dict[str, Any]], errors: List[Dict[str, Any]], checkpoint_csv: Path, errors_csv: Path):
    pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)
    pd.DataFrame(errors).to_csv(errors_csv, index=False)


# =========================================================
# 8) One-car processor
# =========================================================

def process_one_car(
    row_dict: Dict[str, Any],
    args,
    device,
    yolov8m30_weights: str,
    yolov8m120_weights: str,
    primary_col: str,
    secondary_col: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    row = pd.Series(row_dict)

    vin = str(row.get("vin", "")).strip()
    auction = str(row.get("auction", "")).strip().lower()
    errors: List[Dict[str, Any]] = []

    primary_damage_raw = row.get(primary_col, None)
    secondary_damage_raw = row.get(secondary_col, None)

    primary_damage = visual_or_none(primary_damage_raw)
    secondary_damage = visual_or_none(secondary_damage_raw)

    try:
        image_urls, n_imgs = get_ordered_image_urls(row)
    except Exception as e:
        result = {
            "vin": vin,
            "auction": auction,
            "damage_primary_severity": None,
            "damage_secondary_severity": None,
        }
        errors.append({"vin": vin, "stage": "parse_urls", "error": str(e), "url": ""})
        return result, errors, vin

    if n_imgs not in (4, 6):
        msg = f"expected 4 or 6 images, got {n_imgs}"
        result = {
            "vin": vin,
            "auction": auction,
            "damage_primary_severity": None,
            "damage_secondary_severity": None,
        }
        errors.append({"vin": vin, "stage": "image_count", "error": msg, "url": ""})
        return result, errors, vin

    images_by_idx, dl_errors = load_images_parallel(image_urls[:n_imgs], max_workers=args.download_workers)
    for e in dl_errors:
        errors.append({"vin": vin, **e})

    primary_sev = None
    secondary_sev = None

    try:
        model_yolov8m30, model_yolov8m120 = get_thread_models(
            yolov8m30_weights,
            yolov8m120_weights,
        )

        if primary_damage is not None:
            view_yolov8m30 = score_views_for_images(
                model=model_yolov8m30,
                images_by_idx=images_by_idx,
                n_imgs=n_imgs,
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

        if secondary_damage is not None:
            view_yolov8m120 = score_views_for_images(
                model=model_yolov8m120,
                images_by_idx=images_by_idx,
                n_imgs=n_imgs,
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

    except Exception as e:
        errors.append({"vin": vin, "stage": "inference", "error": str(e), "url": ""})

    result = {
        "vin": vin,
        "auction": auction,
        "damage_primary_severity": primary_sev,
        "damage_secondary_severity": secondary_sev,
    }
    return result, errors, vin


# =========================================================
# 9) Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--yolov8m30_weights", type=str, required=True)
    parser.add_argument("--yolov8m120_weights", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)

    parser.add_argument("--yolov8m30_imgsz", type=int, default=640)
    parser.add_argument("--yolov8m30_conf", type=float, default=0.5)
    parser.add_argument("--yolov8m30_iou_tile", type=float, default=0.45)
    parser.add_argument("--yolov8m30_iou_merge", type=float, default=0.25)
    parser.add_argument("--yolov8m30_overlap", type=float, default=0.1)

    parser.add_argument("--yolov8m120_imgsz", type=int, default=1024)
    parser.add_argument("--yolov8m120_conf", type=float, default=0.5)
    parser.add_argument("--yolov8m120_iou_tile", type=float, default=0.45)
    parser.add_argument("--yolov8m120_iou_merge", type=float, default=0.25)
    parser.add_argument("--yolov8m120_overlap", type=float, default=0.1)

    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--car_workers", type=int, default=2)

    return parser.parse_args()


# =========================================================
# 10) Main
# =========================================================

def main():
    args = parse_args()

    meta_csv = Path(args.meta_csv)
    yolov8m30_weights = Path(args.yolov8m30_weights)
    yolov8m120_weights = Path(args.yolov8m120_weights)
    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_csv = out_csv.with_name(out_csv.stem + "_checkpoint.csv")
    errors_csv = out_csv.with_name(out_csv.stem + "_errors.csv")

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta_csv not found: {meta_csv}")
    if not yolov8m30_weights.exists():
        raise FileNotFoundError(f"yolov8m30_weights not found: {yolov8m30_weights}")
    if not yolov8m120_weights.exists():
        raise FileNotFoundError(f"yolov8m120_weights not found: {yolov8m120_weights}")

    meta = pd.read_csv(meta_csv)

    required_cols = {"vin", "auction", "photo_urls_json"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"meta_csv missing required columns: {sorted(missing)}")

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

    meta["vin"] = meta["vin"].astype(str).str.strip()

    if args.limit > 0:
        meta = meta.head(args.limit).copy()

    device = 0 if torch.cuda.is_available() else "cpu"
    print("device =", device)
    print("download_workers =", args.download_workers)
    print("car_workers =", args.car_workers)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    processed_vins = set()

    if args.resume:
        checkpoint_df = load_existing_checkpoint(checkpoint_csv)
        errors_df = load_existing_errors(errors_csv)

        if not checkpoint_df.empty:
            rows = checkpoint_df.to_dict(orient="records")
            processed_vins = set(checkpoint_df["vin"].astype(str).tolist())

        if not errors_df.empty:
            errors = errors_df.to_dict(orient="records")

        print(f"[resume] loaded processed vins: {len(processed_vins)}")

    todo_df = meta[~meta["vin"].isin(processed_vins)].copy()
    todo_records = todo_df.to_dict(orient="records")
    total_all = len(meta)
    total_todo = len(todo_records)

    if total_todo == 0:
        print("Nothing to process.")
        final_df = pd.DataFrame(rows, columns=[
            "vin",
            "auction",
            "damage_primary_severity",
            "damage_secondary_severity",
        ])
        final_df = dedupe_keep_last(final_df, "vin")
        final_df.to_csv(out_csv, index=False)
        pd.DataFrame(errors).to_csv(errors_csv, index=False)
        final_df.to_csv(checkpoint_csv, index=False)
        print(f"Saved final      : {out_csv}")
        print(f"Saved checkpoint : {checkpoint_csv}")
        print(f"Saved errors     : {errors_csv}")
        return

    workers = max(1, args.car_workers)
    completed_new = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                process_one_car,
                rec,
                args,
                device,
                str(yolov8m30_weights),
                str(yolov8m120_weights),
                primary_col,
                secondary_col,
            )
            for rec in todo_records
        ]

        for fut in as_completed(futures):
            try:
                result, car_errors, vin = fut.result()
            except Exception as e:
                vin = ""
                result = {
                    "vin": "",
                    "auction": "",
                    "damage_primary_severity": None,
                    "damage_secondary_severity": None,
                }
                car_errors = [{"vin": vin, "stage": "future", "error": str(e), "url": ""}]

            rows.append(result)
            errors.extend(car_errors)

            if vin:
                processed_vins.add(vin)

            completed_new += 1
            print(
                f"{len(processed_vins)}/{total_all} total | "
                f"{completed_new}/{total_todo} new | "
                f"{result.get('vin', '')} ({result.get('auction', '')}) | "
                f"primary={result.get('damage_primary_severity')} | "
                f"secondary={result.get('damage_secondary_severity')}"
            )

            if args.save_every > 0 and (completed_new % args.save_every == 0):
                temp_df = pd.DataFrame(rows, columns=[
                    "vin",
                    "auction",
                    "damage_primary_severity",
                    "damage_secondary_severity",
                ])
                temp_df = dedupe_keep_last(temp_df, "vin")
                temp_rows = temp_df.to_dict(orient="records")
                save_tables(temp_rows, errors, checkpoint_csv, errors_csv)
                print(f"[checkpoint] saved {len(temp_rows)} rows -> {checkpoint_csv}")

    final_df = pd.DataFrame(rows, columns=[
        "vin",
        "auction",
        "damage_primary_severity",
        "damage_secondary_severity",
    ])
    final_df = dedupe_keep_last(final_df, "vin")
    final_df.to_csv(out_csv, index=False)

    pd.DataFrame(errors).to_csv(errors_csv, index=False)
    final_df.to_csv(checkpoint_csv, index=False)

    print(f"\nSaved final      : {out_csv}")
    print(f"Saved checkpoint : {checkpoint_csv}")
    print(f"Saved errors     : {errors_csv}")


if __name__ == "__main__":
    main()