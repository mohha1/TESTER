import os
from pathlib import Path
import json

import numpy as np
import cv2
import pandas as pd

# YAML (info.yml)
import yaml

# OpenVINO
try:
    from openvino.runtime import Core
except Exception:
    from openvino import Core


# ============================================================
# 1) EDIT PATHS / SETTINGS HERE
# ============================================================

# --- Model package paths (DASH export) ---
MODEL_XML = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\openvino\y00x00\model.xml"
MODEL_BIN = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\openvino\y00x00\model.bin"  # optional; OpenVINO usually finds it automatically
INFO_YML  = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\info_id50885.yml"
META_JSON = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\openvino\y00x00\meta_data.json"            # ← IMPORTANT for PaDiM

# --- Inference IO ---
IMAGES_DIR = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\PROVA"
OUT_DIR    = r"C:\Users\PQMOHAMMED\Downloads\GD1X EXTRA SCRE_Main-Dataset_v1-3_PaDiM_v1-6-0_ID50885\RESULTAT"

# Choose threshold set from info.yml
MODE = "auto"   # "auto" or "manual"

# OpenVINO device
DEVICE = "CPU"

# Save artifacts (can be heavy for thousands of images)
SAVE_ARTIFACTS = True
SAVE_ONLY_NG_ARTIFACTS = False

# Blob settings
CONNECTIVITY = 8
MIN_BLOB_AREA = 0

# Preprocessing
COLOR_ORDER = "RGB"
SCALE_01 = True

# --- NEW SETTINGS -----------------------------------------------------------

# Normalization of raw anomaly map using meta_data.json
#   "auto"   -> normalize if model is PaDiM (detected from info.yml) or if
#               raw map values fall outside [0,1]
#   "always" -> always normalize using meta_data.json
#   "never"  -> never normalize (original PatchCore behavior)
NORMALIZE_MODE = "never"                                                # ← NEW

# Threshold comparison operator
#   ">="  -> match the original script (cand8 >= binary_thr)
#   ">"   -> strict greater-than
THRESHOLD_OP = ">="                                                    # ← NEW

# Score metric used for the final OK/NG decision
#   "max_blob_area"  -> area of the largest blob (original behavior)
#   "sum_blob_area"  -> total area of all blobs
#   "max_pixel"      -> maximum value in cand8
SCORE_METRIC = "max_blob_area"                                         # ← NEW

# ============================================================
# 2) INTERNAL HELPERS
# ============================================================

def imread_unicode_safe(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    except Exception:
        return None


def load_info_yml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_meta_json(path: str) -> dict | None:                         # ← NEW
    """
    Load meta_data.json exported alongside the OpenVINO model.
    Expected keys: image_threshold, pixel_threshold, min, max
    Returns None if file does not exist or cannot be parsed.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_anomaly_map(raw_map: np.ndarray, meta: dict) -> np.ndarray:   # ← NEW
    """
    Normalize raw anomaly map using min/max from meta_data.json.
    This is critical for PaDiM, whose raw output is NOT in [0,1].

    Formula (same as anomalib / DASH internal):
        normalized = (raw - min) / (max - min)
        clipped to [0, 1]
    """
    vmin = float(meta["min"])
    vmax = float(meta["max"])

    if vmax - vmin < 1e-12:
        # Degenerate case: constant map
        return np.zeros_like(raw_map, dtype=np.float32)

    norm = (raw_map - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def preprocess_to_model(img_bgr: np.ndarray, input_shape, input_dtype):
    n, c, h, w = list(input_shape)
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    if COLOR_ORDER.upper() == "RGB":
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb = resized

    x = rgb.astype(np.float32) if np.dtype(input_dtype).kind == "f" else rgb.astype(input_dtype)

    if np.dtype(input_dtype).kind == "f" and SCALE_01:
        x = x / 255.0

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)

    if c == 1 and x.shape[1] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if SCALE_01:
            gray /= 255.0
        x = gray[None, None, :, :]

    return x


def pick_anomaly_map_output(compiled_model, inference_result):         # ← NEW
    """
    Some DASH/anomalib exports have TWO outputs:
      - output 0: anomaly_map   (1,1,H,W)
      - output 1: anomaly_score (scalar or 1,)

    Others have only one output (the map).

    This function picks the spatial anomaly map by choosing the output
    with the highest number of dimensions (4D > 1D).
    """
    outputs = compiled_model.outputs

    if len(outputs) == 1:
        return np.array(inference_result[outputs[0]])

    # Multiple outputs: pick the one that looks like a spatial map
    best_output = None
    best_ndim = -1

    for out in outputs:
        arr = np.array(inference_result[out])
        if arr.ndim > best_ndim:
            best_ndim = arr.ndim
            best_output = arr

    return best_output


def squeeze_to_hw(output_array: np.ndarray) -> np.ndarray:
    m = np.array(output_array)
    if m.ndim == 4:
        m = m[0, 0]
    elif m.ndim == 3:
        m = m[0]
    elif m.ndim != 2:
        raise RuntimeError(f"Unexpected output shape: {m.shape}")
    return m.astype(np.float32)


def to_uint8_map_like_sample(raw_map: np.ndarray) -> np.ndarray:
    return np.clip(raw_map * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def apply_binary_threshold(cand8: np.ndarray, binary_thr: int) -> np.ndarray:   # ← NEW
    """Apply threshold with configurable operator."""
    if THRESHOLD_OP == ">":
        return (cand8 > binary_thr).astype(np.uint8)
    else:
        return (cand8 >= binary_thr).astype(np.uint8)


def remove_small_blobs(mask01: np.ndarray, min_area: int, connectivity: int):
    if min_area <= 0:
        return mask01
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask01.astype(np.uint8), connectivity=connectivity
    )
    cleaned = np.zeros_like(mask01, dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lbl] = 1
    return cleaned


def compute_blob_areas(mask01: np.ndarray, connectivity: int):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask01.astype(np.uint8), connectivity=connectivity
    )
    if num_labels <= 1:
        return 0, [], 0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
    return len(areas), areas, int(max(areas)) if areas else 0


def compute_anomaly_score(cand8, max_blob_area, sum_blob_area):        # ← NEW
    """Select score metric based on configuration."""
    if SCORE_METRIC == "max_blob_area":
        return max_blob_area
    elif SCORE_METRIC == "sum_blob_area":
        return sum_blob_area
    elif SCORE_METRIC == "max_pixel":
        return int(cand8.max())
    else:
        return max_blob_area


def overlay_outputs(img_bgr: np.ndarray, candidate8: np.ndarray, mask01: np.ndarray):
    h, w = candidate8.shape
    vis = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap(candidate8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(vis, 0.65, heat, 0.35, 0)
    contours, _ = cv2.findContours(
        (mask01 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


# ============================================================
# 3) MAIN
# ============================================================

def main():
    # Validate paths
    for p in [MODEL_XML, INFO_YML, IMAGES_DIR]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_ARTIFACTS:
        (out_dir / "candidate8").mkdir(exist_ok=True)
        (out_dir / "binary").mkdir(exist_ok=True)
        (out_dir / "overlay").mkdir(exist_ok=True)

    # Load info.yml
    info = load_info_yml(INFO_YML)
    ai_model = info.get("AiModel", "Unknown")                         # ← NEW

    if MODE.lower() == "auto":
        binary_thr = int(info["BinaryThresholdAuto"])
        judge_thr  = float(info["JudgeThresholdAuto"])
    else:
        binary_thr = int(info["BinaryThresholdManual"])
        judge_thr  = float(info["JudgeThresholdManual"])

    # Load meta_data.json                                              # ← NEW
    meta = load_meta_json(META_JSON)

    # Decide whether to normalize                                      # ← NEW
    is_padim = "padim" in ai_model.lower() if ai_model else False
    is_autoencoder = "autoencoder" in ai_model.lower() if ai_model else False

    if NORMALIZE_MODE == "always":
        do_normalize = True
    elif NORMALIZE_MODE == "never":
        do_normalize = False
    elif NORMALIZE_MODE == "auto":
        # Normalize for PaDiM and Autoencoder; skip for PatchCore
        do_normalize = is_padim or is_autoencoder
    else:
        do_normalize = False

    if do_normalize and meta is None:
        print("[WARN] Normalization requested but meta_data.json not found or unreadable.")
        print("       Raw map will be used directly (may cause incorrect results for PaDiM).")
        do_normalize = False

    print("====================================================")
    print(f"[INFO] AiModel={ai_model}  AiCategory={info.get('AiCategory')}  Ver={info.get('Ver')}")
    print(f"[INFO] Expected image size: {info.get('ImageWidth')}x{info.get('ImageHeight')} ch={info.get('ImageChannels')}")
    print(f"[THR ] MODE={MODE}  BinaryThreshold={binary_thr}  JudgeThreshold={judge_thr}")
    print(f"[BLOB] connectivity={CONNECTIVITY}  min_blob_area={MIN_BLOB_AREA}")
    print(f"[NORM] normalize={do_normalize}  mode={NORMALIZE_MODE}  meta_loaded={meta is not None}")  # ← NEW
    if do_normalize and meta:                                          # ← NEW
        print(f"       meta min={meta.get('min')}  max={meta.get('max')}  "
              f"image_thr={meta.get('image_threshold')}  pixel_thr={meta.get('pixel_threshold')}")
    print(f"[SCORE] metric={SCORE_METRIC}  threshold_op={THRESHOLD_OP}")  # ← NEW
    print("====================================================")

    # OpenVINO load & compile
    core = Core()
    model = core.read_model(MODEL_XML)
    compiled = core.compile_model(model, DEVICE)

    input_any = compiled.input(0)
    input_shape = list(input_any.shape)
    input_dtype = input_any.element_type.to_dtype()

    # Print ALL outputs (some models have 2)                           # ← CHANGED
    print(f"[OV] input:  name={input_any.any_name}  shape={input_shape}  dtype={input_dtype}")
    for i, out in enumerate(compiled.outputs):
        print(f"[OV] output[{i}]: name={out.any_name}  shape={list(out.shape)}")
    print("====================================================")

    # List images
    img_dir = Path(IMAGES_DIR)
    img_paths = sorted([
        p for p in img_dir.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ])
    if not img_paths:
        raise RuntimeError(f"No images found in: {IMAGES_DIR}")

    rows = []
    first_image = True                                                 # ← NEW (for diagnostics)

    for p in img_paths:
        img = imread_unicode_safe(str(p))
        if img is None:
            print(f"[WARN] cannot read {p}")
            continue

        # Preprocess
        x = preprocess_to_model(img, input_shape, input_dtype)

        # Inference
        inference_result = compiled([x])

        # Pick the spatial anomaly map output                          # ← CHANGED
        raw_output = pick_anomaly_map_output(compiled, inference_result)
        raw_map = squeeze_to_hw(raw_output)

        # --- Diagnostics on first image ---                           # ← NEW
        if first_image:
            print(f"[DIAG] First image: {p.name}")
            print(f"       raw_map shape={raw_map.shape}  "
                  f"min={raw_map.min():.6f}  max={raw_map.max():.6f}  "
                  f"mean={raw_map.mean():.6f}")
            if do_normalize:
                print(f"       -> Will normalize using meta min={meta['min']} max={meta['max']}")
            else:
                if raw_map.max() > 1.5:
                    print(f"       [WARN] raw_map max={raw_map.max():.4f} > 1.5 but normalization is OFF.")
                    print(f"              Consider setting NORMALIZE_MODE='always' or check meta_data.json.")
            first_image = False

        # --- Normalize if needed ---                                  # ← NEW
        if do_normalize:
            norm_map = normalize_anomaly_map(raw_map, meta)
        else:
            norm_map = raw_map

        # Convert to 8-bit anomaly map
        cand8 = to_uint8_map_like_sample(norm_map)

        # Binarize                                                     # ← CHANGED
        mask01 = apply_binary_threshold(cand8, binary_thr)

        # Optional small blob removal
        mask01 = remove_small_blobs(mask01, MIN_BLOB_AREA, CONNECTIVITY)

        # Blob processing
        blob_count, areas, max_blob_area = compute_blob_areas(mask01, CONNECTIVITY)
        sum_blob_area = int(sum(areas)) if areas else 0

        # Anomaly score                                                # ← CHANGED
        anomaly_score = compute_anomaly_score(cand8, max_blob_area, sum_blob_area)

        # Final decision
        if THRESHOLD_OP == ">":                                        # ← NEW
            product = "NG" if anomaly_score > judge_thr else "OK"
        else:
            product = "NG" if anomaly_score >= judge_thr else "OK"

        # Save artifacts
        if SAVE_ARTIFACTS and (not SAVE_ONLY_NG_ARTIFACTS or product == "NG"):
            cv2.imwrite(str(out_dir / "candidate8" / f"{p.stem}_candidate.png"), cand8)
            cv2.imwrite(str(out_dir / "binary" / f"{p.stem}_binary.png"), (mask01 * 255).astype(np.uint8))
            ov = overlay_outputs(img, cand8, mask01)
            cv2.imwrite(str(out_dir / "overlay" / f"{p.stem}_overlay.png"), ov)

        # Record
        rows.append({
            "file": str(p),
            "ai_model": ai_model,                                     # ← NEW
            "normalized": do_normalize,                                # ← NEW
            "mode": MODE,
            "binary_threshold_255": binary_thr,
            "judge_threshold": judge_thr,
            "threshold_op": THRESHOLD_OP,                              # ← NEW
            "score_metric": SCORE_METRIC,                              # ← NEW
            "connectivity": CONNECTIVITY,
            "min_blob_area_filter": MIN_BLOB_AREA,
            "anomaly_score": anomaly_score,                            # ← CHANGED
            "product": product,
            "blob_count": blob_count,
            "max_blob_area": max_blob_area,
            "sum_blob_area": sum_blob_area,
            "ng_pixels_total": int(mask01.sum()),
            "cand8_max": int(cand8.max()),
            "cand8_mean": float(cand8.mean()),
            "raw_map_max": float(raw_map.max()),                       # raw before normalization
            "raw_map_mean": float(raw_map.mean()),
            "norm_map_max": float(norm_map.max()),                     # ← NEW: after normalization
            "norm_map_mean": float(norm_map.mean()),                   # ← NEW
        })

        print(f"[{p.name}] {product} | score({SCORE_METRIC})={anomaly_score} thr={judge_thr} "
              f"| blobs={blob_count} ng_px={int(mask01.sum())} "
              f"| raw_max={raw_map.max():.4f} norm_max={norm_map.max():.4f}")

    # Export Excel
    df = pd.DataFrame(rows)
    xlsx_path = out_dir / "results_dash_anomaly.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"\n{'='*52}")
    print(f"Done. {len(rows)} images processed.")
    print(f"Excel saved: {xlsx_path}")
    if SAVE_ARTIFACTS:
        print(f"Artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()