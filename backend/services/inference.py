"""
Inference Service
Loads the YOLO11m-P2P3-CBAM champion model and provides a complete
inference pipeline that reuses the existing project modules:
  - model_service.py  → CBAM architecture + weight loading
  - inference_pipeline.py → multi-pass tiled + CLAHE inference
  - detection_processing.py → physical merging + refinement
  - feature_extraction.py → severity scoring
  - visualization.py → annotated image rendering
"""

import sys
import os
import types
import uuid
import logging
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("roadhealth.inference")

# ── Ensure the project root is on sys.path so existing modules resolve ──
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Import CBAM architecture pieces (must happen before torch.load) ──
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

from backend.config import BASELINE_MODEL_PATH, CBAM_MODEL_PATH, RESULTS_DIR, UPLOADS_DIR


# ═══════════════════════════════════════════════════════════════════
# CBAM Architecture Definitions (from model_service.py)
# Must be registered in both tasks.DetectionModel and __main__
# so that torch.load / pickle can resolve them.
# ═══════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Channel Attention Module: identifies WHAT feature channels to emphasize."""
    def __init__(self, channels: int = 128, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        avg = self.mlp(self.avg_pool(x).view(B, C))
        mx = self.mlp(self.max_pool(x).view(B, C))
        return x * torch.sigmoid(avg + mx).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module: identifies WHERE in the image defects are located."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: Channel + Spatial Attention."""
    def __init__(self, channels: int = 128, reduction: int = 4):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class P2P3CBAMDetail(nn.Module):
    """
    Detail branch: extracts high-res features from P2,
    filters noise via CBAM, and fuses residually into P3.
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 256,
                 hidden: int = 128, reduction: int = 4):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.cbam = CBAM(hidden, reduction)
        self.restore = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        x = self.downsample(p2)
        x = self.dw_conv(x)
        x = self.cbam(x)
        return self.restore(x)


def _p2p3_cbam_predict(self, x, profile=False, visualize=False,
                       augment=False, embed=None):
    """Exact P2→P3 CBAM forward pass (captures P2 at layer 2, fuses at layer 16)."""
    if augment:
        raise NotImplementedError(
            "Augmented prediction is not supported for P2P3-CBAM model."
        )
    outputs = {}
    current = x
    p2 = None

    for i, layer in enumerate(self.model):
        if layer.f == -1:
            inp = current
        elif isinstance(layer.f, int):
            inp = outputs[layer.f]
        else:
            inp = [current if j == -1 else outputs[j] for j in layer.f]

        current = layer(inp)
        outputs[i] = current

        if i == 2:
            p2 = current
        if i == 16:
            if p2 is None:
                raise RuntimeError("P2 feature map was not captured at layer 2.")
            detail = self.p2p3_cbam_detail(p2)
            current = current + detail
            outputs[i] = current

    return current


# ── Register on DetectionModel + __main__ for pickle compatibility ──
tasks.DetectionModel._p2p3_cbam_predict = _p2p3_cbam_predict
tasks.DetectionModel.P2P3CBAMDetail = P2P3CBAMDetail

_main_mod = sys.modules.get("__main__")
if _main_mod is not None:
    for name, obj in [("P2P3CBAMDetail", P2P3CBAMDetail), ("CBAM", CBAM),
                      ("ChannelAttention", ChannelAttention),
                      ("SpatialAttention", SpatialAttention),
                      ("_p2p3_cbam_predict", _p2p3_cbam_predict)]:
        setattr(_main_mod, name, obj)


# ═══════════════════════════════════════════════════════════════════
# Model Singleton
# ═══════════════════════════════════════════════════════════════════

_model = None


def load_model() -> YOLO:
    """
    Reconstruct and load the fine-tuned YOLO11m-P2P3-CBAM champion model.
    Replicates the logic from model_service.load_model() without Streamlit.
    """
    global _model
    if _model is not None:
        return _model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model on device: {device}")

    # Ensure __main__ module has classes registered
    _main = sys.modules.get("__main__")
    if _main is not None:
        for name, obj in [("P2P3CBAMDetail", P2P3CBAMDetail), ("CBAM", CBAM),
                          ("ChannelAttention", ChannelAttention),
                          ("SpatialAttention", SpatialAttention),
                          ("_p2p3_cbam_predict", _p2p3_cbam_predict)]:
            setattr(_main, name, obj)

    # 1. Load baseline YOLO11m inner model (architecture shell)
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_MODEL_PATH}\n"
            "Ensure models/best.pt is present in the project root."
        )
    baseline = YOLO(str(BASELINE_MODEL_PATH))
    net = baseline.model.to(device)
    net.eval()

    # 2. Attach P2→P3 CBAM detail module
    net.p2p3_cbam_detail = P2P3CBAMDetail().to(device)

    # 3. Bind custom forward pass
    bound_predict = types.MethodType(_p2p3_cbam_predict, net)
    net.p2p3_cbam_predict = bound_predict
    net.predict = bound_predict

    # 4. Load learned weights from CBAM checkpoint
    if not CBAM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CBAM checkpoint not found: {CBAM_MODEL_PATH}\n"
            "Ensure models/cbam_best.pt is present in the project root."
        )
    checkpoint = torch.load(str(CBAM_MODEL_PATH), map_location="cpu",
                            weights_only=False)
    model_obj = checkpoint.get("ema") or checkpoint.get("model")
    if model_obj is not None and hasattr(model_obj, "state_dict"):
        raw_state = model_obj.state_dict()
    elif isinstance(model_obj, dict):
        raw_state = model_obj
    else:
        raw_state = checkpoint.get("state_dict", {})

    learned_state = {
        key: value.detach().cpu().clone()
        for key, value in raw_state.items()
        if torch.is_tensor(value)
        and "total_ops" not in key
        and "total_params" not in key
    }

    missing, unexpected = net.load_state_dict(learned_state, strict=True,
                                              assign=True)
    assert len(missing) == 0, f"Missing weights: {missing}"
    assert len(unexpected) == 0, f"Unexpected weights: {unexpected}"

    net = net.to(device)
    net.eval()

    # 5. Wrap back into Ultralytics YOLO interface
    model = YOLO(str(BASELINE_MODEL_PATH))
    model.model = net

    total_params = sum(p.numel() for p in net.parameters())
    logger.info(
        f"YOLO11m-P2P3-CBAM loaded successfully "
        f"({total_params:,} parameters) on {device}"
    )

    _model = model
    return _model


def get_class_names() -> dict:
    """Return the model's class name mapping."""
    model = load_model()
    names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


# ═══════════════════════════════════════════════════════════════════
# Full Inference Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_full_pipeline(image_bgr: np.ndarray) -> dict:
    """
    Run the complete YOLO11m-P2P3-CBAM analysis pipeline on a BGR image.

    Uses the existing project modules:
      inference_pipeline.run_inference → detection_processing.process_detections
      → feature_extraction.extract_features → zone_analysis.compute_zone_damage
      → road_health.compute_road_health → visualization.draw_detections

    Returns a dict with all detection data, enriched features, PHS, and
    the path to the saved annotated image.
    """
    from inference_pipeline import run_inference, check_image_quality
    from detection_processing import process_detections
    from feature_extraction import extract_features
    from zone_analysis import compute_zone_damage
    from road_health import compute_road_health
    from visualization import draw_detections
    from backend.services.groq_validator import validate_road_image_quality

    # 0. Image-Level Validation & Quality Gate
    quality_gate = validate_road_image_quality(image_bgr)
    logger.info(f"Image Quality Gate: is_valid={quality_gate['is_valid']}, reason='{quality_gate['reason']}'")

    # If the image is invalid (non-road, severe dark, severe glare, severe blur), bypass YOLO
    if not quality_gate.get("is_valid", True):
        logger.warning(f"Bypassing YOLO defect detection: {quality_gate.get('user_message')}")
        all_raw = []
        nms_detections = []
        final_detections = []
        features = []
        H, W = image_bgr.shape[:2]
        
        health_summary = {
            "phs": None,
            "condition": "Non-Road / Unusable",
            "pothole_count": 0,
            "crack_count": 0,
            "total_defects": 0,
            "quality_warnings": [quality_gate.get("user_message", "Image quality is unsuitable for road analysis.")],
            "validation_status": "rejected",
            "validation_reason": quality_gate.get("reason", ""),
        }
    else:
        model = load_model()

        # 1. Multi-pass inference (full + tiled) on verified road image
        logger.info("Running YOLO11m-P2P3-CBAM inference pipeline...")
        try:
            all_raw, nms_detections = run_inference(model, image_bgr)
            logger.info(f"Raw detections: {len(all_raw)}, after NMS: {len(nms_detections)}")
        except Exception as e:
            logger.error(f"[DIAG ERROR] run_inference pipeline failed: {e}\n{traceback.format_exc()}")
            raise

        # 2. Physical merging & crack refinement
        logger.info(f"[DIAG 4/10] Starting detection processing (merging/refinement) on {len(nms_detections)} detections...")
        try:
            final_detections = process_detections(nms_detections, image_bgr)
            logger.info(f"[DIAG 5/10] Detection processing completed: {len(final_detections)} final detections.")
        except Exception as e:
            logger.error(f"[DIAG ERROR] process_detections failed: {e}\n{traceback.format_exc()}")
            raise

        # 3. Feature extraction (severity scoring)
        try:
            H, W = image_bgr.shape[:2]
            features = extract_features(final_detections, image_bgr)

            # 4. Zone damage analysis
            zone_damage, critical_zone_id, critical_zone_score, critical_zone_risk = \
                compute_zone_damage(features, W, H)

            # 5. PHS computation (uses existing road_health.py)
            health_summary = compute_road_health(features, zone_damage, [])
        except Exception as e:
            logger.error(f"[DIAG ERROR] Feature extraction / PHS computation failed: {e}\n{traceback.format_exc()}")
            raise

    # 6. Draw annotated image
    logger.info(f"[DIAG 6/10] Generating annotated image for {len(features)} features...")
    try:
        annotated = draw_detections(image_bgr, features)
        logger.info(f"[DIAG 7/10] Annotated image generated successfully (shape={annotated.shape}).")
    except Exception as e:
        logger.error(f"[DIAG ERROR] draw_detections failed: {e}\n{traceback.format_exc()}")
        raise

    # 7. Save original and annotated results using StorageProvider abstraction
    logger.info("[DIAG 8/10] Saving original and annotated image files via StorageProvider...")
    try:
        from backend.services.storage import get_storage_provider
        storage = get_storage_provider()

        uid = uuid.uuid4().hex[:12]
        original_filename = f"upload_{uid}.jpg"
        result_filename = f"result_{uid}.jpg"

        original_url = storage.save_cv2_image(image_bgr, original_filename, folder="uploads", quality=95)
        annotated_url = storage.save_cv2_image(annotated, result_filename, folder="results", quality=92)
        logger.info(f"[DIAG 9/10] Files saved successfully: original={original_url}, annotated={annotated_url}")
    except Exception as e:
        logger.error(f"[DIAG ERROR] Storage save_cv2_image failed: {e}\n{traceback.format_exc()}")
        raise

    # 8. Build detection list for API response (ensure native Python types)
    detection_list = []
    for d in features:
        detection_list.append({
            "class_name": str(d["class_name"]),
            "confidence": round(float(d["confidence"]), 4),
            "severity_score": round(float(d.get("severity_score", 0.0)), 4),
            "box": [round(float(v), 1) for v in d["box"]],
            "bbox_area_ratio": round(float(d.get("bbox_area_ratio", 0.0)), 6),
            "merged_count": int(d.get("merged_count", 1)),
        })

    return {
        "detections": detection_list,
        "summary": health_summary,
        "original_image_filename": original_filename,
        "result_image_filename": result_filename,
        "original_image_url": original_url,
        "result_image_url": annotated_url,
        "raw_count": int(len(all_raw)),
        "nms_count": int(len(nms_detections)),
        "final_count": int(len(final_detections)),
        "image_width": int(W),
        "image_height": int(H),
    }

