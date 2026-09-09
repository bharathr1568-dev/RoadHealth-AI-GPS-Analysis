"""
RoadHealth AI — YOLO11m-P2P3-CBAM Validation Script
Imports the champion model, attaches CBAM & P2->P3 detail modules,
and runs official validation against the RoadHealthDataset_Final3000 test set.
"""

import sys
import time
import types
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# ── Paths & Configuration ──────────────────────────────────────
BASELINE_MODEL_PATH = Path(r"D:\STUDIES\FINAL YEAR PROJECT\Models\RoadHealthDataset_Final3000_YOLO11m\weights\best.pt")
CBAM_MODEL_PATH     = Path(r"D:\STUDIES\FINAL YEAR PROJECT\Models\YOLO11m_P2P3_CBAM_FineTuned10\weights\best.pt")
DATASET_YAML        = Path(r"D:\STUDIES\FINAL YEAR PROJECT\Final Dataset\RoadHealthDataset_Final3000\data.yaml")

IMG_SIZE   = 640
BATCH_SIZE = 8
DEVICE     = 0 if torch.cuda.is_available() else "cpu"


# ── CBAM Attention Module Definitions ──────────────────────────
class ChannelAttention(nn.Module):
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
        mx  = self.mlp(self.max_pool(x).view(B, C))
        return x * torch.sigmoid(avg + mx).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int = 128, reduction: int = 4):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class P2P3CBAMDetail(nn.Module):
    def __init__(self, in_ch: int = 256, out_ch: int = 256, hidden: int = 128, reduction: int = 4):
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


def _p2p3_cbam_predict(self, x, profile=False, visualize=False, augment=False, embed=None):
    if augment:
        raise NotImplementedError("Augmented prediction is not supported for P2P3-CBAM model.")

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


# Monkey-patch on DetectionModel class
tasks.DetectionModel._p2p3_cbam_predict = _p2p3_cbam_predict
tasks.DetectionModel.P2P3CBAMDetail = P2P3CBAMDetail

_main = sys.modules.get("__main__")
if _main is not None:
    setattr(_main, "P2P3CBAMDetail", P2P3CBAMDetail)
    setattr(_main, "CBAM", CBAM)
    setattr(_main, "ChannelAttention", ChannelAttention)
    setattr(_main, "SpatialAttention", SpatialAttention)
    setattr(_main, "_p2p3_cbam_predict", _p2p3_cbam_predict)


# ── Model Loader Function ──────────────────────────────────────
def load_p2p3_cbam_model(baseline_path: Path, cbam_path: Path, device: str = "cuda:0") -> YOLO:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[1/4] Loading baseline YOLO11m structure from: {baseline_path.name}")
    model_wrapper = YOLO(str(baseline_path))
    net = model_wrapper.model.to(dev)
    net.eval()

    print("[2/4] Attaching P2P3CBAMDetail attention module...")
    net.p2p3_cbam_detail = P2P3CBAMDetail().to(dev)

    print("[3/4] Binding custom forward pass...")
    bound_predict = types.MethodType(_p2p3_cbam_predict, net)
    net.p2p3_cbam_predict = bound_predict
    net.predict = bound_predict

    print(f"[4/4] Loading learned champion weights from: {cbam_path.name}")
    checkpoint = torch.load(str(cbam_path), map_location="cpu", weights_only=False)
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

    missing, unexpected = net.load_state_dict(learned_state, strict=True, assign=True)
    assert len(missing) == 0, f"Missing weights: {missing}"
    assert len(unexpected) == 0, f"Unexpected weights: {unexpected}"

    net = net.to(dev)
    net.eval()
    model_wrapper.model = net

    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n🎉 YOLO11m-P2P3-CBAM Model Loaded Successfully! ({total_params:,} parameters)")
    return model_wrapper


# ── Main Execution ─────────────────────────────────────────────
def main():
    print("=" * 75)
    print("      🛣️ YOLO11m-P2P3-CBAM MODEL VALIDATION & PERFORMANCE EVALUATION")
    print("=" * 75)

    if not BASELINE_MODEL_PATH.exists():
        print(f"❌ Baseline checkpoint not found: {BASELINE_MODEL_PATH}")
        return
    if not CBAM_MODEL_PATH.exists():
        print(f"❌ CBAM checkpoint not found: {CBAM_MODEL_PATH}")
        return
    if not DATASET_YAML.exists():
        print(f"❌ Dataset YAML not found: {DATASET_YAML}")
        return

    device_str = f"cuda:{DEVICE}" if isinstance(DEVICE, int) else DEVICE
    model = load_p2p3_cbam_model(BASELINE_MODEL_PATH, CBAM_MODEL_PATH, device=device_str)

    print("\n🚀 Executing validation on test split (900 test images)...")
    t0 = time.time()
    val_results = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=0,
        device=DEVICE,
        plots=True,
        save_json=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Extract metrics
    metrics_dict = val_results.results_dict
    p_overall    = metrics_dict.get("metrics/precision(B)", 0.0)
    r_overall    = metrics_dict.get("metrics/recall(B)", 0.0)
    map50_all    = metrics_dict.get("metrics/mAP50(B)", 0.0)
    map50_95_all = metrics_dict.get("metrics/mAP50-95(B)", 0.0)
    f1_overall   = 2 * (p_overall * r_overall) / max(1e-6, (p_overall + r_overall))

    class_names = val_results.names
    p_classes   = val_results.box.p
    r_classes   = val_results.box.r
    map50_cls   = val_results.box.ap50
    map5095_cls = val_results.box.ap
    f1_cls      = val_results.box.f1

    rows = []
    for c_idx, c_name in class_names.items():
        p_val = float(p_classes[c_idx]) if c_idx < len(p_classes) else 0.0
        r_val = float(r_classes[c_idx]) if c_idx < len(r_classes) else 0.0
        m50   = float(map50_cls[c_idx]) if c_idx < len(map50_cls) else 0.0
        m5095 = float(map5095_cls[c_idx]) if c_idx < len(map5095_cls) else 0.0
        f1_v  = float(f1_cls[c_idx]) if c_idx < len(f1_cls) else 2 * (p_val * r_val) / max(1e-6, p_val + r_val)
        rows.append({
            "Distress Class": c_name.title(),
            "Precision (P)": f"{p_val*100:.2f}%",
            "Recall (R)": f"{r_val*100:.2f}%",
            "F1-Score": f"{f1_v*100:.2f}%",
            "mAP @ 0.50": f"{m50*100:.2f}%",
            "mAP @ 0.50:0.95": f"{m5095*100:.2f}%",
        })

    rows.append({
        "Distress Class": "ALL CLASSES (Overall)",
        "Precision (P)": f"{p_overall*100:.2f}%",
        "Recall (R)": f"{r_overall*100:.2f}%",
        "F1-Score": f"{f1_overall*100:.2f}%",
        "mAP @ 0.50": f"{map50_all*100:.2f}%",
        "mAP @ 0.50:0.95": f"{map50_95_all*100:.2f}%",
    })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 75)
    print("           🏆 YOLO11m-P2P3-CBAM OFFICIAL VALIDATION RESULTS")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)
    print(f"Validation completed in {elapsed:.2f}s ({elapsed/60:.2f} mins).")
    print(f"Artifacts and curve plots saved to: {val_results.save_dir}")


if __name__ == "__main__":
    main()
