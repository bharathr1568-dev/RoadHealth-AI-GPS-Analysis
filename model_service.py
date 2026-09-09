"""
RoadHealth AI — Model Service
Loads the proposed YOLO11m-P2P3-CBAM (Champion) model with Streamlit caching.

Architecture:
    Baseline YOLO11m is used as the Ultralytics wrapper shell.
    The P2→P3 detail branch is enhanced with CBAM (Channel & Spatial Attention)
    and the custom forward pass is bound before loading the fine-tuned learned weights.
"""

import types
import torch
import torch.nn as nn
import streamlit as st
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from config import BASELINE_MODEL_PATH, CBAM_MODEL_PATH


# ── CBAM Attention Modules ─────────────────────────────────────

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


# ── P2→P3 CBAM Detail Branch ───────────────────────────────────

class P2P3CBAMDetail(nn.Module):
    """
    Detail branch extracting high-resolution features from P2,
    filtering noise via CBAM, and fusing residually into P3.
    """

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


# ── Custom Forward Pass ────────────────────────────────────────

def _p2p3_cbam_predict(self, x, profile=False, visualize=False, augment=False, embed=None):
    """
    Exact P2→P3 CBAM forward pass.
    Captures P2 at layer 2, extracts & attends detail, and adds into P3 at layer 16.
    """
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


# Monkey-patch on DetectionModel class for pickle compatibility
tasks.DetectionModel._p2p3_cbam_predict = _p2p3_cbam_predict
tasks.DetectionModel.P2P3CBAMDetail = P2P3CBAMDetail

# Also inject into __main__ so torch.load unpickler resolves it seamlessly in all contexts
import sys
_main_mod = sys.modules.get("__main__")
if _main_mod is not None:
    setattr(_main_mod, "P2P3CBAMDetail", P2P3CBAMDetail)
    setattr(_main_mod, "CBAM", CBAM)
    setattr(_main_mod, "ChannelAttention", ChannelAttention)
    setattr(_main_mod, "SpatialAttention", SpatialAttention)
    setattr(_main_mod, "_p2p3_cbam_predict", _p2p3_cbam_predict)


# ── Model Loader ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    """
    Reconstruct and load the fine-tuned YOLO11m-P2P3-CBAM champion model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure __main__ module has classes registered
    _main = sys.modules.get("__main__")
    if _main is not None:
        setattr(_main, "P2P3CBAMDetail", P2P3CBAMDetail)
        setattr(_main, "CBAM", CBAM)
        setattr(_main, "ChannelAttention", ChannelAttention)
        setattr(_main, "SpatialAttention", SpatialAttention)
        setattr(_main, "_p2p3_cbam_predict", _p2p3_cbam_predict)


    # 1. Load baseline YOLO11m inner model (architecture shell)
    baseline = YOLO(str(BASELINE_MODEL_PATH))
    net = baseline.model.to(device)
    net.eval()

    # 2. Attach P2→P3 CBAM detail module
    net.p2p3_cbam_detail = P2P3CBAMDetail().to(device)

    # 3. Bind custom forward pass
    bound_predict = types.MethodType(_p2p3_cbam_predict, net)
    net.p2p3_cbam_predict = bound_predict
    net.predict = bound_predict

    # 4. Load learned weights from best.pt
    checkpoint = torch.load(str(CBAM_MODEL_PATH), map_location="cpu", weights_only=False)
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

    # Load weights
    missing, unexpected = net.load_state_dict(learned_state, strict=True, assign=True)
    assert len(missing) == 0, f"Missing weights: {missing}"
    assert len(unexpected) == 0, f"Unexpected weights: {unexpected}"

    net = net.to(device)
    net.eval()

    # 5. Wrap back into Ultralytics YOLO interface
    model = YOLO(str(BASELINE_MODEL_PATH))
    model.model = net

    return model


def get_device() -> tuple[str, str]:
    """Return device string and friendly name."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", f"GPU ({name})"
    return "cpu", "CPU"


