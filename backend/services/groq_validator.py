"""
RoadHealth AI — Image-Level Validation & Quality Gate
Uses existing Groq Vision integration as an image-level gatekeeper to determine:
1. Is the image actually a road scene?
2. Is the road surface sufficiently visible and usable?
3. Is the image too dark for reliable analysis?
4. Is there severe sunlight/glare affecting visibility?
5. Is the image severely blurred/out of focus?

If the image is invalid or poor quality, YOLO inference is bypassed and a clear,
actionable user message is provided. If valid, the image proceeds to YOLO11m-P2P3-CBAM.
"""

import io
import re
import json
import base64
import logging
import requests
import cv2
import numpy as np

from backend.config import PROJECT_ROOT
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import GROQ_API_KEY

logger = logging.getLogger("roadhealth.quality_gate")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
MAX_PAYLOAD_DIM = 360


def _compress_bgr_to_base64(img_bgr: np.ndarray, max_dim: int = MAX_PAYLOAD_DIM) -> str:
    """Compress BGR OpenCV image to a compact Base64 JPEG to preserve token quota."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 45])
    return base64.b64encode(buf).decode("utf-8")


def _local_heuristic_check(img_bgr: np.ndarray) -> dict:
    """
    Sub-millisecond local vision heuristics used for safety checks and
    resilient fallback when network or external API limits occur.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_b = float(np.mean(gray))
    std_c = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 1. Extreme darkness check
    if mean_b < 24.0:
        return {
            "is_road": False, "is_usable": False, "too_dark": True,
            "severe_glare": False, "too_blurry": False, "is_valid": False,
            "user_message": "The image is too dark for reliable road-defect analysis. Please capture another image with better lighting.",
            "reason": f"Extreme low scene brightness ({mean_b:.1f}/255)",
            "source": "local_heuristic"
        }

    # 2. Extreme sun flare / overexposed upper frame
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    bright_ratio = np.mean(v_chan > 230)
    top_bright_ratio = np.mean(v_chan[:int(h * 0.4), :] > 220)
    if bright_ratio > 0.45 or (top_bright_ratio > 0.60 and mean_b > 180.0):
        return {
            "is_road": False, "is_usable": False, "too_dark": False,
            "severe_glare": True, "too_blurry": False, "is_valid": False,
            "user_message": "Strong sunlight/glare is affecting road visibility. Please capture another image from a better angle.",
            "reason": "Severe sun flare / overexposure detected",
            "source": "local_heuristic"
        }

    # 3. Severe blur check
    if lap_var < 15.0:
        return {
            "is_road": True, "is_usable": False, "too_dark": False,
            "severe_glare": False, "too_blurry": True, "is_valid": False,
            "user_message": "The image is too blurry for reliable analysis. Please capture a clearer image.",
            "reason": f"Severe blur (Laplacian variance {lap_var:.1f})",
            "source": "local_heuristic"
        }

    return {
        "is_road": True, "is_usable": True, "too_dark": False,
        "severe_glare": False, "too_blurry": False, "is_valid": True,
        "user_message": "", "reason": "Road surface verified",
        "source": "local_heuristic"
    }


def validate_road_image_quality(img_bgr: np.ndarray) -> dict:
    """
    Image-Level Quality Gate.
    Answers 6 image-level questions:
    1. Is the image actually a road scene? (is_road)
    2. Is the road surface sufficiently visible? (is_usable)
    3. Is the image too dark for reliable analysis? (too_dark)
    4. Is there severe sunlight/glare affecting visibility? (severe_glare)
    5. Is the image severely blurred/out of focus? (too_blurry)
    6. Is the image generally suitable for road-defect analysis? (is_valid)

    Returns a structured dictionary with decision flags and user-facing messages.
    """
    # 1. Quick local baseline check
    local_eval = _local_heuristic_check(img_bgr)
    if not local_eval["is_valid"]:
        return local_eval

    # 2. If API Key is not present, use local evaluation
    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        return local_eval

    # 3. Call Groq Vision AI for multimodal scene validation
    try:
        b64_data = _compress_bgr_to_base64(img_bgr)
        prompt = (
            "Task: Image Quality Gate for Road Condition AI.\n"
            "Evaluate ONLY the image-level scene quality:\n"
            "1. is_road: Is this an outdoor road, street, highway, asphalt, or concrete road surface? (false if sky, sunset, building silhouette, indoor room, person, vehicle interior)\n"
            "2. is_usable: Is the road surface clearly visible for defect inspection?\n"
            "3. too_dark: Is the scene too dark/underexposed for reliable inspection?\n"
            "4. severe_glare: Is there strong sunlight glare obscuring the road?\n"
            "5. too_blurry: Is the image severely blurry or out of focus?\n\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "is_road": true/false,\n'
            '  "is_usable": true/false,\n'
            '  "too_dark": true/false,\n'
            '  "severe_glare": true/false,\n'
            '  "too_blurry": true/false,\n'
            '  "reason": "Brief 1-sentence reason"\n'
            "}"
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": GROQ_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }

        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=6)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            json_match = re.search(r"\{.*?\}", clean_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                is_road = bool(parsed.get("is_road", True))
                is_usable = bool(parsed.get("is_usable", True))
                too_dark = bool(parsed.get("too_dark", False))
                severe_glare = bool(parsed.get("severe_glare", False))
                too_blurry = bool(parsed.get("too_blurry", False))
                reason = str(parsed.get("reason", ""))

                # Decision Logic as specified:
                if not is_road:
                    return {
                        "is_road": False, "is_usable": False, "too_dark": too_dark,
                        "severe_glare": severe_glare, "too_blurry": too_blurry,
                        "is_valid": False,
                        "user_message": "This image does not appear to contain a road scene. Please upload a clear road image.",
                        "reason": reason, "source": "ai_vision_gate"
                    }
                elif too_dark:
                    return {
                        "is_road": True, "is_usable": False, "too_dark": True,
                        "severe_glare": severe_glare, "too_blurry": too_blurry,
                        "is_valid": False,
                        "user_message": "The image is too dark for reliable road-defect analysis. Please capture another image with better lighting.",
                        "reason": reason, "source": "ai_vision_gate"
                    }
                elif severe_glare:
                    return {
                        "is_road": True, "is_usable": False, "too_dark": False,
                        "severe_glare": True, "too_blurry": too_blurry,
                        "is_valid": False,
                        "user_message": "Strong sunlight/glare is affecting road visibility. Please capture another image from a better angle.",
                        "reason": reason, "source": "ai_vision_gate"
                    }
                elif too_blurry or not is_usable:
                    return {
                        "is_road": True, "is_usable": False, "too_dark": False,
                        "severe_glare": False, "too_blurry": True,
                        "is_valid": False,
                        "user_message": "The image is too blurry for reliable analysis. Please capture a clearer image.",
                        "reason": reason, "source": "ai_vision_gate"
                    }
                else:
                    return {
                        "is_road": True, "is_usable": True, "too_dark": False,
                        "severe_glare": False, "too_blurry": False,
                        "is_valid": True,
                        "user_message": "",
                        "reason": reason, "source": "ai_vision_gate"
                    }
        else:
            logger.info(f"AI Vision quality gate status {resp.status_code}, using local heuristic evaluation.")
    except Exception as exc:
        logger.info(f"AI Vision quality gate exception ({exc}), using local heuristic evaluation.")

    return local_eval
