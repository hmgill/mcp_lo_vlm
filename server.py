"""
server.py — lo-vlm-mcp
=======================
FastMCP server exposing LO-VLM retinal OCT captioning as an MCP tool,
deployed via Prefect Horizon.

Image preprocessing runs locally (decode + validate + resize); GPU inference
is dispatched to the Modal serverless endpoint — Horizon needs no GPU or
model weights.

Required environment variables:
    MODAL_ENDPOINT_URL   Full Modal endpoint URL,
                         e.g. https://mathgcloud--lo-vlm-api.modal.run

Optional environment variables:
    MAX_SIDE             Resize longest edge before sending (default: 512).
    DEFAULT_MAX_LENGTH   Maximum caption token length (default: 256).
    DEFAULT_NUM_BEAMS    Beam search width (default: 4).

Tools:
    caption_oct(bscan_b64, image_id, max_length, num_beams)
        → OCT layer-level narrative caption

    health()
        → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone

import requests
from fastmcp import FastMCP
from PIL import Image

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_ENDPOINT_URL  = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")
DEFAULT_MAX_SIDE    = int(os.environ.get("MAX_SIDE",            "512"))
DEFAULT_MAX_LENGTH  = int(os.environ.get("DEFAULT_MAX_LENGTH",  "256"))
DEFAULT_NUM_BEAMS   = int(os.environ.get("DEFAULT_NUM_BEAMS",   "4"))

if not MODAL_ENDPOINT_URL:
    logger.warning("MODAL_ENDPOINT_URL is not set — inference calls will fail.")


# ── Modal client ──────────────────────────────────────────────────────────────

def _modal_dispatch(payload: dict, image_id: str) -> dict:
    """POST payload to the Modal LO-VLM worker and return parsed response."""
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    logger.info(f"[{image_id}] Dispatching to Modal: {MODAL_ENDPOINT_URL}/infer")
    resp = requests.post(
        f"{MODAL_ENDPOINT_URL}/infer",
        json=payload,
        timeout=120,   # T4 cold-start ~30s; BLIP inference ~1s once warm
    )
    if not resp.ok:
        # Surface validation errors (422) clearly in logs
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        logger.error(f"[{image_id}] Modal {resp.status_code}: {detail}")
        resp.raise_for_status()
    return resp.json()


# ── Preprocessing — runs locally, no GPU needed ───────────────────────────────

def _preprocess_bscan(
    bscan_b64: str,
    image_id:  str,
    max_side:  int = DEFAULT_MAX_SIDE,
) -> tuple[str, int, int, int, int]:
    """
    Decode, validate, and optionally downscale a base64 OCT B-scan.

    Returns:
        (clean_b64, sent_w, sent_h, orig_w, orig_h)
    """
    logger.info(f"[{image_id}] Preprocessing bscan (len={len(bscan_b64)})")
    try:
        raw = base64.b64decode(bscan_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"[{image_id}] Could not decode bscan: {e}")

    orig_w, orig_h = img.size
    if orig_w < 32 or orig_h < 32:
        raise ValueError(
            f"[{image_id}] B-scan too small ({orig_w}×{orig_h}). "
            "Minimum 32×32 expected."
        )

    if max(orig_w, orig_h) > max_side:
        scale  = max_side / max(orig_w, orig_h)
        sent_w = max(1, int(orig_w * scale))
        sent_h = max(1, int(orig_h * scale))
        img    = img.resize((sent_w, sent_h), Image.LANCZOS)
        logger.info(
            f"[{image_id}] bscan: resized {orig_w}×{orig_h} → {sent_w}×{sent_h}"
        )
    else:
        sent_w, sent_h = orig_w, orig_h
        logger.info(f"[{image_id}] bscan: {orig_w}×{orig_h}, no resize needed")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    clean_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return clean_b64, sent_w, sent_h, orig_w, orig_h


# ── FastMCP app ───────────────────────────────────────────────────────────────

mcp = FastMCP("lo-vlm-mcp")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def caption_oct(
    bscan_b64:  str,
    image_id:   str,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams:  int = DEFAULT_NUM_BEAMS,
    prompt:     str = "",
) -> str:
    """
    Generate a retinal layer-level narrative caption from an OCT B-scan.

    LO-VLM (Layer-wise OCT Vision-Language Model) is a BLIP-based model
    fine-tuned on paired OCT images and layer-level clinical captions.
    It describes retinal anatomy and pathology visible in the B-scan,
    focusing on individual layer structures (RNFL, GCL+IPL, INL, OPL,
    ONL, IS/OS, RPE) and common findings such as subretinal fluid,
    pigment epithelial detachment, and ellipsoid zone disruption.

    Suitable for:
        - First-pass AMD / DME / glaucoma narrative generation
        - Structured input for a downstream LLM (e.g. Claude) to expand
          into a full clinical report
        - Batch captioning of OCT datasets for labelling or retrieval

    Args:
        bscan_b64:   Base64-encoded OCT B-scan (PNG or JPEG, greyscale or RGB).
        image_id:    Identifier for logging and tracing.
        max_length:  Maximum caption token length (default 256).
                     Increase to 384+ for more detailed descriptions.
        num_beams:   Beam search width (default 4).
                     Higher values improve quality at the cost of latency.

    Returns:
        JSON with:
          caption      — generated layer-level text description
          image_id     — echoed for tracing
          model        — model variant used ("base")
          elapsed_s    — inference time in seconds
          image_width  — original image width (pixels)
          image_height — original image height (pixels)
          created_at   — ISO 8601 timestamp
    """
    try:
        clean_b64, _, _, orig_w, orig_h = _preprocess_bscan(
            bscan_b64, image_id
        )

        payload = {
            "image_b64":  clean_b64,
            "image_id":   image_id,
            "model":      "base",
            "max_length": max_length,
            "num_beams":  num_beams,
            "prompt":     prompt,
        }

        result = _modal_dispatch(payload, image_id)

        out = json.dumps({
            "success":      True,
            "image_id":     image_id,
            "caption":      result.get("caption", ""),
            "model":        result.get("model", "base"),
            "elapsed_s":    result.get("elapsed_s", 0.0),
            "image_width":  orig_w,
            "image_height": orig_h,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"caption_oct: {image_id}  "
            f"elapsed={result.get('elapsed_s', '?')}s  "
            f"caption_len={len(result.get('caption', ''))}  "
            f"payload={len(out)/1024:.1f}KB"
        )
        return out

    except ValueError as e:
        return json.dumps({"success": False, "reason": str(e), "image_id": image_id})
    except Exception as e:
        logger.error(f"caption_oct failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "lo-vlm-mcp",
        "modal": {
            "endpoint_url": MODAL_ENDPOINT_URL or "(not set)",
            "configured":   bool(MODAL_ENDPOINT_URL),
        },
        "defaults": {
            "max_side":    DEFAULT_MAX_SIDE,
            "max_length":  DEFAULT_MAX_LENGTH,
            "num_beams":   DEFAULT_NUM_BEAMS,
            "model":       "base",
        },
    })


if __name__ == "__main__":
    mcp.run(
        stateless_http=True,
        json_response=True,
        max_request_body_size=32 * 1024 * 1024,   # 32 MB — OCT B-scans are small
    )
