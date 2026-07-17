# mcp_lo_vlm

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes **LO-VLM**
(Layer-wise OCT Vision-Language Model) — a retinal OCT captioning model — as an
MCP tool, deployed via Prefect Horizon.

The server is deliberately lightweight: image preprocessing (decode, validate,
resize) runs locally on CPU, while GPU inference is dispatched to a **Modal**
serverless endpoint. No model weights or CUDA are baked into the container, so it
stays small and cheap to run.

**Deployed MCP endpoint:** `https://lo-vlm.fastmcp.app/mcp`

## Architecture

```
MCP client
    │  base64 OCT B-scan
    ▼
lo-vlm-mcp  @ https://lo-vlm.fastmcp.app/mcp   (this server — CPU only)
    │  decode · validate · resize to 512px
    ▼
Modal serverless endpoint  (GPU: LO-VLM / BLIP)
    │  layer-level narrative caption
    ▼
JSON response
```

Two URLs are involved and serve different roles: clients connect to the
**MCP endpoint** (`https://lo-vlm.fastmcp.app/mcp`); that server in turn
dispatches GPU work to the **Modal endpoint** (`MODAL_ENDPOINT_URL`, below).

Preprocessing and inference are separated so the always-on MCP layer needs no
GPU, and the model only spins up on Modal when a request arrives.

## Tools

| Tool | Purpose | Returns |
|------|---------|---------|
| `caption_oct` | Generate a layer-level narrative caption for an OCT B-scan | Caption text, timing, and image dimensions |
| `health` | Liveness probe | Status and Modal endpoint configuration |

LO-VLM is a BLIP-based model fine-tuned on paired OCT images and layer-level
clinical captions. It describes retinal anatomy and pathology per layer (RNFL,
GCL+IPL, INL, OPL, ONL, IS/OS, RPE) and common findings such as subretinal
fluid, pigment epithelial detachment, and ellipsoid zone disruption.

Typical uses: first-pass AMD / DME / glaucoma narratives, structured input for a
downstream LLM to expand into a full clinical report, and batch captioning of
OCT datasets for labeling or retrieval.

### `caption_oct` arguments

- `bscan_b64` — base64 OCT B-scan (PNG or JPEG, grayscale or RGB), minimum 32×32.
- `image_id` — identifier echoed back for tracing.
- `max_length` — maximum caption token length (default 256; raise to 384+ for
  more detail).
- `num_beams` — beam search width (default 4; higher improves quality at the cost
  of latency).
- `prompt` — optional text prompt to steer the caption.

Larger inputs are downscaled to a 512px longest edge before dispatch. Tools
return a JSON object with a `success` flag; failures report a `reason`
(validation) or `error` (runtime) rather than raising.

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `MODAL_ENDPOINT_URL` | yes | Modal endpoint URL, e.g. `https://mathgcloud--lo-vlm-api.modal.run` |
| `MAX_SIDE` | no | Resize longest edge before sending (default `512`) |
| `DEFAULT_MAX_LENGTH` | no | Default caption token length (default `256`) |
| `DEFAULT_NUM_BEAMS` | no | Default beam search width (default `4`) |

## Connecting

The server is hosted at `https://lo-vlm.fastmcp.app/mcp`. Point any MCP client at
that URL — for example, in a client config:

```json
{
  "mcpServers": {
    "lo-vlm": {
      "url": "https://lo-vlm.fastmcp.app/mcp"
    }
  }
}
```

To self-host instead, run the server yourself as below.

## Running

### Docker

```bash
docker build -t lo-vlm-mcp .
docker run -p 8080:8080 \
  -e MODAL_ENDPOINT_URL=https://<your-modal-endpoint>.modal.run \
  lo-vlm-mcp
```

The image is a multi-stage `python:3.11-slim` build that runs as a non-root user
and exposes port 8080.

### Local

```bash
pip install -r requirements.txt
export MODAL_ENDPOINT_URL=https://<your-modal-endpoint>.modal.run
python server.py
```

The server runs over stateless HTTP with JSON responses and accepts request
bodies up to 32 MB.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
