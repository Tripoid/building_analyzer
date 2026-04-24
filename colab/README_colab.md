# AlegroCode — JupyterLab launcher

This directory contains the canonical way to run the AlegroCode backend on a
rented GPU box: as a non-blocking FastAPI server spawned from a Jupyter cell.

## TL;DR

```bash
# 1. On the rented machine
cd /path/to/building_analyzer
pip install -r backend/requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git   # SAM2 weights pulled lazily

# 2. Set credentials
export ALEGRO_NGROK_AUTHTOKEN=...your ngrok token...
export ALEGRO_INPAINT_PROVIDER=lama          # or "sd" for higher-quality inpaint

# 3. Open colab_server_v2.ipynb and run cells 1→5 in order
```

## Why this pattern

1. `nest_asyncio.apply()` patches the notebook event loop so FastAPI/uvicorn
   can use it without fighting over `asyncio`.
2. `asyncio.ensure_future(server.serve())` schedules uvicorn as a background
   task — the cell returns immediately, the notebook stays responsive.
3. Models live in kernel globals (`analyzer = FacadeAnalyzer(); analyzer.load_models()`)
   so restarting the server (Cell 6 → Cell 3) does **not** re-download weights.
4. `server.should_exit = True` (Cell 6) gracefully stops the task without
   killing the kernel.

## Troubleshooting

### The cell hangs forever
Make sure you ran Cell 1 first — `nest_asyncio.apply()` is mandatory.
If you still get `RuntimeError: This event loop is already running`, restart
the kernel and try again (stale `asyncio` state in the kernel can linger).

### ngrok "browser-warning" page in Flutter
The Flutter client already sends `ngrok-skip-browser-warning: true` on every
request — if you still see HTML instead of JSON, upgrade your pyngrok to
≥7.0 and make sure the tunnel is HTTP (`ngrok.connect(8000, 'http')`).

### Out of VRAM during inpaint
Set `ALEGRO_INPAINT_PROVIDER=lama` (default). The SD provider needs ~4GB
free VRAM; LaMa is happy with 1-2GB. The system will automatically fall
back to LaMa if SD fails with `MemoryError`.

### Fallback pattern (very rare)
If `nest_asyncio` misbehaves on an unusual Jupyter build, replace Cell 3 with:

```python
import threading, asyncio, uvicorn

def _run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(app, host='0.0.0.0', port=8000, loop='asyncio')
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

thread = threading.Thread(target=_run, daemon=True)
thread.start()
```

This trades graceful shutdown for a guaranteed non-blocking start.
