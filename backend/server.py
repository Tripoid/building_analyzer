"""
Thin uvicorn entrypoint for non-Jupyter runs.

Prefer `colab/colab_server_v2.ipynb` when launching on a rented GPU — the
notebook preloads models into the kernel globals and runs the server with
`nest_asyncio` so the cell does not block.
"""

from backend.api import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    from backend.core.config import get_settings

    s = get_settings()
    uvicorn.run(app, host=s.host, port=s.port)
