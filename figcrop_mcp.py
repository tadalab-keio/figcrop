"""Optional MCP server for figcrop.

Run with:
    python figcrop_mcp.py

Requires:
    uv pip install --python .venv\\Scripts\\python.exe -r requirements-mcp.txt
"""

from __future__ import annotations

from typing import Any

import figtools


try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise SystemExit(
        "Missing optional MCP dependency. Install with:\n"
        "  uv pip install --python .venv\\Scripts\\python.exe -r requirements-mcp.txt"
    ) from exc


mcp = FastMCP("figcrop")
_MODEL_CACHE: dict[str, tuple[Any, str]] = {}


def _model(device: str) -> tuple[Any, str]:
    key = device or "auto"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = figtools._engine(key)
    return _MODEL_CACHE[key]


@mcp.tool()
def extract_figures(
    pdf: str,
    out_dir: str,
    figs: list[int] | None = None,
    top: int | None = None,
    panels: bool = False,
    trim_mode: str = "mask",
    caption_mode: str = "exclude",
    device: str = "auto",
) -> dict[str, Any]:
    """Extract figures/tables from a research-paper PDF.

    Args:
        pdf: Path to the input PDF.
        out_dir: Directory where JPEG crops and figures.json will be written.
        figs: Optional real Fig/Table numbers to extract.
        top: Optional positional fallback: first N visual regions per page.
        panels: Output detected regions/panels separately instead of whole figures.
        trim_mode: "mask" (default) or "whiteband".
        caption_mode: "exclude" (default) or "include".
        device: Layout backend/device, usually "auto".
    """
    manifest = figtools.extract(
        pdf,
        out_dir,
        model=_model(device),
        figs=figs,
        top=top,
        panels=panels,
        trim_mode=trim_mode,
        caption_mode=caption_mode,
    )
    clean_out = out_dir.rstrip("/\\")
    return {
        "out_dir": out_dir,
        "count": len(manifest),
        "figures_json": f"{clean_out}\\figures.json",
        "figures": manifest,
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
