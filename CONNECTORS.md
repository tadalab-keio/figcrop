# AI Connectors

figcrop can be exposed to AI tools in two practical ways:

1. **OpenAPI / REST** via the built-in FastAPI server.
2. **MCP** via the optional `figcrop_mcp.py` stdio server.

Use OpenAPI when an AI product wants an HTTP schema. Use MCP when Claude,
Codex, or another local agent can run a tool server on the same machine as the
PDF files.

## OpenAPI / REST

Start the server:

```powershell
.\.venv\Scripts\python.exe figcrop.py serve auto
```

Discovery endpoints:

- `GET http://127.0.0.1:8077/`
- `GET http://127.0.0.1:8077/openapi.json`
- `GET http://127.0.0.1:8077/health`

Extraction request:

```bash
curl -s -X POST http://127.0.0.1:8077/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf":"paper.pdf","out_dir":"out","figs":[1,2],"mode":"caption"}'
```

Important request fields:

- `pdf`: path to the input PDF on the machine running the server
- `out_dir`: output directory for JPEG crops and `figures.json`
- `figs`: optional real figure numbers
- `top`: optional positional fallback
- `mode`: `figure`, `panel`, or `caption`
- `panels`: legacy alias for `mode=panel`
- `trim_mode`: `mask` or `whiteband`
- `caption_mode`: legacy alias; `include` means `mode=caption`

The server is local by default (`127.0.0.1`). Remote AI products cannot call it
unless you intentionally expose it through a secure tunnel or hosted deployment.

## MCP

If installed with pipx from PyPI, the MCP command is available directly:

```powershell
pipx install figcrop
figcrop-mcp
```

For unreleased GitHub source, use
`pipx install git+https://github.com/tadalab-keio/figcrop.git`.

For a source checkout created with `setup.ps1` / `setup.sh`, install the optional
MCP dependency:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-mcp.txt
```

Run the MCP server:

```powershell
.\.venv\Scripts\python.exe figcrop_mcp.py
```

The MCP tool is named `extract_figures` and accepts:

```json
{
  "pdf": "paper.pdf",
  "out_dir": "out",
  "figs": [1, 2],
  "top": null,
  "mode": "panel",
  "panels": false,
  "trim_mode": "mask",
  "caption_mode": "exclude",
  "device": "auto"
}
```

It returns the output directory, crop count, `figures.json` path, and manifest
entries.

### Example Local MCP Config

Use absolute paths when configuring external clients:

```json
{
  "mcpServers": {
    "figcrop": {
      "command": "figcrop-mcp",
      "args": []
    }
  }
}
```

If the client does not inherit the pipx command PATH, use the absolute path
printed by `pipx list` for `figcrop-mcp`.

Keep the server local unless you have a clear security model. PDFs and output
paths may contain private research material.

## AI-Side Verification

An AI agent should treat extraction as incomplete until it has inspected the
images. Recommended flow:

1. Run extraction.
2. Read `figures.json`.
3. Create or open a montage of JPEGs.
4. Check for clipped axes/captions and neighbor frame lines.
5. Retry with `figs=`, `top=`, `mode=panel`, or `trim=whiteband` if needed.
