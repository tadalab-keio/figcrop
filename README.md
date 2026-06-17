# figcrop

Extract figures from research-paper PDFs by real figure number.

figcrop uses MinerU's PP-DocLayoutV2 layout model for visual-region detection,
then adds its own PDF-text and geometry logic to group panels into whole figures,
remove captions, and trim page furniture. It is designed for dense semiconductor
papers where simple heuristic tools often miss figures or split panels badly.

## Features

- Extract whole `Fig.N` outputs, including multi-panel figures, by the real figure
  number found in the PDF text layer.
- Keep figure-internal labels, process-flow text, axes, table borders, and panels,
  while excluding the caption line.
- Run as a persistent local HTTP server so the layout model is loaded once.
- Use OpenVINO by default for fast local inference; torch backends remain available.
- Trim modes:
  - `mask` default: fast trim using an ignore mask for captions, page furniture,
    neighbor frame lines, and thin anti-aliased edge bleed.
  - `whiteband`: extra local whitespace snapping around the detector bbox. This is
    slower but useful as a conservative safety mode for difficult crops.
- Handles slide/poster-style cells in some PDFs by clipping giant page-level
  fallback detections to the caption's local cell.

## Requirements

- Python 3.10-3.13. Windows is the main tested environment.
- `uv` is recommended; the setup scripts install into the project-local `.venv`.
- First run downloads model weights from Hugging Face.
- OpenVINO GPU is the recommended default. NPU is not currently useful for this
  RT-DETR layout model.

## Setup

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1 -Device xpu
```

Linux/macOS:

```bash
bash setup.sh
```

`requirements.txt` is a manifest for the base packages. Device-specific torch
packages are handled by the setup scripts.

## Usage

`<py>` means `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Unix.

Start the local server:

```powershell
<py> figtools.py serve auto
```

Then request crops:

```bash
curl -s -X POST http://127.0.0.1:8077/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf":"paper.pdf","out_dir":"out","figs":[1,2]}'
```

Useful request fields:

- `figs`: real figure numbers to extract, for example `[1,2]`.
- `top`: fallback positional extraction, for example `2` for the first two visual
  regions per page.
- `panels`: `true` to output detected panels/regions separately instead of whole
  figures.
- `trim_mode`: `"mask"` default or `"whiteband"`.

One-shot CLI:

```powershell
<py> figtools.py extract paper.pdf out auto
<py> figtools.py extract paper.pdf out auto figs=1,2
<py> figtools.py extract paper.pdf out_whiteband auto trim=whiteband
```

Output files are JPEG crops plus a `figures.json` manifest in `out_dir`.

## How It Works

1. Render each PDF page at 150 dpi for layout detection.
2. Run PP-DocLayoutV2 through OpenVINO or torch.
3. Keep visual regions labeled `image`, `chart`, or `table`.
4. Read `Fig.N` / `Table N` captions directly from the PDF text layer.
5. Assign each region to the nearest same-column caption below it.
6. Union regions with the same figure number into one whole-figure crop.
7. Render the page at 300 dpi and crop with the selected trim mode.

The numbering and whole-figure grouping are local geometry logic, not MinerU's
full reading-order pipeline.

## Performance

For repeated extraction, use the server. In local tests, server mode avoids
reloading the model for each request and is much faster than one-shot CLI runs,
especially on one-page PDFs.

For occasional use, the one-shot CLI is still practical: most ordinary papers
finish in a few seconds on a local OpenVINO GPU setup, so running the server is a
convenience rather than a hard requirement.

The default `mask` trim mode is optimized for speed and is the recommended mode.
`whiteband` is kept as a higher-conservatism option when local whitespace around
the detector bbox matters more than speed.

## Limitations

- Very dense pages can still confuse region-to-caption assignment. Use `top=` or
  `panels=true` as fallbacks.
- Some PDFs contain slide/poster grids or decorative page furniture that can look
  like a large table. figcrop has a local-cell fallback for common cases, but this
  class of PDF may still need review.
- Outputs are intended for local research workflow use. Always inspect crops when
  building datasets or publications.

## License

Apache-2.0. See `LICENSE`.

Built on MinerU and OpenVINO. See `NOTICE` for attribution and dependency notes.
