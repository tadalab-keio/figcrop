# figcrop

Extract figures from research-paper PDFs — by real figure number, whole multi-panel
figures included. Built on [MinerU](https://github.com/opendatalab/MinerU)'s layout
model (PP-DocLayoutV2) + [OpenVINO](https://github.com/openvinotoolkit/openvino).

- **Detection**: OpenVINO (default, fast cold-start via compiled-blob cache, no per-process JIT)
  or torch (`xpu`/`cuda`). Same PP-DocLayoutV2 model either way.
- **Figure numbering & whole-figure grouping**: own geometry (detected boxes + the PDF
  text layer's `Fig.N` + caption→figure assignment + same-column band union). No dependency
  on MinerU's reading-order / full pipeline at runtime.
- **Persistent server**: load the model once, serve crops over HTTP (warm requests ~1–3 s).

## Why
Heuristic tools (e.g. pdffigures2) miss dense pages; running MinerU's full pipeline per call
reloads many models (tens of seconds). figcrop keeps just the layout model warm and adds
figure-number / whole-figure logic, so you can ask for "Fig.1 and Fig.2" and get clean,
correctly-numbered, complete figures.

## Requirements
- Python **3.10–3.13** (MinerU does not support 3.14). Windows tested.
- [uv](https://github.com/astral-sh/uv) (the setup script installs it if missing).
- Models (~hundreds MB) download from Hugging Face on first run.
- NPU is **not** supported (PyTorch has no NPU backend; via OpenVINO this RT-DETR model is
  ~20× slower than GPU). Use `auto`(GPU)/`xpu`/`cuda`/`CPU`.

## Setup
```powershell
# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1 -Device xpu   # cpu | cuda | xpu
```
```bash
# Linux/macOS (CPU/CUDA)
bash setup.sh
```
`requirements.txt` is just a manifest; nothing installs until setup runs `pip install`,
and it all goes into the project-local `.venv` (your system stays untouched).

## Usage
`<py>` = `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python`.

```bash
# persistent server (recommended; default auto -> OpenVINO GPU)
<py> figtools.py serve auto              # http://127.0.0.1:8077

# request crops
curl -s -X POST http://127.0.0.1:8077/extract -H "Content-Type: application/json" \
     -d '{"pdf":"paper.pdf","out_dir":"out","figs":[1,2]}'
#   figs=[1,2] -> whole Fig.1 & Fig.2 (all panels) by real number
#   top=2      -> first 2 figures per page (positional; for dense pages)

# one-shot CLI
<py> figtools.py extract paper.pdf out auto figs=1,2
```
Output: cropped JPEGs (`fig_pNN_FigN.jpg`) + `figures.json` manifest in `out_dir`.

Manual precise cropping helpers (`grid`/`render`/`find_tables`/`extract_figures`/…) can be
imported in any env for hand-tuning when detection misses.

## Limitations
Ultra-dense multi-figure pages (e.g. IEDM-style 9-figures-per-page) defeat the
caption→figure geometry; use `top=` or the manual helpers there.

## License
Apache-2.0 (see `LICENSE`). Built on MinerU (MinerU Open Source License) — see `NOTICE`.
For a research lab / local use this is unrestricted; MinerU's commercial-threshold and
online-service-attribution terms apply only at large scale / public services.
