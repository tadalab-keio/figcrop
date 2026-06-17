# Agent Guide

Use this repo to extract publication figures from PDFs. Keep changes small and
verify outputs visually: figure extraction is geometry-heavy and regressions are
often only visible in the saved JPEGs.

## Important Paths

- Main tool: `figtools.py`
- Local venv on the development machine: `.venv\Scripts\python.exe`
- Output manifest: `figures.json` in each extraction output directory
- Model cache/IR: `models/`

## Common Commands

Windows:

```powershell
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto figs=1,2
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto mode=panel
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto mode=caption
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto trim=whiteband
.\.venv\Scripts\python.exe figtools.py serve auto
```

Linux/macOS:

```bash
.venv/bin/python figtools.py extract paper.pdf out auto
.venv/bin/python figtools.py serve auto
```

HTTP server request:

```bash
curl -s -X POST http://127.0.0.1:8077/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf":"paper.pdf","out_dir":"out","figs":[1,2],"mode":"caption"}'
```

## Extraction Options

- `figs=1,2`: extract real figure numbers found from PDF text captions.
- `top=N`: fallback to the first N visual regions per page.
- `mode=figure`: default, whole figure body without caption.
- `mode=panel`: split each figure into `(a)` / `(b)` subpanels.
- `mode=caption`: whole figure body plus matched caption text.
- `trim=mask`: default fast trim mode.
- `trim=whiteband`: slower local whitespace snap around the detector bbox.
- `panels=true` and `caption=include` are legacy aliases for `mode=panel` and
  `mode=caption`.

## Verification

Always inspect JPEG outputs, not just command success. At minimum:

1. Open a montage or representative outputs with `view_image`.
2. Check for cut-off axes, table borders, panel labels, and caption text.
3. Check for neighbor figure/table frame lines, section headings, or unrelated
   fallback regions.
4. For `mode=caption`, check that the caption is complete and that nearby
   frame rules were not included.
5. Run `python -m py_compile figtools.py` after code changes.

Small one-shot extractions usually finish in a few seconds after model setup.
Use the server when repeatedly extracting from many PDFs or iterating on trim
logic.

## Implementation Notes

- Layout detection uses MinerU PP-DocLayoutV2. The OpenVINO IR has static
  800x800 input, so detector edge errors are handled by trimming, not by raising
  detector resolution.
- Numbering is local logic: PDF text captions are matched to nearby visual
  regions. It is not MinerU's full reading-order pipeline.
- Default output is one crop per `Fig.N`; `mode=panel` opts into `(a)` / `(b)`
  subpanel output.
- `mode=caption` first trims the figure body, then unions it with a tight
  caption text-ink rectangle from the PDF text block. Long horizontal/vertical
  caption-adjacent rules are ignored for the caption bbox.
- Very dense pages can still need `figs=`, `top=`, or manual review.

## Commit Attribution

If Codex makes commits in this repo, include:

```text
Co-authored-by: Codex <noreply@openai.com>
```

Do not add Codex attribution to commits that were authored by another agent or
by the user. If rewriting already-pushed history, create a backup branch first
and use `git push --force-with-lease`, never plain `--force`.
