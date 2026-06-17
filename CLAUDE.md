# figcrop Guide for Claude Code

This repository contains a single-file tool, `figtools.py`, for extracting
figures from research-paper PDFs by real `Fig.N` / `Table N` captions.

Prefer the existing implementation style. The core behavior is geometry-heavy,
so do not judge changes only by tests or command success: render crops and look
at the images.

## Quick Commands

Windows development environment:

```powershell
.\.venv\Scripts\python.exe figtools.py help
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto figs=1,2
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto caption=include
.\.venv\Scripts\python.exe figtools.py extract paper.pdf out auto trim=whiteband
.\.venv\Scripts\python.exe figtools.py serve auto
```

Server request:

```bash
curl -s -X POST http://127.0.0.1:8077/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf":"paper.pdf","out_dir":"out","figs":[1,2],"caption_mode":"include"}'
```

## Options

- `figs=1,2`: extract real figure numbers from PDF text captions.
- `top=N`: fallback to first N visual regions per page.
- `panels=true`: output detected regions/panels separately.
- `trim=mask`: default fast trim mode.
- `trim=whiteband`: slower local whitespace snap mode.
- `caption=include`: include matched caption text below the figure.

API fields are `pdf`, `out_dir`, `figs`, `top`, `panels`, `trim_mode`, and
`caption_mode`.

## Verification Workflow

After any code change:

```powershell
.\.venv\Scripts\python.exe -m py_compile figtools.py
```

For behavior changes, extract at least one dense PDF and inspect a montage or
representative JPEGs. Look for:

- clipped axes, table borders, panel labels, or captions
- neighbor frame-line bleed
- page furniture, section headings, or fallback regions mixed into crops
- incomplete captions when `caption=include`

Known useful local PDFs on the development machine include:

- `C:\pf2\ye_p3.pdf`
- `C:\pf2\ye_full.pdf`
- `C:\pf2\attn.pdf`
- `C:\pf2\t_isscc.pdf`
- `C:\pf2\t_jssc.pdf`
- `C:\pf2\t_ncomm.pdf`

## Implementation Notes

- MinerU PP-DocLayoutV2 detects `image`, `chart`, and `table` regions.
- OpenVINO IR input is static 800x800, so edge mistakes are handled in trim
  logic instead of increasing detector resolution.
- PDF text captions are read directly from `page.get_text("dict")`; figure
  numbering and region-to-caption assignment are local logic.
- Default output is one whole-figure crop per `Fig.N`. Use `panels=true` only
  when region-level output is desired.
- `caption=include` first trims the figure body, then unions it with a tight
  caption text-ink rectangle from the same PDF text block. Long caption-adjacent
  horizontal/vertical rules are ignored for the caption bbox.
- Always preserve user changes in the worktree. Avoid unrelated refactors.

## Commit Attribution

If Claude Code makes commits, keep its normal attribution conventions. If Codex
made the commit, include:

```text
Co-authored-by: Codex <noreply@openai.com>
```

Do not add Codex attribution to Claude/user commits. If rewriting pushed
history, create a backup branch first, verify the tree diff is empty, and use
`git push --force-with-lease`, not plain `--force`.
