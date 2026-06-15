# -*- coding: utf-8 -*-
"""figcrop / figtools = ①手動精密クロップ ②レイアウト自動抽出 ③常駐サーバ を1ファイルに。
論文PDFから図を抽出。検出は OpenVINO(既定)/torch、キャプション対応と図全体の束ね(帯)は自前幾何。

セットアップ: setup.ps1 (Windows) / setup.sh。以下 <py> = プロジェクトの venv python（.venv/Scripts/python.exe 等）。
- **②抽出CLI**:  <py> figtools.py extract <pdf> <out_dir> [auto|GPU|NPU|xpu|cuda] [figs=1,2]
- **③常駐サーバ**: <py> figtools.py serve auto    （既定 auto→OpenVINO GPU・127.0.0.1:8077）
      POST /extract {pdf,out_dir,figs?,top?}  →  figs=[1,2]=実Fig番号で図全体 / top=2=上から2図
- **①手動クロップ**（grid/render/borders/find系/extract_figures/gaps/vlm_figures）は import して使用:
    import importlib.util,sys; sys.dont_write_bytecode=True
    z=importlib.util.module_from_spec(importlib.util.spec_from_file_location("z","figtools.py")); ...
重い import は全て関数内＝両環境で安全に import 可。検出モデルは MinerU の PP-DocLayoutV2（初回 IR 自動生成）。
Built on MinerU (https://github.com/opendatalab/MinerU) + OpenVINO。詳細は README.md。
"""
import sys, os, json, io, re, fitz
from collections import defaultdict, deque

try:                                            # Windows コンソール(cp932)でも µ 等を出せるように
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ② レイアウト抽出の定数
FIG_LABELS = {"image", "chart", "table"}        # 切り出す対象（視覚的な図）
CAP_LABELS = {"figure_title"}                   # キャプション行
DET_DPI = 150                                   # 検出用レンダリング解像度
CROP_DPI = 300                                  # 切り出し解像度（原寸）
EXPAND_PT = 3.0                                  # 検出枠の数px内入りを救う微小拡張（隣図の孤立線は_trim_boxで除去）
PAD_PX = 6                                       # 余白トリム後に残す均一マージン(px)。0で完全タイト
TRIM_THRESH = 12                                # 余白トリムの白判定しきい値（これ未満の濃さ＝白＝余白）
EDGE_LINE_FRAC = 0.60                           # 隣図の枠線扱いする最小投影長（文字ストローク誤削除を避ける）
CAP_RE = re.compile(r"(?i)^\s*(fig(?:ure)?|table)\.?\s*(\d+)")   # 「Fig 3」始まりのみ＝本文/見出し誤検出を排除
# PP-DocLayoutV2 のクラス（label_id→名前。OpenVINO版の後処理で使用）
PP_LABELS = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
             "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
             "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
             "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]
OV_IR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "layout.xml")  # IR保存先(プロジェクト相対・初回torchから自動生成)


# ========================= ① 手動精密クロップ（主環境）=========================
_CAP_RE = re.compile(r'^(fig|figure|table)\.?\s*\d', re.I)


def _caption_lines(page):
    words = page.get_text('words')
    lines = defaultdict(list)
    for w in words:
        lines[(w[5], w[6])].append(w)
    rects = []
    keys = set()
    for k, ws in lines.items():
        ws = sorted(ws, key=lambda w: w[0])
        text = ' '.join(w[4] for w in ws)
        if _CAP_RE.match(text) or ws[0][4].rstrip('.').lower() in ('figure', 'fig', 'table'):
            keys.add(k)
            rects.append(fitz.Rect(min(w[0] for w in ws), min(w[1] for w in ws),
                                   max(w[2] for w in ws), max(w[3] for w in ws)))
    return keys, rects


def extract_figures(page, gap=10, min_graphic_area=3000):
    g_rects = [fitz.Rect(r) for r in page.cluster_drawings(x_tolerance=gap, y_tolerance=gap)]
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            g_rects.append(fitz.Rect(r))
    g_rects = [r for r in g_rects if r.width > 1 and r.height > 1 and r.width < page.rect.width]

    cap_keys, cap_rects = _caption_lines(page)
    words = page.get_text('words')
    w_rects = [fitz.Rect(w[:4]) for w in words if (w[5], w[6]) not in cap_keys]

    nodes = [(r, True) for r in g_rects] + [(r, False) for r in w_rects]
    n = len(nodes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rects = [r for r, _ in nodes]
    isg = [g for _, g in nodes]
    for i in range(n):
        for j in range(i + 1, n):
            if (rects[i] + (-gap, -gap, gap, gap)).intersects(rects[j]):
                union(i, j)

    comp_box = {}
    comp_garea = defaultdict(float)
    for i in range(n):
        r = find(i)
        comp_box[r] = (comp_box[r] | rects[i]) if r in comp_box else fitz.Rect(rects[i])
        if isg[i]:
            comp_garea[r] += rects[i].width * rects[i].height

    figs = [(comp_box[r], comp_garea[r]) for r in comp_box if comp_garea[r] >= min_graphic_area]
    clamped = []
    for b, a in figs:
        below = [c.y0 for c in cap_rects if c.y0 >= b.y0 + 0.3 * b.height and c.y0 <= b.y1 + 6
                 and c.x0 < b.x1 and c.x1 > b.x0]
        if below:
            b = fitz.Rect(b.x0, b.y0, b.x1, min(b.y1, min(below) - 2))
        clamped.append(((b & page.rect), a))
    clamped.sort(key=lambda t: -t[1])
    return clamped


def _page(src, page=0):
    """src が str(PDFパス)なら開いて page を返す。Page オブジェクトならそのまま返す。"""
    if isinstance(src, str):
        return fitz.open(src)[page]
    return src


def render(src, page_or_bbox, bbox_or_out, out_or_dpi=None, dpi=300):
    """切り出し描画。2通りの呼び方:
      render(page_obj, bbox, out, dpi=300)
      render(pdf_path, page_no, (x0,y0,x1,y1), out, dpi=300)
    """
    if isinstance(src, str):
        pg = fitz.open(src)[page_or_bbox]
        bbox, out = bbox_or_out, out_or_dpi
    else:
        pg = src
        bbox, out = page_or_bbox, bbox_or_out
    pg.get_pixmap(dpi=dpi, clip=fitz.Rect(bbox)).save(out)
    return out


def borders(src, page=0, region=None, min_w=80, min_h=40):
    """**矩形ストローク（枠・テーブル罫線の囲み）を面積降順で返す**: [(x0,y0,x1,y1,w,h),...]。
    figure/table の外枠を PDF ベクタから厳密座標で拾い、その枠ぴったり（線を含め±1pt外側）で切る用。
    画像認識やグリッド目視より正確。枠が密接に入れ子（図自身の枠＋まとめ枠）でも面積順で区別できる。
    ※枠がラスタに焼かれている場合はベクタに出ないので、その時だけ opencv 等が必要。"""
    pg = _page(src, page)
    clip = fitz.Rect(*region) if region else pg.rect
    seen, out = set(), []
    for d in pg.get_drawings():
        r = fitz.Rect(d['rect'])
        if r.width >= min_w and r.height >= min_h and r.intersects(clip):
            key = (round(r.x0), round(r.y0), round(r.x1), round(r.y1))
            if key in seen:
                continue
            seen.add(key)
            out.append((round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1),
                        round(r.width, 1), round(r.height, 1)))
    out.sort(key=lambda t: -(t[4] * t[5]))
    return out


def grid(src, page=0, dpi=110, step=50, region=None, out=None):
    """ページ(or region)を **座標グリッド付き**で描画して out に保存。
    赤=x(pt)縦線・青=y(pt)横線を step pt 刻みでラベル付き。これを Read すれば
    図/サブパネルの bbox を pt 単位で目視で読める（所在特定＆精密クロップの主役）。"""
    import math
    from PIL import Image, ImageDraw
    pg = _page(src, page)
    clip = fitz.Rect(*region) if region else pg.rect
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = ImageDraw.Draw(img)
    s = dpi / 72.0
    x0, y0 = clip.x0, clip.y0
    for xpt in range(int(math.floor(x0 / step) * step), int(clip.x1) + 1, step):
        px = int(round((xpt - x0) * s))
        if 0 <= px < img.width:
            d.line([(px, 0), (px, img.height)], fill=(230, 60, 60), width=1)
            d.text((px + 2, 2), str(xpt), fill=(190, 0, 0))
    for ypt in range(int(math.floor(y0 / step) * step), int(clip.y1) + 1, step):
        py = int(round((ypt - y0) * s))
        if 0 <= py < img.height:
            d.line([(0, py), (img.width, py)], fill=(60, 60, 230), width=1)
            d.text((2, py + 1), str(ypt), fill=(0, 0, 190))
    if out is None:
        out = "grid.jpg"
    img.save(out, quality=85)
    return out


def gaps(src, page=0, region=None, axis="y", dpi=150, min_gap=4, white=245):
    """region 内の **空白帯（パネル境界）** を投影法で検出し、pt 単位の (start,end,幅) リストを返す。
    axis='y' で水平帯（行の隙間）、'x' で垂直帯（列の隙間）。ラスタ/密図のパネル分割の目安に。"""
    from PIL import Image
    pg = _page(src, page)
    clip = fitz.Rect(*region) if region else pg.rect
    pix = pg.get_pixmap(dpi=dpi, clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    W, H = img.size
    px = img.load()
    s = dpi / 72.0
    if axis == "y":
        occ = [any(px[x, y] < white for x in range(0, W, 2)) for y in range(H)]
    else:
        occ = [any(px[x, y] < white for y in range(0, H, 2)) for x in range(W)]
    runs = []
    st = None
    for i, v in enumerate(occ):
        if not v and st is None:
            st = i
        elif v and st is not None:
            runs.append((st, i - 1))
            st = None
    if st is not None:
        runs.append((st, len(occ) - 1))
    base = clip.y0 if axis == "y" else clip.x0
    out = [(round(base + a / s, 1), round(base + b / s, 1), round((b - a) / s, 1))
           for a, b in runs if (b - a) / s >= min_gap]
    return out


def vlm_figures(src, page, out_dir=None, dpi=300, also_tables=True, device="auto"):
    """**密ページ（図だけが密に並び grid/連結成分では1塊になる）専用の自動図分割**。
    Docling の VLM パイプライン（GraniteDocling 視覚モデル・258M）でページを画像として解釈し、
    図/表を個別領域に分割。`[(bbox(pt,TOPLEFT), kind, caption_or_None), ...]` を返す。
    out_dir 指定時は各領域を dpi で fig{n}.jpg に切り出す（**最後は必ず Read で目視**）。

    device: "auto"（既定＝XPU/Intel Arc GPU が使えれば XPU、無ければ CPU）/ "xpu" / "cuda" / "cpu"。
      GraniteDocling は XPU 正式対応。**XPU は速度のため＝分割の精度はモデル依存で CPU と同じ**。

    注意:
    - **CPU だと1ページ数分と遅い**（XPU で短縮）。普通の本文+図ページは grid/borders/find_tables の方が速く確実。
      「全面1塊になって手に負えない密ページ」専用の最後の手段。
    - 分割に粗さが残る（隣接図を束ねる/1図を数片に割る）ことがある＝出力 bbox を grid で微調整してよい。
    - 要 `pip install docling`（導入済 2.102.x）。実測: IEDM 密ページを 11 領域に分離（2026-06-14）。
    """
    import fitz
    import torch
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.vlm_model_specs import GRANITEDOCLING_TRANSFORMERS
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_core.types.doc import PictureItem, TableItem, CoordOrigin

    if device == "auto":
        device = "xpu" if (getattr(torch, "xpu", None) and torch.xpu.is_available()) else "cpu"
    dev = {"xpu": AcceleratorDevice.XPU, "cuda": AcceleratorDevice.CUDA,
           "cpu": AcceleratorDevice.CPU}[device]
    opts = VlmPipelineOptions(vlm_options=GRANITEDOCLING_TRANSFORMERS,
                              accelerator_options=AcceleratorOptions(device=dev))

    ph = fitz.open(src)[page].rect.height
    conv = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)})
    # 注意: 本ツールは page を fitz 流の 0 始まりで受けるが、docling の page_range は 1 始まり。
    doc = conv.convert(source=src, page_range=(page + 1, page + 1)).document

    kinds = (PictureItem, TableItem) if also_tables else (PictureItem,)
    out = []
    for el, _lvl in doc.iterate_items():
        if not isinstance(el, kinds) or not getattr(el, "prov", None):
            continue
        bb = el.prov[0].bbox
        if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
            bb = bb.to_top_left_origin(ph)
        rect = (round(bb.l, 1), round(bb.t, 1), round(bb.r, 1), round(bb.b, 1))
        try:
            cap = el.caption_text(doc) or None
        except Exception:
            cap = None
        out.append((rect, "table" if isinstance(el, TableItem) else "picture", cap))

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        pg = fitz.open(src)[page]
        for i, (r, _k, _c) in enumerate(out):
            pg.get_pixmap(dpi=dpi, clip=fitz.Rect(*r)).save(os.path.join(out_dir, f"fig{i + 1}.jpg"))
    return out


# ========================= ② レイアウト自動抽出（mineru-env）=========================
def _load_model(device):
    """torch 版レイアウト検出（cpu/cuda/xpu）。.predict(pil)→[{label,score,bbox,index}]。"""
    import torch
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from mineru.utils.enum_class import ModelPath
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    if device == "auto":
        device = "xpu" if (getattr(torch, "xpu", None) and torch.xpu.is_available()) else "cpu"
    weight = os.path.join(auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2),
                          ModelPath.pp_doclayout_v2)
    m = PPDocLayoutV2LayoutModel(weight, device=device)
    half = (device == "xpu")
    if half:                                   # XPU は fp16 で XMX を使い約1.5倍速・精度同等
        m.model.half()
        o = m._preprocess_single_image
        m._preprocess_single_image = lambda im, _o=o: (lambda pv, ts: (pv.half(), ts))(*_o(im))
    return m, device


def _export_ov_ir(out_xml=OV_IR):
    """torch の PP-DocLayoutV2 を OpenVINO IR に変換して保存（初回のみ・要 torch+mineru）。
    出力は (logits, pred_boxes) だけのラッパー＝reading-order は捨てる（キャプションは自前幾何で対応）。"""
    import torch, openvino as ov
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from mineru.utils.enum_class import ModelPath
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    weight = os.path.join(auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2),
                          ModelPath.pp_doclayout_v2)
    m = PPDocLayoutV2LayoutModel(weight, device="cpu")

    class _W(torch.nn.Module):
        def __init__(s, mdl):
            super().__init__(); s.m = mdl.eval()

        def forward(s, pixel_values):
            o = s.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    ovm = ov.convert_model(_W(m.model), example_input=torch.randn(1, 3, 800, 800))
    ovm.reshape([1, 3, 800, 800])                 # 静的形状（NPU必須・GPU最適化）
    os.makedirs(os.path.dirname(out_xml), exist_ok=True)
    ov.save_model(ovm, out_xml)
    return out_xml


class OVLayout:
    """OpenVINO 版レイアウト検出（**torch非依存・起動はIR+キャッシュで数秒・推論~33ms**）。
    .predict(pil)→[{label,score,bbox,index}]（torch版と同形式）。RT-DETR 後処理を自前で復元。"""
    IMGSZ = (800, 800)
    CONF = 0.45

    def __init__(self, device="GPU"):
        import openvino as ov
        if not os.path.exists(OV_IR):
            _export_ov_ir(OV_IR)                  # 初回だけ torch から変換
        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.join(os.path.dirname(OV_IR), "cache")})  # device コンパイルをキャッシュ
        self.device = device
        self.cm = core.compile_model(OV_IR, device)

    def predict(self, pil):
        import numpy as np
        from PIL import Image
        W, H = pil.size
        im = pil.convert("RGB").resize(self.IMGSZ, Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None]) / 255.0
        a, b = self.cm(x).to_tuple()[:2]
        logits, boxes = (b[0], a[0]) if a.shape[-1] == 4 else (a[0], b[0])   # (Q,C),(Q,4)
        c, d = boxes[:, :2], boxes[:, 2:]
        xyxy = np.concatenate([c - 0.5 * d, c + 0.5 * d], -1) * np.array([W, H, W, H], np.float32)
        scores = 1.0 / (1.0 + np.exp(-logits))    # sigmoid [Q,C]
        Q, C = scores.shape
        flat = scores.ravel()
        idx = np.argpartition(-flat, Q - 1)[:Q]   # 上位 Q（torch の topk(num_top_queries) 相当）
        res = []
        for j in idx:
            sc = float(flat[j])
            if sc < self.CONF:
                continue
            q, lab = int(j // C), int(j % C)
            x0, y0, x1, y1 = xyxy[q]
            x0, x1 = max(0.0, min(W, x0)), max(0.0, min(W, x1))
            y0, y1 = max(0.0, min(H, y0)), max(0.0, min(H, y1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            res.append({"label": PP_LABELS[lab] if lab < len(PP_LABELS) else str(lab),
                        "score": round(sc, 4), "index": 0,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)]})   # np.float32→pythonでJSON可
        return res


def _engine(device):
    """device 名で torch/OpenVINO を振り分け (predictor, 表示名) を返す。
    OpenVINO（起動キャッシュで速い・推論~33ms・torch非依存）: 'auto'|'ov'|'GPU'|'NPU'|'CPU'（auto/ov→GPU）。
    torch（fp16等）: 'xpu'|'cuda'。"""
    if device in ("auto", "ov"):
        return OVLayout("GPU"), "ov:GPU"
    if device.upper() in ("GPU", "NPU", "CPU"):
        return OVLayout(device.upper()), f"ov:{device.upper()}"
    return _load_model(device)                    # torch (xpu/cuda)


_PANEL_RE = re.compile(r"^\(?([a-h])\)?$")        # (a) / a) / a 等のパネル記号（括弧なし裸文字も）。誤検出は _panel_of の位置判定で抑制

def _line_boxes(page):
    """PDFテキスト層の各行を (text, bbox_pt) で返す（キャプション/パネル記号の検出に使う）。"""
    out = []
    for b in page.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            sp = ln.get("spans", [])
            if not sp:
                continue
            txt = "".join(s["text"] for s in sp).strip()
            bb = (min(s["bbox"][0] for s in sp), min(s["bbox"][1] for s in sp),
                  max(s["bbox"][2] for s in sp), max(s["bbox"][3] for s in sp))
            out.append((txt, bb))
    return out


def _text_captions(page):
    """**PDFテキスト層から直接** 「Fig.N / Table N で始まる行」を拾う（モデルの figure_title 検出漏れに非依存）。
    返り値 [(bbox_pt, 'Fig2', 2), ...]。bbox は PDF point。"""
    out = []
    for txt, bb in _line_boxes(page):
        mm = CAP_RE.match(txt)
        if mm:
            kind = "Table" if mm.group(1).lower().startswith("tab") else "Fig"
            out.append((bb, f"{kind}{int(mm.group(2))}", int(mm.group(2))))
    return out


def _panel_labels(page):
    """テキスト層から (a)..(h) のパネル記号を位置つきで拾う。返り値 [(bbox_pt, 'a'), ...]。"""
    out = []
    for txt, bb in _line_boxes(page):
        mm = _PANEL_RE.match(txt)
        if mm:
            out.append((bb, mm.group(1)))
    return out


def _panel_of(region_bb, panels):
    """図領域の左上付近にあるパネル記号を返す（'a' 等）。無ければ None。"""
    fx0, fy0, fx1, fy1 = region_bb
    fw, fh = max(1.0, fx1 - fx0), max(1.0, fy1 - fy0)
    best, bestd = None, 1e9
    for (px0, py0, px1, py1), lab in panels:
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        if fx0 - 6 <= cx <= fx0 + 0.55 * fw and fy0 - 6 <= cy <= fy0 + 0.35 * fh:  # 領域の左上域
            d = (cx - fx0) + (cy - fy0)
            if d < bestd:
                best, bestd = lab, d
    return best


def _assign_bands(region_bbs, caps, page_width=None):
    """各図領域を「同カラムで直下にある最も近いキャプション」の Fig 番号に割当てる（union しない）。
    多パネル図は同じ番号を共有（領域ごとに別ファイルで出す）。番号はキャプション側が持つので、
    そのページが図5始まりでも実番号が付く。返り値 {region_index: num}。"""
    cs = sorted(caps, key=lambda c: c[0][1])
    single_caption_page = page_width is not None and len(cs) == 1
    row_unique = []
    for cbb, _l, _num in cs:
        cy0 = cbb[1]
        row_unique.append(sum(1 for obb, _ol, _on in cs if abs(obb[1] - cy0) < 50) == 1)
    assign = {}
    for i, (fx0, fy0, fx1, fy1) in enumerate(region_bbs):
        fw = max(1.0, fx1 - fx0)
        best, bestgap = None, 1e9
        for ci, (cbb, _l, num) in enumerate(cs):
            cx0, cy0, cx1, cy1 = cbb
            if single_caption_page:
                cx0, cx1 = 0.0, page_width
            ov = min(fx1, cx1) - max(fx0, cx0)
            gap = cy0 - fy1                             # 図の下端→キャプション上端
            same_col = ov > 0.2 * min(fw, cx1 - cx0)
            same_row_caption = row_unique[ci] and -8 <= gap <= 24
            if not same_col and not same_row_caption:
                continue
            if -8 <= gap < bestgap:                     # 直下で最も近いキャプション＝その図の番号
                best, bestgap = num, gap
        if best is not None and bestgap < 350:          # 遠すぎる対応は捨てる
            assign[i] = best
    return assign


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bool_run_count(mask):
    total = 0
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        total += 1
        while i < n and mask[i]:
            i += 1
    return total


def _bool_run_bounds(mask, offset=0):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        runs.append((offset + start, offset + i))
    return runs


def _trim_box(img, ignore_mask=None, return_edges=False):
    """Return the content box, trimming white margin and detached neighbor-frame lines.

    The ordinary white trim uses the ink projection. A second pass only peels off
    long, thin edge strokes when they are detached from the figure body, so table
    borders, axes, and edge labels stay intact.
    """
    import numpy as np
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    ink = gray < (255 - TRIM_THRESH)
    dark = gray < 80
    ignore = None
    if ignore_mask is not None:
        ignore = np.asarray(ignore_mask, dtype=bool)
        if ignore.shape == ink.shape:
            ink &= ~ignore
            dark &= ~ignore
        else:
            ignore = None
    colc, rowc = ink.sum(axis=0), ink.sum(axis=1)
    dark_colc, dark_rowc = dark.sum(axis=0), dark.sum(axis=1)

    def axis_size(axis):
        return (W, H) if axis == "x" else (H, W)

    def axis_dark_counts(axis):
        return dark_colc if axis == "x" else dark_rowc

    def edge_component_is_long_line(axis, start, end):
        # Flood fill is only used after projection tests say "maybe a frame".
        if axis == "x":
            seeds = np.argwhere(ink[:, start:end])
            if seeds.size == 0:
                return False
            seeds[:, 1] += start
        else:
            seeds = np.argwhere(ink[start:end, :])
            if seeds.size == 0:
                return False
            seeds[:, 0] += start

        seen = np.zeros_like(ink, dtype=bool)
        q = deque((int(y), int(x)) for y, x in seeds)
        miny, maxy, minx, maxx = H, -1, W, -1
        while q:
            y, x = q.pop()
            if y < 0 or y >= H or x < 0 or x >= W or seen[y, x] or not ink[y, x]:
                continue
            seen[y, x] = True
            miny, maxy = min(miny, y), max(maxy, y)
            minx, maxx = min(minx, x), max(maxx, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        q.append((y + dy, x + dx))
        if maxy < miny:
            return False
        bw, bh = maxx - minx + 1, maxy - miny + 1
        axis_len, _span = axis_size(axis)
        strip = max(3, int(axis_len * 0.06))
        return (bw <= strip and bh >= EDGE_LINE_FRAC * H) if axis == "x" else \
               (bh <= strip and bw >= EDGE_LINE_FRAC * W)

    def has_body_connections(axis, start, end, side):
        depth = 4
        if axis == "x":
            c0, c1 = (end, min(W, end + depth)) if side == "lo" else (max(0, start - depth), start)
            if c0 >= c1:
                return False
            inner_strip = ink[:, c0:c1]
            dark_cand = dark[:, start:end].any(axis=1)
            dark_inner = dark[:, c0:c1].any(axis=1)
        else:
            r0, r1 = (end, min(H, end + depth)) if side == "lo" else (max(0, start - depth), start)
            if r0 >= r1:
                return False
            inner_strip = ink[r0:r1, :]
            dark_cand = dark[start:end, :].any(axis=0)
            dark_inner = dark[r0:r1, :].any(axis=0)
        filled_body = int(inner_strip.sum()) >= 0.5 * inner_strip.size
        return filled_body or _bool_run_count(dark_cand & dark_inner) >= 2

    def peel_edge_lines(lo, hi, axis, side):
        """Peel detached long edge lines, but stop at connected axes/table borders."""
        axis_len, span = axis_size(axis)
        counts = axis_dark_counts(axis)
        cur_lo, cur_hi = lo, hi
        changed = False

        def outside_sparse(run, base_lo, base_hi):
            start, end = run
            outside = int(counts[base_lo:start].sum()) if side == "lo" else int(counts[end:base_hi].sum())
            dist = max(1, (start - base_lo) if side == "lo" else (base_hi - end))
            return outside < max(20, int(0.005 * span * dist))

        while cur_hi - cur_lo > 4:
            scan = min(cur_hi - cur_lo, max(36, min(96, int(axis_len * 0.08))))
            if side == "lo":
                s0, s1 = cur_lo, min(cur_hi, cur_lo + scan)
            else:
                s0, s1 = max(cur_lo, cur_hi - scan), cur_hi
            runs = _bool_run_bounds(counts[s0:s1] >= EDGE_LINE_FRAC * span, s0)
            if not runs:
                break
            start, end = runs[0] if side == "lo" else runs[-1]
            connected = has_body_connections(axis, start, end, side)
            if connected:
                if outside_sparse((start, end), cur_lo, cur_hi):
                    if side == "lo":
                        cur_lo = start
                    else:
                        cur_hi = end
                    changed = True
                break
            if side == "lo":
                cur_lo = end
            else:
                cur_hi = start
            changed = True
        return (cur_lo if side == "lo" else cur_hi), changed

    def trim_axis(counts, axis):
        axis_len, span = axis_size(axis)
        nz = np.flatnonzero(counts)
        if nz.size == 0:
            return 0, axis_len, False, False
        lo, hi = int(nz[0]), int(nz[-1]) + 1
        cut_lo = cut_hi = False
        substantive = counts > max(2, 0.05 * span)
        strip = max(3, int(axis_len * 0.06))
        inner_win = max(16, PAD_PX * 4)

        def has_near_inner_line(start, end, side):
            if side == "lo":
                c0, c1 = end, min(hi, end + inner_win)
            else:
                c0, c1 = max(lo, start - inner_win), start
            return c0 < c1 and counts[c0:c1].max(initial=0) >= EDGE_LINE_FRAC * span

        def is_frame_line(start, end, side):
            return (1 <= (end - start) <= strip
                    and counts[start:end].max(initial=0) >= EDGE_LINE_FRAC * span
                    and (edge_component_is_long_line(axis, start, end)
                         or has_near_inner_line(start, end, side)
                         or not has_body_connections(axis, start, end, side)))

        j = lo
        while j < hi and substantive[j]:
            j += 1
        k = j
        while k < hi and not substantive[k]:
            k += 1
        if is_frame_line(lo, j, "lo") and (k - j) >= 2 and k < hi:
            lo = j
            cut_lo = True

        j = hi - 1
        while j >= lo and substantive[j]:
            j -= 1
        k = j
        while k >= lo and not substantive[k]:
            k -= 1
        if is_frame_line(j + 1, hi, "hi") and (j - k) >= 2 and (k + 1) > lo:
            hi = j + 1
            cut_hi = True

        nlo, snapped = peel_edge_lines(lo, hi, axis, "lo")
        if snapped:
            lo = nlo
            cut_lo = True
        nhi, snapped = peel_edge_lines(lo, hi, axis, "hi")
        if snapped:
            hi = nhi
            cut_hi = True
        return lo, hi, cut_lo, cut_hi

    x0, x1, cut_l, cut_r = trim_axis(colc, "x")
    y0, y1, cut_t, cut_b = trim_axis(rowc, "y")
    if x1 - x0 < 4 or y1 - y0 < 4:
        box = (0, 0, W, H)
        edges = (False, False, False, False)
    else:
        box = (x0, y0, x1, y1)
        edges = (cut_l, cut_t, cut_r, cut_b)
    return (box, edges) if return_edges else box


def _caption_gap_bottom_px(gray, left, right, height, cbb, scale):
    """Caption直前の白い水平帯の下端(px)。テキストbboxではなく描画結果で決める。"""
    import numpy as np
    _cx0, cy0, _cx1, _cy1 = cbb
    y0 = max(0, round((cy0 - 5.0) * scale))
    y1 = min(height, round((cy0 + 2.0) * scale))
    if right <= left or y1 <= y0:
        return round(cy0 * scale)
    rows = (gray[y0:y1, left:right] < 245).sum(axis=1)
    max_ink = max(5, int((right - left) * 0.001))
    white_rows = np.flatnonzero(rows <= max_ink)
    return y0 + int(white_rows[-1]) + 1 if white_rows.size else round(cy0 * scale)


def _caption_guard_bottom_px(full_img, gray_ref, crop_box, fig_bb, cap_bbs, scale):
    """Return a bottom crop guard at the white band immediately above the caption."""
    if not cap_bbs:
        return None
    left, _top, right, _bottom = crop_box
    _width, height = full_img.size
    bx0, by0, bx1, by1 = fig_bb
    fig_w = max(1.0, bx1 - bx0)
    guard = None

    for cbb in cap_bbs:
        cx0, cy0, cx1, _cy1 = cbb
        cap_w = max(1.0, cx1 - cx0)
        overlap = min(bx1, cx1) - max(bx0, cx0)
        if overlap <= 0.2 * min(fig_w, cap_w):
            continue
        if cy0 + 2.0 <= by0 or cy0 - 5.0 > by1 + EXPAND_PT + 2.0:
            continue
        if gray_ref[0] is None:
            import numpy as np
            gray_ref[0] = np.asarray(full_img.convert("L"))
        gap_bottom = _caption_gap_bottom_px(gray_ref[0], left, right, height, cbb, scale)
        gap_bottom_pt = gap_bottom / scale
        if by0 < gap_bottom_pt <= by1 + EXPAND_PT + 2.0:
            candidate = max(crop_box[1] + 1, gap_bottom)
            guard = candidate if guard is None else min(guard, candidate)
    return guard


def _rect_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _mask_rect(mask, rect, crop_box, scale, pad=1):
    left, top, _right, _bottom = crop_box
    H, W = mask.shape
    x0 = max(0, round(rect[0] * scale) - left - pad)
    y0 = max(0, round(rect[1] * scale) - top - pad)
    x1 = min(W, round(rect[2] * scale) - left + pad)
    y1 = min(H, round(rect[3] * scale) - top + pad)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def _trim_ignore_mask(shape, crop_box_px, fig_bb, text_lines, drawing_bbs, scale):
    """Mask PDF page furniture for trim decisions only; never alters saved pixels."""
    import numpy as np
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    left, top, right, bottom = crop_box_px
    crop_pt = (left / scale, top / scale, right / scale, bottom / scale)
    fx0, fy0, fx1, fy1 = fig_bb
    fig_w, fig_h = max(1.0, fx1 - fx0), max(1.0, fy1 - fy0)

    def outside_figure(rect, tol=0.6):
        rx0, ry0, rx1, ry1 = rect
        cx, cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        return not (fx0 - tol <= cx <= fx1 + tol and fy0 - tol <= cy <= fy1 + tol)

    for _txt, rect in text_lines:
        if not CAP_RE.match(_txt):
            continue
        if _rect_overlap(rect, crop_pt) <= 0 or not outside_figure(rect):
            continue
        _mask_rect(mask, rect, crop_box_px, scale, pad=2)

    fig_area = fig_w * fig_h
    for item in drawing_bbs:
        rect, fill, color, width = item
        if _rect_overlap(rect, crop_pt) <= 0:
            continue
        rx0, ry0, rx1, ry1 = rect
        rw, rh = rx1 - rx0, ry1 - ry0
        area = max(0.1, rw * rh)
        overlap_ratio = _rect_overlap(rect, fig_bb) / min(area, fig_area)
        if overlap_ratio > 0.20:
            continue
        cy = (ry0 + ry1) / 2.0
        outside_vert = cy < fy0 or cy > fy1
        dark_fill = fill is not None and sum(fill[:3]) / 3.0 < 0.90
        thin_rule = fill is None and (width or 0.0) <= 1.5 and rh <= 3.0
        long_hline = rw >= 0.35 * fig_w and thin_rule
        long_bar = rw >= 0.35 * fig_w and rh <= max(14.0, 0.08 * fig_h) and dark_fill
        if outside_vert and (long_hline or long_bar):
            _mask_rect(mask, rect, crop_box_px, scale, pad=2)
    if W and H:
        mask[mask.sum(axis=1) >= 0.70 * W, :] = True
        mask[:, mask.sum(axis=0) >= 0.70 * H] = True
    return mask


def _whole_figures(region_bbs, assign):
    """Fig.N ごとに **図本体**の bbox を返す {num: bbox}＝所属領域(image/chart/table)の外接矩形。
    矩形なので領域の範囲内にある図中テキスト(プロセス説明文・パネル記号等)は入るが、
    **キャプション行『Fig.N: …』は union に含めない**ので巻き込まない（図だけ欲しい用途）。"""
    out = {}
    for i, num in assign.items():
        out[num] = region_bbs[i] if num not in out else _union(out[num], region_bbs[i])
    return out


def extract(pdf_path, out_dir, device="auto", model=None, figs=None, top=None, panels=False):
    """論文PDFから図を切り出す。**既定＝図ごとに一括（全パネル＋間のテキスト込みで1枚）**。
      （既定 None,None,False）… 各 Fig.N を1枚に（所属領域の外接矩形）。番号不明領域は x##。
      panels=True … a/b/c… パネル単位に分割（各領域を個別切出・パネル記号で命名）。
      figs=[1,2]  … その Fig 番号だけ（一括/分割どちらにも効く）。Fig5 始まりページでも実番号で当たる。
      top=2       … 各ページ上から2領域（位置基準・密ページ用フォールバック）。
    番号は「同カラム直下の最寄りキャプションの Fig.N」を各領域へ割当（PDFテキスト層・MinerU非依存）。"""
    import fitz
    m, dev = model if model is not None else _engine(device)       # model=(m,dev) を渡せば常駐再利用
    os.makedirs(out_dir, exist_ok=True)
    want = set(figs) if figs else None
    doc = fitz.open(pdf_path)
    s = 72.0 / DET_DPI                          # 検出画素 -> PDF point
    cs = CROP_DPI / 72.0                         # PDF point -> 切出画素
    import PIL.Image as Image
    manifest = []
    for pno in range(len(doc)):
        page = doc[pno]
        pix = page.get_pixmap(dpi=DET_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        preds = m.predict(img)
        caps = _text_captions(page)                                # テキスト層からFig.N行を直接(検出漏れに強い)
        text_lines = _line_boxes(page)
        plabels = _panel_labels(page)                              # (a)(b)… パネル記号
        regions = [d for d in preds if d["label"] in FIG_LABELS]
        regions.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))   # 上→下, 左→右
        if not regions:
            continue
        region_bbs = [tuple(x * s for x in d["bbox"]) for d in regions]
        assign = _assign_bands(region_bbs, caps, page.rect.width)   # 各領域→所属Fig番号
        cap_map = {}
        for cbb, _line, cnum in caps:
            cap_map.setdefault(cnum, []).append(cbb)
        drawing_bbs = []
        try:
            for dr in page.get_drawings():
                r = dr.get("rect")
                if r and not r.is_empty:
                    drawing_bbs.append(((r.x0, r.y0, r.x1, r.y1),
                                        dr.get("fill"), dr.get("color"), dr.get("width")))
        except Exception:
            pass
        cpix = [None, None]                                        # [pixmap, PIL] 遅延描画用
        cap_gray = [None]                                          # caption実インク位置検出用

        def _crop(bb, fn, cap_bbs=(), **extra):                    # 検出枠→余白&孤立縁トリム→均一マージン
            if cpix[0] is None:
                cpix[0] = page.get_pixmap(dpi=CROP_DPI)
                cpix[1] = Image.frombytes("RGB", (cpix[0].width, cpix[0].height), cpix[0].samples)
            full = cpix[1]; W, H = full.size
            m = round(EXPAND_PT * cs)
            x0, y0, x1, y1 = (round(v * cs) for v in bb)
            left, top = max(0, x0 - m), max(0, y0)
            right, bottom = min(W, x1 + m), min(H, y1 + m)
            guard_bottom = _caption_guard_bottom_px(
                full, cap_gray, (left, top, right, bottom), bb, cap_bbs, cs)
            if guard_bottom is not None:
                bottom = min(H, guard_bottom)
            sub = full.crop((left, top, right, bottom))
            ignore = _trim_ignore_mask(sub.size[::-1], (left, top, right, bottom), bb, text_lines, drawing_bbs, cs)
            (tx0, ty0, tx1, ty1), edges = _trim_box(sub, ignore_mask=ignore, return_edges=True)
            cut_l, cut_t, cut_r, cut_b = edges
            if tx0 > 0 and ignore[:, max(0, tx0 - PAD_PX):tx0].any():
                cut_l = True
            if ty0 > 0 and ignore[max(0, ty0 - PAD_PX):ty0, :].any():
                cut_t = True
            if tx1 < sub.width and ignore[:, tx1:min(sub.width, tx1 + PAD_PX)].any():
                cut_r = True
            if ty1 < sub.height and ignore[ty1:min(sub.height, ty1 + PAD_PX), :].any():
                cut_b = True
            sx0 = tx0 if cut_l else max(0, tx0 - PAD_PX)
            sy0 = ty0 if cut_t else max(0, ty0 - PAD_PX)
            sx1 = tx1 if cut_r else min(sub.width, tx1 + PAD_PX)
            sy1 = ty1 if cut_b else min(sub.height, ty1 + PAD_PX)
            if guard_bottom is not None:
                sy1 = sub.height                              # キャプション直前までの白い隙間は保持
            lp, tp = max(0, PAD_PX - (tx0 - sx0)), max(0, PAD_PX - (ty0 - sy0))
            rp, bp = max(0, PAD_PX - (sx1 - tx1)), max(0, PAD_PX - (sy1 - ty1))
            sub = sub.crop((sx0, sy0, sx1, sy1))
            if lp or tp or rp or bp:
                canvas = Image.new(sub.mode, (sub.width + lp + rp, sub.height + tp + bp), "white")
                canvas.paste(sub, (lp, tp))
                sub = canvas
            sub.save(os.path.join(out_dir, fn), quality=90)
            manifest.append({"file": fn, "page": pno + 1, "bbox_pt": [round(x, 1) for x in bb], **extra})

        if top:                                                    # ── 位置基準（領域ごと）
            for k, d in enumerate(regions[:top], 1):
                _crop(region_bbs[k - 1], f"fig_p{pno+1:02d}_top{k:02d}_{d['label']}.jpg",
                      fig=None, label=d["label"])
            continue
        if panels:                                                 # ── a/b/c パネル分割（領域ごと）
            seen = {}
            for i, d in enumerate(regions):
                num = assign.get(i)
                if (want is not None and num not in want) or (num is None and want is not None):
                    continue
                if num is not None:
                    pl = _panel_of(region_bbs[i], plabels)
                    base = f"Fig{num}_{pl}" if pl else f"Fig{num}_{d['label']}"
                else:
                    base = f"x_{d['label']}"
                seen[base] = seen.get(base, 0) + 1
                tag = base + (f"-{seen[base]}" if seen[base] > 1 else "")
                _crop(region_bbs[i], f"fig_p{pno+1:02d}_{tag}.jpg",
                      cap_bbs=cap_map.get(num, ()) if num is not None else (),
                      fig=(f"Fig{num}" if num is not None else None), label=d["label"])
            continue
        # ── 既定：図ごとに一括（全体）
        wholes = _whole_figures(region_bbs, assign)
        for num in sorted(wholes):
            if want is not None and num not in want:
                continue
            _crop(wholes[num], f"fig_p{pno+1:02d}_Fig{num}.jpg",
                  cap_bbs=cap_map.get(num, ()), fig=f"Fig{num}", label="figure")
        if want is None:                                           # 番号に属さない領域は個別フォールバック
            xn = 0
            for i, d in enumerate(regions):
                if i not in assign:
                    bb = region_bbs[i]
                    area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
                    if any(_rect_overlap(bb, wbb) >= 0.80 * area for wbb in wholes.values()):
                        continue
                    xn += 1
                    _crop(bb, f"fig_p{pno+1:02d}_x{xn:02d}_{d['label']}.jpg",
                          fig=None, label=d["label"])
    json.dump(manifest, open(os.path.join(out_dir, "figures.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"device={dev}  pages={len(doc)}  figures={len(manifest)}"
          + (f"  (figs={sorted(want)})" if want else "") + f" -> {out_dir}")
    for x in manifest:
        sc = f"{x['score']:.2f}" if x.get("score") is not None else "-"
        print(f"  {x['file']}  {x['label']:7} {sc}  {x.get('fig') or ''}")
    return manifest

# ========================= ③ 常駐サーバ（mineru-env）=========================
def serve(device="auto", port=None):
    """モデルを1回だけロードして常駐し、HTTP で図抽出要求を受ける（単発1ページを高速化）。"""
    import time
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    port = int(port or os.environ.get("FIG_PORT", "8077"))
    print(f"[fig-server] loading model (device={device}) ...", flush=True)
    t = time.perf_counter()
    model = _engine(device)                     # OpenVINO(既定 auto→ov:GPU・起動キャッシュで速い) / torch(xpu,cuda)
    try:                                        # ★起動時に1回ダミー推論＝(torchならJIT/OVなら初回コンパイル)をここで済ませる
        from PIL import Image as _I
        tw = time.perf_counter()
        model[0].predict(_I.new("RGB", (1024, 1024), "white"))
        print(f"[fig-server] JIT warmup done in {time.perf_counter()-tw:.1f}s", flush=True)
    except Exception as e:
        print("[fig-server] warmup skipped:", repr(e)[:100], flush=True)
    print(f"[fig-server] ready on {model[1]} in {time.perf_counter()-t:.1f}s", flush=True)
    app = FastAPI()

    class Req(BaseModel):
        pdf: str
        out_dir: str
        figs: list[int] | None = None           # 例 [1,2]＝その Fig 番号だけ
        top: int | None = None                  # 例 2＝各ページ上から2図（位置基準・密ページ用）
        panels: bool = False                    # True＝a/b/c パネル分割（既定 False＝図ごと一括）

    @app.get("/health")
    def health():
        return {"status": "ok", "device": model[1]}

    @app.post("/extract")
    def _extract(r: Req):
        import time as _t
        s = _t.perf_counter()
        man = extract(r.pdf, r.out_dir, model=model, figs=r.figs, top=r.top, panels=r.panels)
        return {"device": model[1], "elapsed_s": round(_t.perf_counter() - s, 3),
                "n": len(man), "figures": man}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "serve":
        serve(sys.argv[2] if len(sys.argv) > 2 else "auto")
    elif cmd == "extract":
        if len(sys.argv) < 4:
            print("usage: 図切り抜き.py extract <pdf> <out_dir> [auto|xpu|cpu] [figs=1,2]"); sys.exit(1)
        dev = sys.argv[4] if len(sys.argv) > 4 else "auto"
        figs = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else None
        extract(sys.argv[2], sys.argv[3], dev, figs=figs)
    else:
        print("usage: 図切り抜き.py serve|extract …  （①手動クロップ関数は主環境で import して使用）")
