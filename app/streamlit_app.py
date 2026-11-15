"""
==============================================================================
File      : app/streamlit_app.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
"Le Rayon" - bibliothèque nocturne. Demo interface (assignment step 6), in French.

Pages:
  - Vue d'ensemble   : key figures, the two models and the gate alpha = n / (n + k)
                       (SVG curve on a log axis), how to read the popularity gauge
  - Lecteur existant : named demo readers (grand public / pépites / éclectique),
                       5 models (hybride adaptatif, hybride 50/50, ALS, TF-IDF,
                       popularité), duel ALS vs hybride adaptatif with an honest verdict
  - Nouveau lecteur  : search + add 3 books, content-only recommendations (alpha = 0),
                       other editions of the picked books are excluded
  - Résultats        : relevance vs long-tail chart, French metric table (incl. cold start),
                       how k was chosen, report figures
Every list shows a gauge Best-seller / Intermédiaire / Pépite (thresholds and
vocabulary from src/labels.py) and each book card shows its class, its number
of readers and, for hybrids, the share of the score coming from ALS and from content.
Covers are generated from the title (real covers only if the data has an image_url).

Input:
  - artifacts written by scripts/run_pipeline.py (via src/recommend.load_artifacts)
  - results/tables/test_metrics.csv, k_sweep_val.csv, best_k.json, results/figures/*.png
Output:
  - web page (local). Set BOOKREC_DEBUG=1 to show full error traces.
Run:
  streamlit run app/streamlit_app.py
==============================================================================
"""

import hashlib
import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Allow execution from the project root or with the app path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src import recommend  # noqa: E402
from src.editions import dedupe_candidates, work_keys  # noqa: E402
from src.explain import shelf_key  # noqa: E402
from src.labels import (  # noqa: E402
    METRIC_HELP, METRIC_LABEL, METRIC_SHORT, MODEL_LABEL, NBSP, PERCENT_METRICS, POP_CLASSES, POP_LABEL,
    POP_LABEL_PLURAL,
    fr_metric, fr_number, fr_percent, model_k, model_label, popularity_class,
)

NBSP_HYPHEN = "‑"          # non-breaking hyphen: "best‑seller" is never cut
DEBUG = os.environ.get("BOOKREC_DEBUG", "") == "1"

MODELS = {
    MODEL_LABEL["gate"]: "gate",
    MODEL_LABEL["fixed"]: "fixed",
    MODEL_LABEL["als"]: "als",
    MODEL_LABEL["content"]: "content",
    MODEL_LABEL["popularity"]: "popularity",
}

# Streamlit >= 1.50 : width="stretch" ; versions antérieures : use_container_width.
try:
    _ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    _ST_VERSION = (1, 0)
_NEW_WIDTH_API = _ST_VERSION >= (1, 50)


def stretch() -> dict:
    return {"width": "stretch"} if _NEW_WIDTH_API else {"use_container_width": True}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,800;1,9..144,400;1,9..144,600&family=Hanken+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root {
  --bg:#100f0d; --card:#1a1815; --card2:#211e1a; --line:#363026;
  --paper:#f1e8d6; --muted:#a39a8a; --head:#ff5a36; --mid:#e8b84a; --tail:#4fd1b8;
}
html, body, .stApp { font-family:'Hanken Grotesk',sans-serif; }
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 85% -5%, rgba(255,90,54,.13) 0%, transparent 38%),
    radial-gradient(ellipse at 5% 10%, rgba(79,209,184,.10) 0%, transparent 38%),
    var(--bg);
  color:var(--paper);
}
[data-testid="stSidebar"] { background:#0b0a09; border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:rgba(16,15,13,.94); backdrop-filter:blur(6px); }
.block-container { max-width:1280px; padding-top:1.4rem; padding-bottom:4rem; }
h1,h2,h3,h4 { font-family:'Fraunces',serif; letter-spacing:-.02em; color:var(--paper); }
h4 { font-weight:600; }
label, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
footer { visibility:hidden; }

/* ---------- Hero ---------- */
.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:6px 28px 6px 28px;
  padding:38px 42px 0; margin-bottom:24px; background:linear-gradient(160deg,#1c1915,#13110e); }
.hero-text { position:relative; z-index:2; max-width:560px; padding-bottom:26px; padding-right:24px; }
.eyebrow { font-family:'Space Mono',monospace; font-size:.8rem; letter-spacing:.22em; text-transform:uppercase; color:var(--tail); }
.hero h1 { font-size:clamp(2.2rem,5vw,3.9rem); line-height:1.04; margin:.55rem 0 .7rem; font-weight:800; }
.hero h1 em { font-style:italic; font-weight:600; color:var(--head); }
.hero p { color:#cfc6b4; font-size:1.02rem; max-width:560px; margin:0; }
.shelf { position:absolute; right:34px; bottom:12px; display:flex; align-items:flex-end; gap:4px; z-index:1; opacity:.92; }
.spine { width:var(--w); height:var(--h); background:var(--c); border-radius:3px 3px 0 0;
  box-shadow:inset -4px 0 0 rgba(0,0,0,.22), inset 0 6px 0 rgba(255,255,255,.12);
  transform-origin:bottom; animation:rise .9s cubic-bezier(.2,.8,.2,1) both; animation-delay:var(--d); }
.spine::after { content:''; display:block; margin:14px auto 0; width:60%; height:2px; background:rgba(255,255,255,.35); }
@keyframes rise { from { transform:scaleY(.05); opacity:0 } to { transform:scaleY(1); opacity:1 } }
.shelf-board { position:absolute; left:0; right:0; bottom:0; height:12px; background:#2a251d; border-top:2px solid #46402f; z-index:2; }
.hero.compact { padding:22px 34px 24px; border-radius:6px 22px 6px 22px; }
.hero.compact h1 { font-size:clamp(1.6rem,3vw,2.3rem); margin:.3rem 0 .35rem; }
.hero.compact p { font-size:.95rem; }
.hero.compact .hero-text { padding-bottom:0; }
@media (max-width:900px){ .shelf{ display:none; } }

/* ---------- Titres de section ---------- */
.kicker { font-family:'Space Mono',monospace; font-size:.8rem; letter-spacing:.2em; text-transform:uppercase; color:var(--head); margin-bottom:2px; }

/* ---------- Surfaces ---------- */
.chapter { background:var(--card); border:1px solid var(--line); border-radius:4px 20px 4px 20px; padding:22px 24px; height:100%; }
.chapter .num { font-family:'Fraunces',serif; font-style:italic; font-size:2.6rem; line-height:1; color:var(--head); opacity:.9; }
.chapter h3 { margin:.35rem 0 .35rem; font-size:1.25rem; }
.chapter p { color:#bdb4a3; font-size:.92rem; margin:0; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:18px 20px; }
.panel h4 { margin:0 0 4px; }
.panel p { color:var(--muted); font-size:.88rem; margin:0 0 6px; }

/* ---------- Cartes livres ---------- */
.book { display:flex; gap:16px; background:linear-gradient(120deg,var(--card2),var(--card));
  border:1px solid var(--line); border-radius:4px 16px 4px 16px; padding:14px 16px; margin:12px 0;
  transition:transform .18s ease, border-color .18s ease; }
.book:hover { transform:translateY(-2px); border-color:#5a5242; }
.cover { flex:0 0 70px; width:70px; height:100px; border-radius:3px 6px 6px 3px; object-fit:cover;
  box-shadow:3px 4px 0 rgba(0,0,0,.35), inset 5px 0 0 rgba(0,0,0,.22); }
.gcover { flex:0 0 70px; width:70px; height:100px; border-radius:3px 6px 6px 3px; padding:10px 8px 8px 12px;
  display:flex; flex-direction:column; justify-content:space-between; overflow:hidden;
  box-shadow:3px 4px 0 rgba(0,0,0,.35), inset 5px 0 0 rgba(0,0,0,.25); }
.gcover span { font-family:'Fraunces',serif; font-weight:600; font-size:.62rem; line-height:1.15; color:rgba(255,255,255,.95);
  overflow-wrap:anywhere; hyphens:auto; -webkit-line-clamp:4 !important;
  display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden; }
.gcover i { display:block; width:60%; height:2px; background:rgba(255,255,255,.55); }
.bbody { flex:1; min-width:0; }
.btop { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.rk { font-family:'Fraunces',serif; font-style:italic; font-size:1.7rem; line-height:1; color:var(--head); }
.bt { font-family:'Fraunces',serif; font-weight:600; font-size:1.02rem; color:var(--paper); }
.tags { margin:7px 0 4px; }
.tag { display:inline-block; font-family:'Space Mono',monospace; font-size:.76rem; padding:3px 8px; margin:0 6px 4px 0;
  border:1px solid var(--line); border-radius:3px; color:#d8cfbd; }
.tag.head { color:var(--head); border-color:var(--head); }
.tag.mid  { color:var(--mid);  border-color:var(--mid); }
.tag.tail { color:var(--tail); border-color:var(--tail); }
.voice { display:flex; height:5px; border-radius:3px; overflow:hidden; margin:6px 0 2px; background:#2b2620; }
.voice b { display:block; height:100%; }
.voice .a { background:var(--head); } .voice .c { background:var(--tail); }
.vlabel { font-family:'Space Mono',monospace; font-size:.76rem; color:var(--muted); letter-spacing:.04em; }
.bdesc { margin:7px 0 0; font-size:.86rem; line-height:1.55; color:#bdb4a3; font-style:italic; }

/* ---------- Jauge de découverte ---------- */
.meter { display:flex; height:22px; border-radius:4px; overflow:hidden; border:1px solid var(--line); margin:8px 0 6px; }
.meter div { display:flex; align-items:center; justify-content:center; font-family:'Space Mono',monospace;
  font-size:.76rem; font-weight:700; color:#12100d; min-width:0; white-space:nowrap; }
.meter .h { background:var(--head); } .meter .m { background:var(--mid); } .meter .t { background:var(--tail); }
.legend { font-family:'Space Mono',monospace; font-size:.76rem; color:var(--muted); }
.legend span { margin-right:14px; white-space:nowrap; display:inline-block; }
.legend i { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:5px; }
.verdict { font-family:'Fraunces',serif; font-size:1.15rem; line-height:1.4; border-left:3px solid var(--tail);
  padding:4px 0 4px 16px; margin:14px 0 6px; color:var(--paper); }
.verdict b { color:var(--tail); }

/* ---------- Divers ---------- */
.best { background:var(--card); border:1px solid var(--line); border-radius:4px 14px 4px 14px; padding:14px 16px; }
.best .l { font-family:'Space Mono',monospace; font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.best .v { font-family:'Fraunces',serif; font-size:1.25rem; color:var(--paper); margin-top:3px; }
.best .s { font-family:'Space Mono',monospace; font-size:.74rem; color:var(--tail); }
.brand { font-family:'Fraunces',serif; font-weight:800; font-size:1.5rem; letter-spacing:-.02em; }
.brand em { color:var(--head); font-weight:600; }
.brandsub { font-family:'Space Mono',monospace; font-size:.66rem; letter-spacing:.1em; color:var(--muted); }
[data-testid="stSidebar"] > div { overflow-x:hidden; }
.credit { font-size:.78rem; color:var(--muted); margin-top:4px; }
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:4px 14px 4px 14px; padding:14px 17px; }
[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-family:'Space Mono',monospace; font-size:.8rem; letter-spacing:.06em; }
[data-testid="stMetricValue"] { color:var(--paper); font-family:'Fraunces',serif; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
.svgwrap svg { width:100%; height:auto; display:block; }
.nowrap { white-space:nowrap; }
.howto { font-size:.86rem; color:var(--muted); border:1px dashed var(--line); border-radius:6px; padding:8px 12px; margin:6px 0 1.5rem; }
.howto b { color:var(--paper); }
.slot { flex:0 0 70px; width:70px; height:100px; border:1px dashed var(--line); border-radius:3px 6px 6px 3px;
  display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:1.4rem; }
.note { font-size:.92rem; color:#cfc6b4; border-left:3px solid var(--mid); padding:6px 0 6px 14px; margin:8px 0 14px; }
</style>
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def md(markup: str) -> None:
    """Render HTML on a single line (avoids Markdown code-block parsing)."""
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def show_error(err: Exception, message: str) -> None:
    st.error(f"{message}{NBSP}: {err}")
    if DEBUG:
        st.exception(err)


def heading(kicker: str, title: str, note: str = "") -> None:
    md(f'<div class="kicker">{esc(kicker)}</div>')
    st.subheader(title)
    if note:
        st.caption(note)


def plural(n: int, word: str) -> str:
    """'2887 lecteurs' -> '2 887 lecteurs' (French grouping, plural only above 1)."""
    return f"{fr_number(n, 0)}{NBSP}{word}" + ("s" if n > 1 else "")


SPINES = [
    (88, 16, "#ff5a36"), (122, 20, "#e8b84a"), (104, 14, "#4fd1b8"), (146, 22, "#8f6bff"),
    (96, 16, "#f1e8d6"), (132, 18, "#ff5a36"), (112, 22, "#2f8f83"), (150, 16, "#e8b84a"),
    (100, 20, "#b8472b"), (126, 14, "#4fd1b8"), (92, 18, "#8f6bff"), (140, 20, "#f1e8d6"),
]


def hero(title_html: str, subtitle: str, compact: bool = False) -> None:
    shelf = ""
    if not compact:
        spines = "".join(
            f'<div class="spine" style="--h:{h}px;--w:{w}px;--c:{c};--d:{i * 0.07:.2f}s"></div>'
            for i, (h, w, c) in enumerate(SPINES)
        )
        shelf = f'<div class="shelf">{spines}</div><div class="shelf-board"></div>'
    md(
        f'<section class="hero{" compact" if compact else ""}">'
        f'<div class="hero-text"><div class="eyebrow">BookRec · Goodreads · Systèmes de recommandation</div>'
        f'<h1>{title_html}</h1><p>{esc(subtitle)}</p></div>{shelf}</section>'
    )


# ---------------------------------------------------------------- popularity gauge
def pop_class(value) -> str | None:
    """Gauge class of a row: 'head' / 'mid' / 'tail' (src/labels.py), legacy labels accepted."""
    p = str(value or "").lower()
    if p in POP_CLASSES:
        return p
    if "best" in p or "head" in p:
        return "head"
    if "mid" in p or "interm" in p:
        return "mid"
    if "tail" in p or "new" in p or "pépite" in p or "niche" in p:
        return "tail"
    return None


def meter_counts(df: pd.DataFrame) -> dict | None:
    """Counts and shares of each class, over ALL books of the list (denominator = N)."""
    if df is None or df.empty or "popularity" not in df.columns:
        return None
    classes = [pop_class(v) for v in df["popularity"]]
    total = len(classes)
    counts = {k: classes.count(k) for k in POP_CLASSES}
    return {"n": total, "counts": counts, "shares": {k: counts[k] / total for k in POP_CLASSES}}


def thresholds_text(art: dict) -> str:
    mid_min, head_min = art["pop_thresholds"]
    return (f"Best{NBSP_HYPHEN}seller ≥{NBSP}{head_min:.0f}{NBSP}lecteurs · "
            f"Intermédiaire ≥{NBSP}{mid_min:.0f} · Pépite{NBSP}<{NBSP}{mid_min:.0f}")


def meter_html(meter: dict, art: dict | None = None) -> str:
    cells = ""
    for key, css in (("head", "h"), ("mid", "m"), ("tail", "t")):
        share = meter["shares"][key]
        count = meter["counts"][key]
        if count:
            pct = share * 100
            label = fr_percent(share) if pct >= 12 else ""
            tip = f"{plural(count, 'livre')} · {fr_percent(share)}"
            cells += f'<div class="{css}" style="width:{pct:.1f}%" title="{esc(tip)}">{label}</div>'
    legend = "".join(
        f'<span><i style="background:var(--{k})"></i>{POP_LABEL_PLURAL[k]}{NBSP}({meter["counts"][k]})</span>'
        for k in POP_CLASSES
    )
    rule = f'<div class="legend" style="margin-top:2px">{esc(thresholds_text(art))}</div>' if art else ""
    return f'<div class="meter">{cells}</div><div class="legend">{legend}</div>{rule}'


# ---------------------------------------------------------------- gate curve
def gate_svg(k: float, n_max: float) -> str:
    """alpha = n / (n + k) on a log x axis from 1 to n_max readers."""
    k = max(float(k), 0.0)
    n_max = max(float(n_max), 10.0)
    W, H, L, B, T, R = 640, 270, 64, 46, 16, 18
    lo, hi = 0.0, np.log10(n_max)

    def x_of(n):
        return L + (W - L - R) * (np.log10(n) - lo) / (hi - lo)

    def y_of(a):
        return H - B - (H - B - T) * a

    pts = []
    for i in range(121):
        n = 10 ** (lo + (hi - lo) * i / 120)
        a = 1.0 if k == 0 else n / (n + k)
        pts.append(f"{x_of(n):.1f},{y_of(a):.1f}")
    line = "M" + " L".join(pts)
    area = f"{line} L{W - R},{H - B} L{L},{H - B} Z"
    font = 'font-family="Space Mono, monospace" font-size="13"'
    grid = ""
    for a in (0.25, 0.5, 0.75, 1.0):
        y = y_of(a)
        grid += (f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#363026" stroke-dasharray="3 5"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#a39a8a" {font}>{fr_number(a, 2)}</text>')
    xticks = ""
    for n in (1, 5, 10, 50, 100, 500, 1000, 5000):
        if n <= n_max:
            xticks += (f'<text x="{x_of(n):.1f}" y="{H - B + 20}" text-anchor="middle" fill="#a39a8a" {font}>'
                       f'{fr_number(n, 0)}</text>')
    marker = ""
    if 1 <= k <= n_max:
        kx, ky = x_of(k), y_of(0.5)
        marker = (f'<line x1="{kx:.1f}" x2="{kx:.1f}" y1="{ky:.1f}" y2="{H - B}" stroke="#ff5a36" stroke-dasharray="4 4"/>'
                  f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="6" fill="#ff5a36"/>'
                  f'<text x="{kx + 12:.1f}" y="{ky + 20:.1f}" fill="#ff5a36" {font}>n = k = {fr_number(k, 0)} → α = 0,5</text>')
    return (
        f'<div class="svgwrap"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Courbe de l\'aiguillage alpha">'
        f'{grid}{xticks}'
        f'<path d="{area}" fill="#4fd1b8" opacity=".12"/>'
        f'<path d="{line}" fill="none" stroke="#4fd1b8" stroke-width="3" stroke-linecap="round"/>'
        f'{marker}'
        f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle" fill="#a39a8a" {font}>'
        f'lecteurs du livre dans l\'entraînement (n, échelle log)</text>'
        f'<text transform="translate(16,{(T + H - B) / 2:.1f}) rotate(-90)" text-anchor="middle" fill="#a39a8a" {font}>'
        f'poids d\'ALS (α)</text>'
        f'</svg></div>'
    )


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modèles ALS et TF-IDF…")
def get_artifacts():
    cfg = load_config()
    art = recommend.load_artifacts(cfg)
    return cfg, art


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float, index_col=0):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=index_col)


def read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def build_cover_map(_books: pd.DataFrame) -> dict:
    """book_id -> real cover URL, only if the data has an image column (not the case by default)."""
    for col in ("image_url", "cover_url", "image"):
        if col in _books.columns and "book_id" in _books.columns:
            out = {}
            for bid, url in zip(_books["book_id"].astype(str), _books[col]):
                u = str(url or "")
                if u.startswith("http") and "nophoto" not in u:
                    out[bid] = u
            return out
    return {}


@st.cache_resource(show_spinner=False)
def demo_readers(_art: dict) -> dict:
    """user_id -> readable label. The 3 most contrasted readers come first, with a name."""
    train, th = _art["train"], _art["pop_thresholds"]
    rows = []
    for uid in _art["demo_users"]:
        seen = train[_art["user2idx"][uid]].indices
        if len(seen) < 3:
            continue
        cls = [popularity_class(int(_art["n_items"][i]), thresholds=th) for i in seen]
        rows.append((uid, len(seen), cls.count("head") / len(cls), cls.count("tail") / len(cls)))
    if not rows:
        return {}
    labels, used = {}, set()

    rich = [r for r in rows if r[1] >= 5] or rows        # named readers: at least 5 books if possible

    def take(name, key):
        for uid, n, head, tail in sorted(rich, key=key):
            if uid not in used:
                used.add(uid)
                labels[uid] = f"{name} · {plural(n, 'livre')} lu{'s' if n > 1 else ''}"
                return

    take("Lecteur grand public", key=lambda r: (-r[2], -r[1]))
    take("Lecteur de pépites", key=lambda r: (-r[3], -r[1]))
    take("Lecteur éclectique", key=lambda r: (abs(r[2] - 0.15) + abs(r[3] - 0.5), -r[1]))
    others = [r for r in sorted(rich, key=lambda r: -r[1]) if r[0] not in used][:12]
    for i, (uid, n, head, tail) in enumerate(others, start=1):
        labels[uid] = f"Lecteur {i:02d} · {plural(n, 'livre')} lu{'s' if n > 1 else ''}"
    return labels


def short_title(title: str, limit: int = 42) -> str:
    """Cut a long title after a whole word (generated covers)."""
    title = str(title)
    if len(title) <= limit:
        return title
    return title[:limit].rsplit(" ", 1)[0].rstrip(",:;") + "…"


def generated_cover(title: str) -> str:
    h = int(hashlib.md5(str(title).encode("utf-8")).hexdigest(), 16)
    hue1 = h % 360
    hue2 = (hue1 + 35 + (h >> 8) % 70) % 360
    bg = f"linear-gradient(155deg,hsl({hue1},58%,38%),hsl({hue2},62%,22%))"
    return f'<div class="gcover" style="background:{bg}"><span>{esc(short_title(title))}</span><i></i></div>'


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
HOW_TO_READ = (
    '<div class="howto"><b>Lire une carte</b> : le badge donne la popularité du livre '
    '(best‑seller, intermédiaire ou pépite) et son nombre de lecteurs ; pour les hybrides, la barre montre '
    'quelle part du score vient des lecteurs similaires (ALS, rouge) et du contenu (TF‑IDF, vert) ; '
    'la phrase en italique explique le choix.</div>'
)


def book_card(row: pd.Series, covers: dict) -> None:
    raw_title = str(row.get("title", "Livre sans titre"))
    explanation = esc(row.get("explanation", ""))
    rank = esc(row.get("rank", ""))
    url = covers.get(str(row.get("book_id", "")))
    cover = f'<img class="cover" src="{esc(url)}" alt="">' if url else generated_cover(raw_title)
    pc = pop_class(row.get("popularity"))
    pop_tag = f'<span class="tag {pc}">{POP_LABEL[pc]}</span>' if pc else ""
    try:
        readers_tag = f'<span class="tag">{plural(int(row["n"]), "lecteur")}</span>'
    except (KeyError, TypeError, ValueError):
        readers_tag = ""
    voice = ""
    cf = row.get("cf_share", np.nan)
    try:
        if pd.notna(cf):
            a = min(max(float(cf), 0.0), 1.0)
            voice = (
                f'<div class="voice"><b class="a" style="width:{a * 100:.1f}%"></b>'
                f'<b class="c" style="width:{(1 - a) * 100:.1f}%"></b></div>'
                f'<div class="vlabel">part du score{NBSP}: lecteurs similaires {fr_percent(a)} · '
                f'contenu {fr_percent(1 - a)}</div>'
            )
    except (TypeError, ValueError):
        pass
    md(
        f'<div class="book">{cover}<div class="bbody">'
        f'<div class="btop"><span class="rk">{rank}</span><span class="bt">{esc(raw_title)}</span></div>'
        f'<div class="tags">{pop_tag}{readers_tag}</div>{voice}'
        f'<div class="bdesc">{explanation}</div></div></div>'
    )


def show_recommendations(df: pd.DataFrame, title: str, key: str, covers: dict, art: dict,
                         excluded: str = "les livres déjà lus et leurs autres éditions sont exclus") -> None:
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("Aucune recommandation disponible pour cette sélection.")
        return
    meter = meter_counts(df)
    if meter:
        md(meter_html(meter, art))
    st.caption(f"{plural(len(df), 'livre')} recommandé{'s' if len(df) > 1 else ''} · {excluded}")
    for _, row in df.head(5).iterrows():
        book_card(row, covers)
    if len(df) > 5:
        with st.expander(f"Afficher les {len(df) - 5} suivants"):
            for _, row in df.iloc[5:].iterrows():
                book_card(row, covers)
    export = df.drop(columns=[c for c in ("alpha",) if c in df.columns])
    st.download_button(
        "Exporter la liste (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bookrec_{key}.csv",
        mime="text/csv",
        key=f"export_{key}",
        **stretch(),
    )


def discovery_verdict(als_recs: pd.DataFrame, gate_recs: pd.DataFrame, k: float) -> None:
    """One honest sentence comparing the two lists (gain, tie or loss)."""
    a, g = meter_counts(als_recs), meter_counts(gate_recs)
    if not a or not g:
        return
    n = g["n"]
    ta, tg = a["counts"]["tail"], g["counts"]["tail"]
    ha, hg = a["counts"]["head"], g["counts"]["head"]
    common = len(set(als_recs["book_id"]) & set(gate_recs["book_id"])) if "book_id" in als_recs else None
    same = f" Les deux listes ont {plural(common, 'livre')} en commun." if common is not None else ""
    if tg > ta:
        text = (f"Sur {n}{NBSP}livres, l'hybride propose <b>{tg} pépites</b> contre {ta} pour ALS, "
                f"et {hg} best{NBSP_HYPHEN}seller{'s' if hg > 1 else ''} contre {ha}.{same}")
    elif tg < ta:
        text = (f"Pour ce lecteur, l'hybride propose <b>moins de pépites</b> qu'ALS ({tg} contre {ta} "
                f"sur {n}) : avec k{NBSP}={NBSP}{fr_number(k, 0)}, un livre lu par plus de "
                f"{fr_number(k, 0)}{NBSP}lecteurs est surtout classé par ALS.{same}")
    else:
        text = f"Les deux modèles proposent autant de pépites (<b>{tg} sur {n}</b>).{same}"
    md(f'<div class="verdict">{text}</div>')


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def show_dashboard(cfg: dict, art: dict) -> None:
    hero(f'Au-delà des <em class="nowrap">best{NBSP_HYPHEN}sellers</em>.',
         "Diversifier les découvertes, sans sacrifier la pertinence.")
    a, b, c, d = st.columns(4)
    a.metric("Livres", fr_number(len(art["books"]), 0))
    b.metric("Lecteurs", fr_number(art["train"].shape[0], 0))
    c.metric("Interactions", fr_number(art["train"].nnz, 0))
    d.metric("Seuil k", fr_number(float(art["k"]), 0 if float(art["k"]).is_integer() else 1),
             help="k = nombre de lecteurs à partir duquel ALS et le contenu pèsent autant (α = 0,5).")
    st.write("")
    heading("Le concept", "Deux modèles, un aiguillage",
            "Chaque modèle apporte un point de vue ; l'aiguillage (« gate ») choisit, livre par livre, lequel écouter.")
    cols = st.columns(3)
    concepts = [
        ("I", MODEL_LABEL["als"], "S'appuie sur les goûts partagés entre lecteurs : précis, mais attiré par les livres déjà très lus."),
        ("II", MODEL_LABEL["content"], "Compare les descriptions et les tags des livres : fonctionne même pour un livre que personne n'a encore lu."),
        ("III", MODEL_LABEL["gate"], "Combine I et II : écoute ALS quand le livre a beaucoup de lecteurs, le contenu quand il en a peu."),
    ]
    for col, (num, label, desc) in zip(cols, concepts):
        with col:
            md(f'<div class="chapter"><div class="num">{num}</div><h3>{esc(label)}</h3><p>{esc(desc)}</p></div>')
    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        md('<div class="panel"><div class="kicker">L\'aiguillage en une courbe</div>'
           '<h4>α = n / (n + k)</h4>'
           '<p>Plus un livre a de lecteurs (n), plus la voix d\'ALS pèse. '
           'Les livres peu lus restent portés par leur contenu.</p></div>')
        md(gate_svg(float(art["k"]), float(np.max(art["n_items"]))))
    with right:
        md('<div class="panel"><div class="kicker">Lire la jauge</div><h4>La popularité, enfin visible</h4>'
           '<p>Chaque liste est résumée par une jauge : part de best‑sellers (5 % des livres les plus lus), '
           'd\'intermédiaires (les 15 % suivants) et de pépites (les 80 % restants, la longue traîne).</p></div>')
        md(meter_html({"n": 20, "counts": {"head": 11, "mid": 5, "tail": 4},
                       "shares": {"head": 0.55, "mid": 0.25, "tail": 0.20}}, art))
        st.caption("Exemple illustratif d'une liste dominée par les best‑sellers.")
        st.info("**Lecteur existant** : comparez ALS et l'hybride adaptatif. "
                "**Nouveau lecteur** : choisissez 3 livres, le contenu fait le reste.")
    st.caption("Données : échantillon Goodreads, comics et romans graphiques. "
               "Une meilleure diversité ne garantit pas une meilleure pertinence pour chaque lecteur.")


def show_existing_reader(cfg: dict, art: dict) -> None:
    hero("Un lecteur, <em>deux</em> regards.",
         "Les lectures passées alimentent les modèles ; les explications les rendent lisibles.", compact=True)
    covers = build_cover_map(art["books"])
    readers = demo_readers(art)
    if not readers:
        st.warning("Aucun lecteur de démonstration disponible.")
        return
    opts, controls = st.columns([2.1, 1])
    with opts:
        user_id = st.selectbox("Lecteur", list(readers), format_func=lambda u: readers.get(u, u),
                               key="existing_user")
    with controls:
        top_n = st.slider("Nombre de recommandations", 5, 20,
                          int(min(10, cfg["app"]["default_top_n"])), key="existing_top")
    model_label = st.radio("Modèle de recommandation", list(MODELS), horizontal=True, key="existing_model")
    selected = MODELS[model_label]
    compare = False
    if selected == "gate":
        # the widget disappears for other models; keep the choice in our own key
        compare = st.checkbox(f"Comparer avec {MODEL_LABEL['als']}",
                              value=st.session_state.get("compare_pref", True))
        st.session_state["compare_pref"] = compare

    user_idx = art["user2idx"][str(user_id)]
    seen_idx = art["train"][user_idx].indices
    th = art["pop_thresholds"]
    seen = pd.DataFrame({
        "Titre": [str(art["books"].iloc[i]["title"]) for i in seen_idx],
        "Lecteurs": [int(art["n_items"][i]) for i in seen_idx],
        "popularity": [popularity_class(int(art["n_items"][i]), thresholds=th) for i in seen_idx],
    }).sort_values("Lecteurs", ascending=False)

    m1, m2 = st.columns([1, 2.2], gap="large")
    m1.metric("Livres déjà lus", len(seen))
    with m2:
        meter = meter_counts(seen)
        if meter:
            md('<div class="panel"><div class="kicker">Profil de lecture</div>'
               f'<h4>Ce lecteur lit surtout des {POP_LABEL_PLURAL[max(POP_CLASSES, key=lambda c: meter["counts"][c])].lower()}</h4></div>')
            md(meter_html(meter, art))
    with st.expander("Voir les livres déjà lus", expanded=False):
        if seen.empty:
            st.info("Pas de livres dans l'historique d'entraînement.")
        else:
            table = seen.assign(Catégorie=[POP_LABEL[c] for c in seen["popularity"]]).drop(columns="popularity")
            st.dataframe(table.reset_index(drop=True), hide_index=True, **stretch(),
                         column_config={"Titre": st.column_config.TextColumn("Titre", width="large")})

    try:
        with st.spinner("Calcul des recommandations…"):
            main_recs = recommend.recommend_for_user(user_id, art, model_name=selected, top_n=top_n)
            als_recs = (recommend.recommend_for_user(user_id, art, model_name="als", top_n=top_n)
                        if compare else None)
    except Exception as err:  # noqa: BLE001
        show_error(err, "Erreur de recommandation")
        return

    st.write("")
    md(HOW_TO_READ)
    if compare:
        heading("Le duel", f"{MODEL_LABEL['als']} contre {MODEL_LABEL['gate']}",
                "Même lecteur, même nombre de livres : regardez ce qui change dans la jauge.")
        discovery_verdict(als_recs, main_recs, float(art["k"]))
        # on a narrow screen two columns stack and the comparison is lost: offer tabs
        if st.toggle("Affichage en onglets (petit écran)", key="duel_tabs"):
            left, right = st.tabs([MODEL_LABEL["als"], MODEL_LABEL["gate"]])
        else:
            left, right = st.columns(2, gap="large")
        with left:
            show_recommendations(als_recs, MODEL_LABEL["als"], f"cmp_als_{user_id}", covers, art)
        with right:
            show_recommendations(main_recs, MODEL_LABEL["gate"], f"cmp_gate_{user_id}", covers, art)
    else:
        show_recommendations(main_recs, f"Top {len(main_recs)} · {model_label}",
                             f"main_{selected}_{user_id}", covers, art)


def book_catalog(art: dict) -> pd.DataFrame:
    """One line per work (most-read edition), for the new-reader search."""
    books = art["books"].reset_index(drop=True)
    cat = pd.DataFrame({
        "book_id": books["book_id"].astype(str),
        "title": books["title"].fillna("Sans titre").astype(str),
        "n": np.asarray(art["n_items"]).astype(int),
    })
    work_ids = books["work_id"].astype(str) if "work_id" in books.columns else [None] * len(cat)
    cat["work"] = [sorted(work_keys(t, w))[-1] if w not in (None, "", "nan") else sorted(work_keys(t))[0]
                   for t, w in zip(cat["title"], work_ids)]
    cat = cat.sort_values(["n", "title"], ascending=[False, True]).drop_duplicates("work")
    return cat.reset_index(drop=True)


def has_little_info(art: dict, book_id: str) -> bool:
    i = art["book2idx"].get(str(book_id))
    if i is None:
        return True
    row = art["books"].iloc[i]
    text = str(row.get("text", "") or "")
    shelves = [s for s in (row.get("shelves") if isinstance(row.get("shelves"), (list, np.ndarray)) else [])
               if shelf_key(s) not in art["common_shelves"]]
    return len(text.split()) < 12 and len(shelves) < 2


def _add_picked() -> None:
    """Button callback: add the chosen book, then empty the search box."""
    choice = st.session_state.get("new_choice")
    picked = st.session_state.setdefault("picked_books", [])
    if choice and choice not in picked and len(picked) < 12:
        picked.append(choice)
    st.session_state["new_query"] = ""


def show_new_reader(cfg: dict, art: dict) -> None:
    hero("Votre <em>première</em> étagère.",
         "Pas d'historique ? Trois livres suffisent à dessiner votre profil de lecture.", compact=True)
    heading("Démarrage à froid", "Commencer une nouvelle histoire",
            "Cherchez et ajoutez au moins trois livres que vous appréciez.")
    covers = build_cover_map(art["books"])
    catalog = book_catalog(art)
    picked: list = st.session_state.setdefault("picked_books", [])
    titles = {str(b): str(t) for b, t in zip(art["books"]["book_id"], art["books"]["title"])}
    readers = dict(zip(catalog["book_id"], catalog["n"]))

    query = st.text_input("Rechercher un titre", placeholder="ex. Saga, Maus, Persepolis…", key="new_query")
    if query.strip():
        found = catalog[catalog["title"].str.contains(query.strip(), case=False, regex=False)]
        # hide books already on the shelf AND their other editions
        on_shelf = [titles.get(b, b) for b in picked]
        keep = dedupe_candidates(list(found["title"]), on_shelf, limit=15) if on_shelf else range(min(15, len(found)))
        found = found.iloc[list(keep)]
        if found.empty:
            st.caption("Aucun titre trouvé (ou déjà sur votre étagère).")
        else:
            st.selectbox(
                f"{plural(len(found), 'résultat')}" + (", les plus lus d'abord" if len(found) > 1 else ""),
                list(found["book_id"]),
                format_func=lambda b: f"{titles.get(b, b)} · {plural(int(readers.get(b, 0)), 'lecteur')}",
                key="new_choice",
            )
            st.button("Ajouter à mon étagère", disabled=len(picked) >= 12, key="new_add", on_click=_add_picked)

    # shelf: each cover with its own "Retirer" button below it, plus empty slots up to 3
    n_cols = max(3, len(picked))
    cols = st.columns(min(n_cols, 6))
    for i in range(max(3, len(picked))):
        with cols[i % len(cols)]:
            if i < len(picked):
                b = picked[i]
                md(generated_cover(titles.get(b, b)))
                if st.button("Retirer", key=f"rm_{b}", help=titles.get(b, b)):
                    picked.remove(b)
                    st.rerun()
            else:
                md('<div class="slot">+</div>')
    if picked:
        poor = [titles.get(b, b) for b in picked if has_little_info(art, b)]
        if poor:
            st.caption("Peu d'informations (description ou tags) pour : " + " ; ".join(short_title(t, 40) for t in poor)
                       + ". Ces livres influencent peu les recommandations.")

    missing = 3 - len(picked)
    if missing > 0:
        st.info(f"{plural(len(picked), 'livre')} choisi{'s' if len(picked) > 1 else ''}{NBSP}— "
                f"encore {missing} pour lancer les recommandations.")
        return
    st.caption(f"{plural(len(picked), 'livre')} choisis.")
    top_n = st.slider("Nombre de recommandations", 5, 20, int(min(10, cfg["app"]["default_top_n"])), key="new_top")

    try:
        with st.spinner("Construction de votre profil de lecture…"):
            recs = recommend.recommend_for_new_reader(picked, art, top_n=top_n)
    except Exception as err:  # noqa: BLE001
        show_error(err, "Erreur pendant les recommandations")
        return
    st.success("Profil de lecture créé à partir des livres choisis.")
    md('<div class="note">Démarrage à froid : aucun historique, donc α = 0 et le contenu (TF‑IDF) '
       'fait tout le travail. Les autres éditions des livres choisis sont exclues.</div>')
    recs = recs.assign(cf_share=np.nan)          # no ALS/content bar: it would always say 0 % / 100 %
    show_recommendations(recs, f"Votre sélection · Top {len(recs)}", "new_reader", covers, art,
                         excluded="les livres choisis et leurs autres éditions sont exclus")


def fr_axis(metric: str) -> str:
    """Vega label expression: same format as the values (percent only for shares, comma decimals)."""
    fmt = ".0%" if metric in PERCENT_METRICS else (".1f" if metric == "novelty" else ".2f")
    return f"replace(format(datum.value, '{fmt}'), '.', ',')"


def bar_chart(values: pd.Series, metric: str, highlight: str):
    """Static horizontal bar chart, sorted, the adaptive hybrid in the accent colour."""
    data = pd.DataFrame({"Modèle": values.index, "valeur": values.values})
    data["texte"] = [fr_metric(metric, v) for v in data["valeur"]]
    data["accent"] = data["Modèle"] == highlight
    base = alt.Chart(data).encode(
        y=alt.Y("Modèle:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        x=alt.X("valeur:Q", title=METRIC_LABEL.get(metric, metric),
                axis=alt.Axis(labelFontSize=12, labelExpr=fr_axis(metric))),
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.condition("datum.accent", alt.value("#ff5a36"), alt.value("#4fd1b8")))
    text = base.mark_text(align="left", dx=6, color="#f1e8d6", fontSize=13).encode(text="texte:N")
    return (bars + text).properties(height=max(160, 46 * len(data))).configure_view(strokeWidth=0)


def tradeoff_chart(metrics: pd.DataFrame, highlight: str):
    """Relevance (x) against long-tail share (y): one point per model."""
    data = pd.DataFrame({
        "Modèle": metrics.index,
        "Rappel@20": metrics["all/recall@20"].values,
        "Part de pépites": metrics["tail_share"].values,
    })
    data["accent"] = data["Modèle"] == highlight
    # alternate the label above / below the point so close models stay readable
    data = data.sort_values("Rappel@20").reset_index(drop=True)
    data["dy"] = [-12 if i % 2 == 0 else 18 for i in range(len(data))]
    base = alt.Chart(data).encode(
        x=alt.X("Rappel@20:Q", title="Pertinence : rappel@20 →",
                axis=alt.Axis(labelExpr=fr_axis("all/recall@20"), labelFontSize=12),
                scale=alt.Scale(zero=True, nice=True)),
        y=alt.Y("Part de pépites:Q", title="Découverte : part de pépites ↑",
                axis=alt.Axis(labelExpr=fr_axis("tail_share"), labelFontSize=12), scale=alt.Scale(domain=[0, 1])),
    )
    points = base.mark_circle(size=220).encode(
        color=alt.condition("datum.accent", alt.value("#ff5a36"), alt.value("#4fd1b8")))
    labels_up = base.transform_filter("datum.dy < 0").mark_text(
        align="left", dx=10, dy=-12, color="#f1e8d6", fontSize=13).encode(text="Modèle:N")
    labels_down = base.transform_filter("datum.dy > 0").mark_text(
        align="left", dx=10, dy=18, color="#f1e8d6", fontSize=13).encode(text="Modèle:N")
    return (points + labels_up + labels_down).properties(height=340).configure_view(strokeWidth=0)


def show_results(cfg: dict, art: dict) -> None:
    hero("Ce que disent <em>les chiffres</em>.", "Mesurer le compromis entre pertinence et découverte.",
         compact=True)
    md('<div class="note">Un meilleur score de couverture ou de nouveauté ne garantit pas, à lui seul, '
       'une meilleure pertinence pour chaque lecteur : on cherche le meilleur compromis.</div>')
    table_path = Path(cfg["paths"]["tables_dir"]) / "test_metrics.csv"
    raw = load_csv(str(table_path), table_path.stat().st_mtime if table_path.exists() else 0.0)
    if raw is None or raw.empty:
        st.warning("Les résultats ne sont pas encore disponibles. Lancez le pipeline pour les produire.")
        return

    # k used in the table vs k used by the app
    k_table = next((model_k(m) for m in raw.index if model_k(m) is not None), None)
    if k_table is not None and abs(k_table - float(art["k"])) > 1e-9:
        st.warning(f"Le tableau a été calculé avec k{NBSP}={NBSP}{fr_number(k_table, 0)}, "
                   f"mais l'application utilise k{NBSP}={NBSP}{fr_number(float(art['k']), 0)}. "
                   "Relancez `python scripts/run_pipeline.py` pour mettre le tableau à jour.")

    metrics = raw.copy()
    metrics.index = [model_label(m) for m in metrics.index]
    gate_name = next((m for m in metrics.index if m.startswith(MODEL_LABEL["gate"])), None)
    n_users = int(raw["all/n_users"].iloc[0]) if "all/n_users" in raw.columns else None

    total = art["train"].shape[0]
    heading("Évaluation expérimentale", "Les modèles, côte à côte",
            (f"Jeu de test : {fr_number(n_users, 0)} lecteurs sur {fr_number(total, 0)} "
             f"(les {fr_number(total - n_users, 0)} autres n'ont aucun livre en test, souvent parce qu'il "
             f"n'était pas dans l'entraînement), {fr_number(len(art['books']), 0)} livres.")
            if n_users and n_users < total else
            (f"Jeu de test : {fr_number(n_users, 0)} lecteurs, {fr_number(len(art['books']), 0)} livres."
             if n_users else "Jeu de test."))

    if {"all/recall@20", "tail_share"} <= set(metrics.columns):
        st.markdown("##### Pertinence ou découverte ?")
        st.altair_chart(tradeoff_chart(metrics, gate_name), **stretch())
        st.caption("Chaque point est un modèle. En haut à droite : pertinent et ouvert à la longue traîne.")

    desired = ["all/recall@20", "all/ndcg@20", "cold_items/recall@20", "cold_users/recall@20",
               "coverage", "novelty", "tail_share"]
    selected = [c for c in desired if c in metrics.columns]
    display = pd.DataFrame({METRIC_SHORT.get(c, METRIC_LABEL[c]): [fr_metric(c, v) for v in metrics[c]]
                            for c in selected}, index=metrics.index)
    display.index.name = "Modèle"
    st.dataframe(display, **stretch(),
                 column_config={METRIC_SHORT.get(c, METRIC_LABEL[c]): st.column_config.TextColumn(
                     help=METRIC_HELP.get(c, "")) for c in selected})
    with st.expander("Que mesure chaque colonne ?"):
        for c in selected:
            st.markdown(f"- **{METRIC_LABEL[c]}** — {METRIC_HELP[c]}")
        st.markdown(f"- La **variété des genres** n'est pas affichée : dans ce jeu de données presque "
                    f"tous les livres ont le même genre principal («{NBSP}comics{NBSP}»), elle ne distingue pas les modèles.")
    st.download_button("Télécharger toutes les métriques (CSV)", raw.to_csv().encode("utf-8-sig"),
                       "bookrec_test_metrics.csv", "text/csv")

    plot_choice = st.selectbox("Comparer les modèles selon", selected, format_func=lambda c: METRIC_LABEL[c])
    st.altair_chart(bar_chart(metrics[plot_choice], plot_choice, gate_name), **stretch())

    # how k was chosen
    best = read_json(Path(cfg["paths"]["tables_dir"]) / "best_k.json")
    sweep_path = Path(cfg["paths"]["tables_dir"]) / "k_sweep_val.csv"
    sweep = load_csv(str(sweep_path), sweep_path.stat().st_mtime if sweep_path.exists() else 0.0, index_col=None)
    if best or sweep is not None:
        heading("Validation", "Comment k a été choisi")
        if best:
            st.markdown(f"k{NBSP}={NBSP}**{fr_number(best['k'], 0)}** — règle : plus grand k dont le NDCG@20 de "
                        f"validation reste à moins de {fr_percent(best.get('ndcg_tolerance', 0))} du meilleur. "
                        "Un k plus grand donne plus de poids au contenu, donc plus de pépites et de livres nouveaux.")
        if sweep is not None:
            cols = {"k": "k", "recall": "Rappel@20", "ndcg": "NDCG@20", "cold_recall": "Rappel livres nouveaux",
                    "coverage": "Couverture", "tail_share": "Part de pépites"}
            show = sweep[[c for c in cols if c in sweep.columns]].rename(columns=cols)
            for c in show.columns[1:]:
                key = {"Couverture": "coverage", "Part de pépites": "tail_share"}.get(c, "recall")
                show[c] = [fr_metric(key, v) for v in show[c]]
            show["k"] = [fr_number(v, 0) for v in sweep["k"]]
            st.dataframe(show, hide_index=True, **stretch())
            st.caption("Ces chiffres sont mesurés sur la validation. Sur le test, le rappel est plus bas "
                       "(les livres à retrouver sont plus loin dans le temps), mais couverture et part de "
                       "pépites sont presque identiques : les listes recommandées viennent du même "
                       "entraînement, seuls les livres à retrouver changent.")
    else:
        st.info("Choix de k indisponible : lancez `python scripts/tune_k.py`, puis `python scripts/run_pipeline.py`.")

    st.write("")
    heading("Analyse visuelle", "Figures de l'expérience")
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures = [
        ("Compromis pertinence / découverte selon k", "tradeoff_curve.png"),
        ("Courbe de l'aiguillage", "gate_curve.png"),
        ("Distribution de la popularité (longue traîne)", "long_tail.png"),
        ("Popularité des livres recommandés", "popularity_of_recs.png"),
        ("Comparaison des modèles (toutes métriques)", "baseline_bars.png"),
    ]
    available = [(label, figures_dir / name) for label, name in figures if (figures_dir / name).exists()]
    missing = [name for _, name in figures if not (figures_dir / name).exists()]
    if missing:
        st.info("Figures absentes : " + ", ".join(missing) + ". Elles sont produites par run_pipeline.py "
                "et tune_k.py.")
    for label, path in available:
        with st.expander(label, expanded=(path.name == "tradeoff_curve.png")):
            st.image(str(path), **stretch())


def main() -> None:
    st.set_page_config(
        page_title="BookRec | Au-delà des best-sellers",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        cfg, art = get_artifacts()
    except Exception as err:  # noqa: BLE001
        hero("Le rayon est <em>vide</em> pour l'instant.",
             "Le tableau de bord sera disponible dès que les fichiers du pipeline seront prêts.", compact=True)
        st.error("Impossible de charger les modèles ou les données du projet.")
        if DEBUG:
            st.exception(err)
        else:
            st.caption(str(err))
        st.code("python scripts/run_pipeline.py --force", language="powershell")
        return

    with st.sidebar:
        md(f'<div class="brand">Book<em>Rec</em></div><div class="brandsub">AU-DELÀ DES BEST{NBSP_HYPHEN}SELLERS</div>'
           '<div class="credit">Projet Big Data · Ella &amp; Amine</div>')
        st.divider()
        page = st.radio(
            "Navigation",
            ["Vue d'ensemble", "Lecteur existant", "Nouveau lecteur", "Résultats & analyses"],
            key="bookrec_page",
        )
        st.divider()
        st.markdown("**Jeu de données**")
        st.caption("Goodreads · comics et romans graphiques")
        st.markdown(f"**{fr_number(len(art['books']), 0)}** livres")
        st.markdown(f"**{fr_number(art['train'].shape[0], 0)}** lecteurs")
        st.markdown(f"**k{NBSP}={NBSP}{fr_number(float(art['k']), 0)}** · {MODEL_LABEL['gate'].lower()}")


    if page == "Vue d'ensemble":
        show_dashboard(cfg, art)
    elif page == "Lecteur existant":
        show_existing_reader(cfg, art)
    elif page == "Nouveau lecteur":
        show_new_reader(cfg, art)
    else:
        show_results(cfg, art)


if __name__ == "__main__":
    main()
