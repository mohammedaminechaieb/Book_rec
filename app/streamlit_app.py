"""
==============================================================================
File      : app/streamlit_app.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead) | Amine (co-owner / reviewer)
------------------------------------------------------------------------------
"Le Rayon" - bibliothèque nocturne.

Concept visuel :
  - Hero avec un rayon de livres animé
  - Couvertures réelles (image_url) ou couvertures générées depuis le titre
  - Popularité VISIBLE : jauge Best-seller / Intermédiaire / Pépite
  - Voix du modèle : part collaborative (ALS) vs contenu (TF-IDF) par livre
  - Courbe du gate alpha = n / (n + k) dessinée en SVG
  - Duel ALS vs Hybride avec bilan de découverte

Run:
  streamlit run app/streamlit_app.py
==============================================================================
"""

import hashlib
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow execution from the project root or with the app path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src import recommend  # noqa: E402


MODELS = {
    "Hybride intelligent": "gate",
    "ALS collaboratif": "als",
    "TF-IDF contenu": "content",
    "Popularité": "popularity",
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
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1280px; padding-top:1.4rem; padding-bottom:4rem; }
h1,h2,h3,h4 { font-family:'Fraunces',serif; letter-spacing:-.02em; color:var(--paper); }
h4 { font-weight:600; }
label, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
footer { visibility:hidden; }

/* ---------- Hero ---------- */
.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:6px 28px 6px 28px;
  padding:38px 42px 0; margin-bottom:24px; background:linear-gradient(160deg,#1c1915,#13110e); }
.hero-text { position:relative; z-index:2; max-width:640px; padding-bottom:26px; }
.eyebrow { font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.22em; text-transform:uppercase; color:var(--tail); }
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
.kicker { font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:var(--head); margin-bottom:2px; }

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
.gcover span { font-family:'Fraunces',serif; font-weight:600; font-size:.6rem; line-height:1.15; color:rgba(255,255,255,.95);
  display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical; overflow:hidden; }
.gcover i { display:block; width:60%; height:2px; background:rgba(255,255,255,.55); }
.bbody { flex:1; min-width:0; }
.btop { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.rk { font-family:'Fraunces',serif; font-style:italic; font-size:1.7rem; line-height:1; color:var(--head); }
.bt { font-family:'Fraunces',serif; font-weight:600; font-size:1.02rem; color:var(--paper); }
.tags { margin:7px 0 4px; }
.tag { display:inline-block; font-family:'Space Mono',monospace; font-size:.66rem; padding:3px 8px; margin:0 6px 4px 0;
  border:1px solid var(--line); border-radius:3px; color:#d8cfbd; }
.tag.head { color:var(--head); border-color:var(--head); }
.tag.mid  { color:var(--mid);  border-color:var(--mid); }
.tag.tail { color:var(--tail); border-color:var(--tail); }
.voice { display:flex; height:5px; border-radius:3px; overflow:hidden; margin:6px 0 2px; background:#2b2620; }
.voice b { display:block; height:100%; }
.voice .a { background:var(--head); } .voice .c { background:var(--tail); }
.vlabel { font-family:'Space Mono',monospace; font-size:.62rem; color:var(--muted); letter-spacing:.04em; }
.bdesc { margin:7px 0 0; font-size:.86rem; line-height:1.55; color:#bdb4a3; font-style:italic; }

/* ---------- Jauge de découverte ---------- */
.meter { display:flex; height:22px; border-radius:4px; overflow:hidden; border:1px solid var(--line); margin:8px 0 6px; }
.meter div { display:flex; align-items:center; justify-content:center; font-family:'Space Mono',monospace;
  font-size:.66rem; font-weight:700; color:#12100d; min-width:0; }
.meter .h { background:var(--head); } .meter .m { background:var(--mid); } .meter .t { background:var(--tail); }
.legend { font-family:'Space Mono',monospace; font-size:.64rem; color:var(--muted); }
.legend span { margin-right:14px; }
.legend i { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:5px; }
.verdict { font-family:'Fraunces',serif; font-size:1.15rem; line-height:1.4; border-left:3px solid var(--tail);
  padding:4px 0 4px 16px; margin:14px 0 6px; color:var(--paper); }
.verdict b { color:var(--tail); }

/* ---------- Divers ---------- */
.best { background:var(--card); border:1px solid var(--line); border-radius:4px 14px 4px 14px; padding:14px 16px; }
.best .l { font-family:'Space Mono',monospace; font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.best .v { font-family:'Fraunces',serif; font-size:1.25rem; color:var(--paper); margin-top:3px; }
.best .s { font-family:'Space Mono',monospace; font-size:.74rem; color:var(--tail); }
.brand { font-family:'Fraunces',serif; font-weight:800; font-size:1.5rem; letter-spacing:-.02em; }
.brand em { color:var(--head); font-weight:600; }
.brandsub { font-family:'Space Mono',monospace; font-size:.62rem; letter-spacing:.2em; color:var(--muted); }
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:4px 14px 4px 14px; padding:14px 17px; }
[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.06em; }
[data-testid="stMetricValue"] { color:var(--paper); font-family:'Fraunces',serif; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
.svgwrap svg { width:100%; height:auto; display:block; }
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
    st.error(f"{message} : {err}")
    if st.session_state.get("debug_mode"):
        st.exception(err)


def heading(kicker: str, title: str, note: str = "") -> None:
    md(f'<div class="kicker">{esc(kicker)}</div>')
    st.subheader(title)
    if note:
        st.caption(note)


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


def pop_class(value) -> str | None:
    p = str(value or "").lower()
    if "head" in p:
        return "head"
    if "tail" in p or "niche" in p or "long" in p:
        return "tail"
    if "mid" in p:
        return "mid"
    return None


POP_LABEL = {"head": "Best-seller", "mid": "Intermédiaire", "tail": "Pépite"}


def meter_counts(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or "popularity" not in df.columns:
        return None
    classes = [pop_class(v) for v in df["popularity"]]
    classes = [c for c in classes if c]
    if not classes:
        return None
    total = len(classes)
    return {k: classes.count(k) / total for k in ("head", "mid", "tail")}


def meter_html(shares: dict) -> str:
    cells = ""
    for key, css in (("head", "h"), ("mid", "m"), ("tail", "t")):
        pct = shares[key] * 100
        if pct > 0:
            label = f"{pct:.0f} %" if pct >= 12 else ""
            cells += f'<div class="{css}" style="width:{pct:.1f}%">{label}</div>'
    legend = (
        '<div class="legend">'
        '<span><i style="background:var(--head)"></i>Best-sellers</span>'
        '<span><i style="background:var(--mid)"></i>Intermédiaires</span>'
        '<span><i style="background:var(--tail)"></i>Pépites</span></div>'
    )
    return f'<div class="meter">{cells}</div>{legend}'


def metric_str(value, kind: str = "percent") -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "—"
        return f"{value * 100:.1f} %" if kind == "percent" else f"{value:.2f}"
    except (TypeError, ValueError):
        return "—"


def gate_svg(k: float) -> str:
    k = max(float(k), 1e-9)
    W, H, L, B, T, R = 640, 250, 50, 38, 16, 18
    nmax = 6 * k
    pts = []
    for i in range(61):
        n = nmax * i / 60
        a = n / (n + k)
        x = L + (W - L - R) * i / 60
        y = H - B - (H - B - T) * a
        pts.append(f"{x:.1f},{y:.1f}")
    line = "M" + " L".join(pts)
    area = f"{line} L{W - R},{H - B} L{L},{H - B} Z"
    kx = L + (W - L - R) / 6
    ky = H - B - (H - B - T) * 0.5
    grid = ""
    for a in (0.25, 0.5, 0.75, 1.0):
        y = H - B - (H - B - T) * a
        grid += (
            f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#363026" stroke-dasharray="3 5"/>'
            f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#a39a8a" font-size="11" '
            f'font-family="Space Mono, monospace">{a:.2f}</text>'
        )
    xticks = ""
    for mult in (0, 1, 3, 6):
        x = L + (W - L - R) * mult / 6
        xticks += (
            f'<text x="{x:.1f}" y="{H - B + 18}" text-anchor="middle" fill="#a39a8a" font-size="11" '
            f'font-family="Space Mono, monospace">{mult * k:g}</text>'
        )
    return (
        f'<div class="svgwrap"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Courbe du gate alpha">'
        f'{grid}{xticks}'
        f'<path d="{area}" fill="#4fd1b8" opacity=".12"/>'
        f'<path d="{line}" fill="none" stroke="#4fd1b8" stroke-width="3" stroke-linecap="round"/>'
        f'<line x1="{kx:.1f}" x2="{kx:.1f}" y1="{ky:.1f}" y2="{H - B}" stroke="#ff5a36" stroke-dasharray="4 4"/>'
        f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="6" fill="#ff5a36"/>'
        f'<text x="{kx + 12:.1f}" y="{ky + 18:.1f}" fill="#ff5a36" font-size="12" '
        f'font-family="Space Mono, monospace">n = k → α = 0,5</text>'
        f'<text x="{(L + W - R) / 2:.1f}" y="{H - 4}" text-anchor="middle" fill="#a39a8a" font-size="11" '
        f'font-family="Space Mono, monospace">interactions du livre (n)</text>'
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
def load_metrics(path: str, mtime: float):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0)


@st.cache_resource(show_spinner=False)
def build_cover_map(_books: pd.DataFrame) -> dict:
    """book_id -> URL de couverture réelle (si le dataset en contient)."""
    for col in ("image_url", "cover_url", "image"):
        if col in _books.columns and "book_id" in _books.columns:
            out = {}
            for bid, url in zip(_books["book_id"].astype(str), _books[col]):
                u = str(url or "")
                if u.startswith("http") and "nophoto" not in u:
                    out[bid] = u
            return out
    return {}


def generated_cover(title: str) -> str:
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16)
    hue1 = h % 360
    hue2 = (hue1 + 35 + (h >> 8) % 70) % 360
    bg = f"linear-gradient(155deg,hsl({hue1},58%,38%),hsl({hue2},62%,22%))"
    return f'<div class="gcover" style="background:{bg}"><span>{esc(title)}</span><i></i></div>'


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------
def book_card(row: pd.Series, covers: dict) -> None:
    raw_title = str(row.get("title", "Livre sans titre"))
    title = esc(raw_title)
    explanation = esc(row.get("explanation", ""))
    rank = esc(row.get("rank", ""))

    url = covers.get(str(row.get("book_id", "")))
    cover = f'<img class="cover" src="{esc(url)}" alt="">' if url else generated_cover(raw_title)

    try:
        score_tag = f'<span class="tag">score {float(row["score"]):.3f}</span>'
    except (ValueError, KeyError, TypeError):
        score_tag = ""

    pc = pop_class(row.get("popularity"))
    pop_tag = f'<span class="tag {pc}">{POP_LABEL[pc]}</span>' if pc else ""

    voice = ""
    alpha = row.get("alpha", np.nan)
    try:
        if pd.notna(alpha):
            a = min(max(float(alpha), 0.0), 1.0) * 100
            voice = (
                f'<div class="voice"><b class="a" style="width:{a:.1f}%"></b>'
                f'<b class="c" style="width:{100 - a:.1f}%"></b></div>'
                f'<div class="vlabel">ALS {a:.0f} % · contenu {100 - a:.0f} %</div>'
            )
    except (TypeError, ValueError):
        pass

    md(
        f'<div class="book">{cover}<div class="bbody">'
        f'<div class="btop"><span class="rk">{rank}</span><span class="bt">{title}</span></div>'
        f'<div class="tags">{pop_tag}{score_tag}</div>{voice}'
        f'<div class="bdesc">{explanation}</div></div></div>'
    )


def show_recommendations(df: pd.DataFrame, title: str, key: str, covers: dict, compact: bool = False) -> None:
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("Aucune recommandation disponible pour cette sélection.")
        return
    shares = meter_counts(df)
    if shares:
        md(meter_html(shares))
    st.caption(f"{len(df)} livres recommandés · les livres déjà lus sont exclus")
    for _, row in df.head(5 if compact else len(df)).iterrows():
        book_card(row, covers)
    if compact and len(df) > 5:
        with st.expander(f"Afficher les {len(df) - 5} suivants"):
            for _, row in df.iloc[5:].iterrows():
                book_card(row, covers)
    st.download_button(
        "Exporter (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bookrec_{key}.csv",
        mime="text/csv",
        key=f"export_{key}",
        **stretch(),
    )


def discovery_verdict(als_recs: pd.DataFrame, gate_recs: pd.DataFrame) -> None:
    a, g = meter_counts(als_recs), meter_counts(gate_recs)
    if not a or not g:
        return
    ta, tg = a["tail"] * 100, g["tail"] * 100
    ha, hg = a["head"] * 100, g["head"] * 100
    delta = tg - ta
    if delta > 0.5:
        text = (f"L'hybride propose <b>{tg:.0f} % de pépites</b> contre {ta:.0f} % pour ALS, "
                f"et {hg:.0f} % de best-sellers au lieu de {ha:.0f} %.")
    elif delta < -0.5:
        text = (f"Pour ce lecteur, l'hybride propose <b>moins de pépites</b> qu'ALS "
                f"({tg:.0f} % contre {ta:.0f} %) : le profil est très ancré dans des livres populaires.")
    else:
        text = f"Pour ce lecteur, les deux modèles proposent la même part de pépites (<b>{tg:.0f} %</b>)."
    md(f'<div class="verdict">{text}</div>')


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def show_dashboard(cfg: dict, art: dict) -> None:
    hero("Au-delà des <em>best-sellers</em>.", "Diversifier les découvertes, sans sacrifier la pertinence.")
    a, b, c, d = st.columns(4)
    a.metric("LIVRES", f"{len(art['books']):,}".replace(",", " "))
    b.metric("LECTEURS", f"{art['train'].shape[0]:,}".replace(",", " "))
    c.metric("INTERACTIONS", f"{art['train'].nnz:,}".replace(",", " "))
    d.metric("GATE k", f"{float(art['k']):g}")

    st.write("")
    heading("Le concept", "Trois chapitres, un moteur plus équitable",
            "Chaque modèle apporte une voix ; le gate décide laquelle écouter.")
    cols = st.columns(3)
    concepts = [
        ("I", "ALS collaboratif", "S'appuie sur les goûts partagés entre lecteurs : précis, mais attiré par les best-sellers."),
        ("II", "TF-IDF contenu", "Lit les descriptions et les tags pour rapprocher des livres similaires, même peu connus."),
        ("III", "Hybrid Gate", "Écoute ALS quand le livre a beaucoup d'interactions, le contenu quand il en a peu."),
    ]
    for col, (num, label, desc) in zip(cols, concepts):
        with col:
            md(f'<div class="chapter"><div class="num">{num}</div><h3>{esc(label)}</h3><p>{esc(desc)}</p></div>')

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        md('<div class="panel"><div class="kicker">Le gate en une courbe</div>'
           '<h4>α = n / (n + k)</h4>'
           '<p>Plus un livre a d\'interactions (n), plus la voix collaborative pèse. '
           'Les livres rares restent portés par le contenu.</p></div>')
        md(gate_svg(float(art["k"])))
    with right:
        md('<div class="panel"><div class="kicker">Lire la jauge</div><h4>La popularité, enfin visible</h4>'
           '<p>Chaque liste est résumée par une jauge : part de best-sellers, de livres intermédiaires '
           'et de pépites de la longue traîne.</p></div>')
        md(meter_html({"head": 0.55, "mid": 0.25, "tail": 0.20}))
        st.caption("Exemple illustratif d'une liste dominée par les best-sellers.")
        st.info("**Lecteur existant** : comparez ALS et hybride. **Nouveau lecteur** : choisissez 3 livres, "
                "le contenu fait le reste.")
    st.caption("Données de démonstration : échantillon Goodreads comics & graphic novels. "
               "Les résultats ne constituent pas une mesure d'équité individuelle.")


def show_existing_reader(cfg: dict, art: dict) -> None:
    hero("Un lecteur, <em>deux</em> regards.", "Les lectures passées alimentent les modèles ; les explications les rendent lisibles.", compact=True)
    covers = build_cover_map(art["books"])
    demo_users = [str(x) for x in art.get("demo_users", [])]
    if not demo_users:
        st.warning("Aucun lecteur de démonstration disponible.")
        return

    opts, controls = st.columns([2.1, 1])
    with opts:
        user_id = st.selectbox("Identifiant du lecteur", demo_users, key="existing_user")
    with controls:
        top_n = st.slider("Nombre de recommandations", 5, 20, int(min(10, cfg["app"]["default_top_n"])), key="existing_top")

    model_label = st.radio("Modèle de recommandation", list(MODELS), horizontal=True)
    compare = st.checkbox("Comparer l'hybride avec ALS", value=True)

    user_idx = art["user2idx"][str(user_id)]
    seen_idx = art["train"][user_idx].indices
    book_table = art["books"]
    seen = book_table.iloc[seen_idx].copy() if len(seen_idx) else book_table.iloc[:0].copy()

    m1, m2 = st.columns([1, 2.2], gap="large")
    m1.metric("LIVRES DÉJÀ LUS", len(seen))
    with m2:
        shares = meter_counts(seen)
        if shares:
            md('<div class="panel"><div class="kicker">Profil de lecture</div>'
               '<h4>Ce lecteur lit surtout…</h4></div>')
            md(meter_html(shares))

    with st.expander("Voir les livres déjà lus / appréciés", expanded=False):
        if seen.empty:
            st.info("Pas de livres dans l'historique d'entraînement.")
        else:
            available = [c for c in ["title", "book_id"] if c in seen.columns]
            st.dataframe(seen[available].head(30).reset_index(drop=True), hide_index=True, **stretch())
            if len(seen) > 30:
                st.caption("30 premiers livres affichés.")

    als_recs = gate_recs = main_recs = None
    try:
        with st.spinner("Calcul des scores de recommandation…"):
            if compare:
                als_recs = recommend.recommend_for_user(user_id, art, model_name="als", top_n=top_n)
                gate_recs = recommend.recommend_for_user(user_id, art, model_name="gate", top_n=top_n)
            selected = MODELS[model_label]
            if compare and selected == "als":
                main_recs = als_recs
            elif compare and selected == "gate":
                main_recs = gate_recs
            else:
                main_recs = recommend.recommend_for_user(user_id, art, model_name=selected, top_n=top_n)
    except Exception as err:  # noqa: BLE001
        show_error(err, "Erreur de recommandation")
        return

    st.write("")
    if compare:
        heading("Le duel", "ALS classique contre hybride adaptatif",
                "Même lecteur, même nombre de livres : regardez ce qui change dans la jauge.")
        discovery_verdict(als_recs, gate_recs)
        left, right = st.columns(2, gap="large")
        with left:
            show_recommendations(als_recs, "ALS collaboratif", f"cmp_als_{user_id}", covers, compact=True)
        with right:
            show_recommendations(gate_recs, "Hybrid Gate", f"cmp_gate_{user_id}", covers, compact=True)
        # La liste principale n'est affichée à part que si elle n'est pas déjà dans le duel.
        if MODELS[model_label] in ("content", "popularity"):
            st.divider()
            show_recommendations(main_recs, f"Top {len(main_recs)} · {model_label}",
                                 f"main_{MODELS[model_label]}_{user_id}", covers)
    else:
        show_recommendations(main_recs, f"Top {len(main_recs)} · {model_label}",
                             f"main_{MODELS[model_label]}_{user_id}", covers)


def show_new_reader(cfg: dict, art: dict) -> None:
    hero("Votre <em>première</em> étagère.", "Pas d'historique ? Trois livres suffisent à dessiner votre profil de lecture.", compact=True)
    heading("Démarrage à froid", "Commencer une nouvelle histoire",
            "Sélectionnez au moins trois livres que vous appréciez. Aucun identifiant Goodreads nécessaire.")
    books = art["books"].reset_index(drop=True)
    if "book_id" not in books or "title" not in books:
        st.error("Métadonnées des livres incomplètes : book_id et title sont nécessaires.")
        return
    covers = build_cover_map(art["books"])
    catalog = books[["book_id", "title"]].copy()
    catalog["book_id"] = catalog["book_id"].astype(str)
    catalog["title"] = catalog["title"].fillna("Sans titre").astype(str)
    labels = [f"{row.title} · ID {row.book_id}" for row in catalog.itertuples(index=False)]
    label_to_id = dict(zip(labels, catalog["book_id"]))

    picked = st.multiselect(
        "Vos livres préférés",
        options=labels,
        placeholder="Rechercher un titre dans le catalogue…",
        max_selections=12,
        key="cold_book_picker",
    )
    st.caption(f"{len(picked)} livre(s) sélectionné(s) · minimum recommandé : 3")

    if picked:
        shelf = "".join(
            generated_cover(lbl.rsplit(" · ID ", 1)[0]).replace('class="gcover"', 'class="gcover" ')
            for lbl in picked
        )
        md(f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 14px">{shelf}</div>')

    top_n = st.slider("Nombre de recommandations", 5, 20, int(min(10, cfg["app"]["default_top_n"])), key="new_top")
    if len(picked) < 3:
        st.info("Choisissez au moins trois livres pour activer les recommandations.")
        return
    selected_ids = [label_to_id[label] for label in picked]
    try:
        with st.spinner("Construction de votre profil de lecture…"):
            recs = recommend.recommend_for_new_reader(selected_ids, art, top_n=top_n)
    except Exception as err:  # noqa: BLE001
        show_error(err, "Erreur pendant les recommandations")
        return
    st.success("Profil de lecture créé à partir des livres sélectionnés (TF-IDF uniquement).")
    show_recommendations(recs, f"Votre sélection · Top {len(recs)}", "new_reader", covers)
    st.caption("Cold start : aucun facteur collaboratif ALS n'est disponible pour ce nouveau lecteur, "
               "donc alpha vaut 0.")


def show_results(cfg: dict, art: dict) -> None:
    hero("Ce que disent <em>les chiffres</em>.", "Mesurer le compromis entre pertinence et découverte.", compact=True)
    heading("Évaluation expérimentale", "Les modèles, côte à côte",
            "Scores obtenus sur l'échantillon d'essai (results/tables/test_metrics.csv).")
    table = Path(cfg["paths"]["tables_dir"]) / "test_metrics.csv"
    metrics = load_metrics(str(table), table.stat().st_mtime if table.exists() else 0.0)
    if metrics is None or metrics.empty:
        st.warning("Le fichier test_metrics.csv n'est pas disponible. Exécutez le pipeline pour produire les résultats.")
        return

    desired = ["all/recall@20", "all/ndcg@20", "coverage", "novelty", "tail_share", "genre_variety"]
    selected = [c for c in desired if c in metrics.columns]
    if not selected:
        st.dataframe(metrics, **stretch())
        return

    # Meilleur modèle par métrique
    cols = st.columns(min(len(selected), 3))
    for i, metric in enumerate(selected):
        series = pd.to_numeric(metrics[metric], errors="coerce").dropna()
        if series.empty:
            continue
        with cols[i % len(cols)]:
            md(f'<div class="best"><div class="l">Meilleur · {esc(metric)}</div>'
               f'<div class="v">{esc(series.idxmax())}</div>'
               f'<div class="s">{series.max():.4f}</div></div>')
            st.write("")

    display = metrics[selected].copy()
    st.dataframe(display.style.format(precision=4), **stretch())
    st.download_button("Télécharger les métriques (CSV)", metrics.to_csv().encode("utf-8-sig"),
                       "bookrec_test_metrics.csv", "text/csv")
    plot_choice = st.selectbox("Comparer les modèles selon", selected)
    st.bar_chart(metrics[plot_choice], **stretch())

    st.write("")
    heading("Analyse visuelle", "Figures de l'expérience")
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures = [
        ("Distribution longue traîne", "long_tail.png"),
        ("Comparaison des modèles", "baseline_bars.png"),
        ("Popularité des recommandations", "popularity_of_recs.png"),
        ("Compromis rappel / couverture", "tradeoff.png"),
        ("Courbe du gate", "gate.png"),
    ]
    available = [(label, figures_dir / name) for label, name in figures if (figures_dir / name).exists()]
    if not available:
        st.info("Aucune figure enregistrée pour le moment.")
    else:
        for label, path in available:
            with st.expander(label, expanded=(label == "Comparaison des modèles")):
                st.image(str(path), **stretch())
    st.caption("Important : un meilleur score de couverture ou de nouveauté ne garantit pas, "
               "à lui seul, une meilleure pertinence pour chaque utilisateur.")


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
        hero("Le rayon est <em>vide</em> pour l'instant.", "Le tableau de bord sera disponible dès que les artefacts du pipeline seront prêts.", compact=True)
        st.error("Impossible de charger les modèles ou les données du projet.")
        st.exception(err)
        st.code("python scripts/run_pipeline.py --users 5000 --force", language="powershell")
        return

    with st.sidebar:
        md('<div class="brand">Book<em>Rec</em></div><div class="brandsub">AU-DELÀ DES BEST-SELLERS</div>')
        st.divider()
        page = st.radio(
            "Navigation",
            ["Vue d'ensemble", "Lecteur existant", "Nouveau lecteur", "Résultats & analyses"],
            key="bookrec_page",
        )
        st.divider()
        st.markdown("**Dataset**")
        st.caption("Goodreads · Comics & Graphic Novels")
        st.markdown(f"**{len(art['books']):,}** livres".replace(",", " "))
        st.markdown(f"**{art['train'].shape[0]:,}** lecteurs".replace(",", " "))
        st.markdown(f"**k = {art['k']:g}** · hybride adaptatif")
        st.divider()
        st.checkbox("Mode debug", key="debug_mode", help="Affiche les traces d'erreur complètes.")
        st.caption("Projet Big Data · Ella & Amine")

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