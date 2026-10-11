"""
SentinelPay: Real-Time Financial Fraud & Anomaly Intelligence Platform
=======================================================================
University of Mumbai — BDA Semester 7 Project
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from algorithms import BloomFilter, DGIMAlgorithm, BaselineProfile
from pipeline import TransactionStream, extract_feature_vector, build_synthetic_transaction

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SentinelPay | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "#0E1117"
BG_CARD   = "#1A1F2E"
BG_CARD2  = "#141824"
BORDER    = "#2D3748"
TXT       = "#E2E8F0"
TXT_MUTED = "#94A3B8"
EMERALD   = "#10B981"
AMBER     = "#F59E0B"
CRIMSON   = "#EF4444"
BLUE      = "#3B82F6"
PURPLE    = "#8B5CF6"

# ═════════════════════════════════════════════════════════════════════════════
#  CSS — targets real Streamlit 1.32+ DOM structure
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── Root background & text ── */
.stApp {{
    background-color: {BG};
    color: {TXT};
}}
.stApp * {{
    color: inherit;
}}

/* ── Main block container ── */
section.main > div.block-container {{
    padding: 1rem 1.8rem 2rem 1.8rem;
    max-width: 100% !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {TXT} !important;
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {{
    color: {TXT_MUTED} !important;
}}

/* ── Sidebar slider labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider p {{
    color: {TXT} !important;
    font-size: 0.82rem !important;
}}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    background: {BLUE};
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    width: 100%;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: #2563EB;
    color: #ffffff !important;
}}

/* ── All regular text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stText, p, span, label {{
    color: {TXT};
}}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 {{
    color: {TXT} !important;
}}

/* ── st.info / success / warning / error boxes ── */
[data-testid="stNotification"],
div[class*="stAlert"] {{
    background: {BG_CARD} !important;
    border-radius: 10px !important;
    color: {TXT} !important;
}}
div[data-testid="stAlert"] p {{
    color: {TXT} !important;
}}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {{
    background: {BG_CARD};
    border-radius: 10px;
    border: 1px solid {BORDER};
}}
[data-testid="stDataFrame"] * {{
    color: {TXT} !important;
    background-color: transparent;
}}

/* ── st.metric ── */
[data-testid="metric-container"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.7rem 1rem;
}}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
    color: {TXT_MUTED} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] *  {{
    color: {TXT} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {BG_CARD};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    padding: 8px 16px;
    color: {TXT_MUTED} !important;
    font-size: 0.84rem;
    font-weight: 600;
    border: none;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{
    background: {BORDER};
    color: {TXT} !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: {BLUE}33 !important;
    color: {BLUE} !important;
}}
[data-testid="stTabs"] [aria-selected="true"] p {{
    color: {BLUE} !important;
}}
[data-testid="stTabPanel"] {{
    background: transparent;
    padding-top: 1rem;
}}

/* ── Selectbox / text-input / number-input ── */
[data-testid="stSelectbox"] select,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background: {BG_CARD2} !important;
    color: {TXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {{
    color: {TXT_MUTED} !important;
    font-size: 0.82rem !important;
}}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label p {{
    color: {TXT} !important;
}}

/* ── Form submit button ── */
[data-testid="stForm"] .stButton > button {{
    background: {BLUE};
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    width: 100%;
}}
[data-testid="stForm"] .stButton > button:hover {{
    background: #2563EB;
}}

/* ── Divider ── */
hr {{
    border-color: {BORDER} !important;
    margin: 0.8rem 0;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] p {{
    color: {TXT_MUTED} !important;
}}

/* ── KPI card (custom HTML) ── */
.sp-kpi-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    height: 100%;
}}
.sp-kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}}
.sp-kpi-label {{
    font-size: 0.72rem;
    color: {TXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 5px;
    font-weight: 600;
}}
.sp-kpi-sub {{
    font-size: 0.78rem;
    margin-top: 5px;
    font-weight: 500;
}}

/* ── Section header ── */
.sp-section {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: {TXT_MUTED};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 5px;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}}

/* ── Platform header card ── */
.sp-header {{
    background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.sp-logo {{
    font-size: 1.65rem;
    font-weight: 900;
    background: linear-gradient(90deg, {EMERALD}, {BLUE});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.sp-sub {{
    font-size: 0.75rem;
    color: {TXT_MUTED};
    margin-top: 3px;
}}
.sp-badge-live {{
    display: inline-block;
    background: {CRIMSON}22;
    border: 1px solid {CRIMSON}88;
    color: {CRIMSON};
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-top: 4px;
}}
.sp-badge-idle {{
    display: inline-block;
    background: {TXT_MUTED}22;
    border: 1px solid {TXT_MUTED}66;
    color: {TXT_MUTED};
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin-top: 4px;
}}

/* ── Algo description card ── */
.sp-algo-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {BLUE};
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.9rem;
    color: {TXT};
}}
.sp-algo-card h4 {{
    margin: 0 0 6px 0;
    font-size: 0.93rem;
    font-weight: 700;
    color: {BLUE};
}}
.sp-algo-card p {{
    margin: 0;
    font-size: 0.82rem;
    color: {TXT_MUTED};
    line-height: 1.65;
}}
.sp-algo-card b {{ color: {TXT}; }}

/* ── Verdict banner ── */
.sp-verdict {{
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 700;
}}

/* ── Footer ── */
.sp-footer {{
    text-align: center;
    font-size: 0.7rem;
    color: {TXT_MUTED};
    border-top: 1px solid {BORDER};
    padding-top: 1rem;
    margin-top: 1.5rem;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
def _init_state() -> None:
    defaults = {
        "bloom": None, "dgim": None, "baseline": None,
        "stream": None, "stream_iter": None,
        "tx_log": [], "distance_log": [],
        "total_streamed": 0, "total_flagged": 0,
        "throughput_ts": [], "running": False,
        "batch_size": 50, "window_size": 1000,
        "max_rows": 3000,
        "anomaly_threshold_cos": 0.35,
        "anomaly_threshold_maha": 4.5,
        "initialized": False, "blacklist_size": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM INIT
# ═════════════════════════════════════════════════════════════════════════════
def init_algorithms() -> None:
    blacklist_path = str(ROOT / "data" / "blacklisted_ips.json")
    bf = BloomFilter(capacity=10_000, fp_rate=0.01)
    st.session_state.blacklist_size = bf.load_blacklist(blacklist_path)
    st.session_state.bloom = bf
    st.session_state.dgim = DGIMAlgorithm(window_size=st.session_state.window_size)

    baseline = BaselineProfile(n_features=3)
    for tx in TransactionStream().iter_transactions(max_rows=300):
        baseline.update(extract_feature_vector(tx))
    st.session_state.baseline = baseline

    st.session_state.stream = TransactionStream()
    st.session_state.stream_iter = st.session_state.stream.iter_transactions(
        max_rows=st.session_state.max_rows
    )
    st.session_state.tx_log = []
    st.session_state.distance_log = []
    st.session_state.total_streamed = 0
    st.session_state.total_flagged = 0
    st.session_state.initialized = True

# ═════════════════════════════════════════════════════════════════════════════
#  TRANSACTION PROCESSING
# ═════════════════════════════════════════════════════════════════════════════
def process_transaction(tx: dict) -> dict:
    bloom: BloomFilter    = st.session_state.bloom
    dgim:  DGIMAlgorithm  = st.session_state.dgim
    base:  BaselineProfile = st.session_state.baseline

    fv = extract_feature_vector(tx)
    base.update(fv)

    is_blacklisted = bloom.contains(tx["ip_address"]) or bloom.contains(tx["nameOrig"])
    dgim.add_transaction(tx["amount"])
    dgim_surge = dgim.is_surge

    if base.ready(30):
        cos_d  = base.cosine_dist(fv)
        maha_d = base.mahalanobis_dist(fv)
    else:
        cos_d = maha_d = 0.0

    cos_anomaly  = cos_d  > st.session_state.anomaly_threshold_cos
    maha_anomaly = maha_d > st.session_state.anomaly_threshold_maha
    is_flagged   = is_blacklisted or cos_anomaly or maha_anomaly

    tags = []
    if is_blacklisted:              tags.append("⛔ BLACKLISTED")
    if dgim_surge:                  tags.append("⚠️ DGIM SURGE")
    if cos_anomaly or maha_anomaly: tags.append("🚨 ANOMALY")
    if not tags:                    tags.append("✅ CLEAN")

    return {
        **tx,
        "is_blacklisted": is_blacklisted,
        "dgim_surge":     dgim_surge,
        "dgim_estimate":  dgim.estimate_count(),
        "cosine_dist":    round(cos_d, 4),
        "mahalanobis":    round(maha_d, 4),
        "cos_anomaly":    cos_anomaly,
        "maha_anomaly":   maha_anomaly,
        "is_flagged":     is_flagged,
        "tags":           " | ".join(tags),
        "ts":             datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }


def process_batch(n: int) -> int:
    if not st.session_state.stream_iter:
        return 0
    done = 0
    for _ in range(n):
        try:
            tx = next(st.session_state.stream_iter)
        except StopIteration:
            st.session_state.running = False
            break
        r = process_transaction(tx)
        st.session_state.tx_log.append(r)
        st.session_state.total_streamed += 1
        if r["is_flagged"]:
            st.session_state.total_flagged += 1
        st.session_state.distance_log.append({
            "step": r["step"], "cosine": r["cosine_dist"],
            "mahalanobis": r["mahalanobis"], "is_fraud": r["isFraud"],
            "is_flagged": r["is_flagged"], "type": r["type"],
        })
        done += 1

    if len(st.session_state.tx_log) > 500:
        st.session_state.tx_log = st.session_state.tx_log[-500:]
    if len(st.session_state.distance_log) > 2000:
        st.session_state.distance_log = st.session_state.distance_log[-2000:]

    st.session_state.throughput_ts.append((time.time(), done))
    if len(st.session_state.throughput_ts) > 20:
        st.session_state.throughput_ts.pop(0)
    return done


def calc_tps() -> float:
    ts = st.session_state.throughput_ts
    if len(ts) < 2:
        return 0.0
    elapsed = ts[-1][0] - ts[0][0]
    total   = sum(t[1] for t in ts)
    return round(total / elapsed, 1) if elapsed > 0 else 0.0

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════
_LAYOUT = dict(
    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD2,
    font=dict(color=TXT, family="Inter,sans-serif", size=11),
    margin=dict(l=44, r=20, t=44, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TXT_MUTED),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TXT_MUTED),
)


def chart_bloom(bloom: BloomFilter) -> go.Figure:
    bits = bloom.bit_density_vector
    colors = [EMERALD if b else "#2A3040" for b in bits]
    fig = go.Figure(go.Bar(
        x=list(range(len(bits))), y=bits.tolist(),
        marker_color=colors,
        hovertemplate="Bit %{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text="Bloom Filter — Bit-Array Density", font=dict(size=13, color=TXT)),
        xaxis_title="Bit Position (sampled to 1024)", yaxis_title="Bit Value",
        showlegend=False, height=230)
    return fig


def chart_dgim(dgim: DGIMAlgorithm) -> go.Figure:
    buckets = dgim.get_bucket_tree()
    if not buckets:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT, title=dict(text="DGIM Bucket Tree (empty)", font=dict(color=TXT)), height=230)
        return fig
    sizes = [b["size"] for b in buckets]
    tss   = [b["timestamp"] for b in buckets]
    lsz   = [b["log2_size"] for b in buckets]
    fig = go.Figure(go.Scatter(
        x=tss, y=sizes, mode="markers+lines",
        marker=dict(size=[max(9, l * 5) for l in lsz],
                    color=lsz, colorscale="Plasma", showscale=True,
                    colorbar=dict(title="log₂", thickness=10, tickfont=dict(color=TXT_MUTED))),
        line=dict(color=BORDER, width=1, dash="dot"),
        text=[f"size={s} ts={t}" for s, t in zip(sizes, tss)],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text="DGIM Bucket Tree — newest → oldest", font=dict(size=13, color=TXT)),
        xaxis_title="Stream Timestamp", yaxis_title="Bucket Size (2^i)",
        height=270)
    fig.update_yaxes(type="log")
    return fig


def chart_scatter(dist_log: list) -> go.Figure:
    fig = go.Figure()
    if not dist_log:
        fig.update_layout(**_LAYOUT, title=dict(text="Awaiting stream data…", font=dict(color=TXT)), height=420)
        return fig
    df = pd.DataFrame(dist_log[-500:])
    for label, mask, color, symbol, size in [
        ("Normal",  df["is_fraud"] == 0, EMERALD,  "circle", 5),
        ("Fraud",   df["is_fraud"] == 1, CRIMSON,  "x",      9),
    ]:
        sub = df[mask]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["cosine"], y=sub["mahalanobis"],
                mode="markers", name=label,
                marker=dict(color=color, size=size, symbol=symbol, opacity=0.75),
                hovertemplate=f"Cosine: %{{x:.3f}}<br>Mahal: %{{y:.2f}}<extra>{label}</extra>",
            ))
    fig.add_vline(x=st.session_state.anomaly_threshold_cos,
                  line=dict(color=AMBER, width=1.5, dash="dash"),
                  annotation_text=f"cos τ={st.session_state.anomaly_threshold_cos}",
                  annotation_font_color=AMBER)
    fig.add_hline(y=st.session_state.anomaly_threshold_maha,
                  line=dict(color=AMBER, width=1.5, dash="dash"),
                  annotation_text=f"maha τ={st.session_state.anomaly_threshold_maha}",
                  annotation_font_color=AMBER)
    fig.update_layout(**_LAYOUT,
        title=dict(text="Cosine vs Mahalanobis — last 500 transactions", font=dict(size=13, color=TXT)),
        xaxis_title="Cosine Distance", yaxis_title="Mahalanobis Distance",
        height=420, legend=dict(x=0.01, y=0.99,
            bgcolor=BG_CARD, bordercolor=BORDER, borderwidth=1,
            font=dict(color=TXT)))
    return fig


def chart_timeline(tx_log: list) -> go.Figure:
    fig = go.Figure()
    if not tx_log:
        fig.update_layout(**_LAYOUT, title=dict(text="Awaiting stream data…", font=dict(color=TXT)), height=230)
        return fig
    df = pd.DataFrame(tx_log[-300:])
    colors = [CRIMSON if f else EMERALD for f in df["is_flagged"].tolist()]
    fig.add_trace(go.Bar(
        x=list(range(len(df))), y=df["amount"].tolist(),
        marker_color=colors,
        hovertemplate="$%{y:,.0f}<br>%{customdata}<extra></extra>",
        customdata=df["tags"].tolist(),
    ))
    fig.update_layout(**_LAYOUT,
        title=dict(text="Amount Timeline — red=flagged  green=clean", font=dict(size=13, color=TXT)),
        xaxis_title="Transaction Index", yaxis_title="Amount (USD)",
        height=230, showlegend=False)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div style="font-size:1.2rem;font-weight:900;color:{EMERALD};'
        f'letter-spacing:-0.3px;margin-bottom:2px;">🛡️ SentinelPay</div>'
        f'<div style="font-size:0.7rem;color:{TXT_MUTED};margin-bottom:1.2rem;">'
        f'Real-Time Fraud Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<p style="font-size:0.8rem;font-weight:700;color:{TXT};margin-bottom:6px;">⚙️ Stream Config</p>', unsafe_allow_html=True)
    st.session_state.max_rows    = st.slider("Max Rows",         500,  10000, 3000, 500)
    st.session_state.batch_size  = st.slider("Batch / Refresh",   10,    200,   50,  10)
    st.session_state.window_size = st.slider("DGIM Window (N)",  100,   5000, 1000, 100)

    st.markdown(f'<p style="font-size:0.8rem;font-weight:700;color:{TXT};margin:10px 0 6px 0;">🎯 Anomaly Thresholds</p>', unsafe_allow_html=True)
    st.session_state.anomaly_threshold_cos  = st.slider("Cosine τ",     0.05, 1.0,  0.35, 0.05)
    st.session_state.anomaly_threshold_maha = st.slider("Mahalanobis τ", 1.0, 20.0,  4.5,  0.5)

    st.divider()
    c_a, c_b = st.columns(2)
    with c_a:
        if st.button("▶ Init", use_container_width=True):
            with st.spinner("Bootstrapping…"):
                init_algorithms()
            st.success("✓ Ready!")
    with c_b:
        lbl = "⏸ Pause" if st.session_state.running else "▶ Stream"
        if st.button(lbl, use_container_width=True):
            if not st.session_state.initialized:
                st.warning("Init first.")
            else:
                st.session_state.running = not st.session_state.running

    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init_state()
        st.rerun()

    st.divider()
    st.markdown(
        f'<div style="font-size:0.75rem;color:{TXT_MUTED};line-height:1.8;">'
        f'<span style="color:{TXT};font-weight:700;">Algorithms</span><br>'
        f'<span style="color:{BLUE};">●</span> Bloom Filter<br>'
        f'<span style="color:{AMBER};">●</span> DGIM (log² N)<br>'
        f'<span style="color:{EMERALD};">●</span> Cosine Distance<br>'
        f'<span style="color:{CRIMSON};">●</span> Mahalanobis Dist.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
#  PROCESS BATCH
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.running and st.session_state.initialized:
    process_batch(st.session_state.batch_size)

# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
now_str  = datetime.now().strftime("%d %b %Y  %H:%M:%S")
tps      = calc_tps()
buf_size = len(st.session_state.tx_log)
badge    = f'<span class="sp-badge-live">● LIVE</span>' if st.session_state.running else f'<span class="sp-badge-idle">○ IDLE</span>'

st.markdown(f"""
<div class="sp-header">
  <div>
    <div class="sp-logo">🛡️ SentinelPay</div>
    <div class="sp-sub">Real-Time Financial Fraud &amp; Anomaly Intelligence &nbsp;·&nbsp; MU BDA Sem 7</div>
  </div>
  <div style="display:flex;gap:2rem;align-items:center;">
    <div style="text-align:right;">
      <div style="font-size:1.1rem;font-weight:800;color:{TXT};">{tps}<span style="font-size:0.7rem;color:{TXT_MUTED};margin-left:3px;">tx/s</span></div>
      <div style="font-size:0.68rem;color:{TXT_MUTED};text-transform:uppercase;letter-spacing:.08em;">Throughput</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.1rem;font-weight:800;color:{TXT};">{buf_size}<span style="font-size:0.7rem;color:{TXT_MUTED};margin-left:3px;">rows</span></div>
      <div style="font-size:0.68rem;color:{TXT_MUTED};text-transform:uppercase;letter-spacing:.08em;">Buffer</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.85rem;font-weight:600;color:{TXT};">{now_str}</div>
      {badge}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  KPI RIBBON
# ═════════════════════════════════════════════════════════════════════════════
bloom: BloomFilter    = st.session_state.bloom
dgim:  DGIMAlgorithm  = st.session_state.dgim

total_s  = st.session_state.total_streamed
total_f  = st.session_state.total_flagged
flag_pct = round(total_f / max(1, total_s) * 100, 1) if total_s else 0.0
bl_size  = st.session_state.blacklist_size
fp_rate  = round(bloom.current_fp_rate * 100, 4) if bloom else 0.0
dgim_est = dgim.estimate_count() if dgim else 0
bf_fill  = round(bloom.fill_ratio * 100, 1) if bloom else 0.0
bf_m     = bloom.m if bloom else "—"
bkt_cnt  = dgim.stats()["bucket_count"] if dgim else 0
surge    = bool(dgim and dgim.is_surge)
surge_c  = CRIMSON if surge else TXT_MUTED
surge_l  = "🔥 SURGE" if surge else "Normal"

def kpi(value: str, label: str, sub: str, color: str, sub_color: str = TXT_MUTED) -> str:
    return (
        f'<div class="sp-kpi-card">'
        f'<div class="sp-kpi-value" style="color:{color};">{value}</div>'
        f'<div class="sp-kpi-label">{label}</div>'
        f'<div class="sp-kpi-sub" style="color:{sub_color};">{sub}</div>'
        f'</div>'
    )

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.markdown(kpi(f"{total_s:,}",   "Total Streamed",    f"{tps} tx/s",             BLUE),    unsafe_allow_html=True)
k2.markdown(kpi(f"{total_f:,}",   "Flagged Frauds",    f"{flag_pct}% flag rate",  CRIMSON,  CRIMSON), unsafe_allow_html=True)
k3.markdown(kpi(str(bl_size),     "Blacklist Size",     f"FP rate: {fp_rate}%",    AMBER),   unsafe_allow_html=True)
k4.markdown(kpi(str(dgim_est),    "DGIM HV Estimate",  f"Window N={st.session_state.window_size}", PURPLE), unsafe_allow_html=True)
k5.markdown(kpi(f"{bf_fill}%",    "Bloom Fill Ratio",  f"m={bf_m} bits",          EMERALD), unsafe_allow_html=True)
k6.markdown(kpi(str(bkt_cnt),     "DGIM Buckets",      surge_l,                   AMBER,    surge_c), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  LIVE STREAM + TIMELINE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sp-section">📡 Live Transaction Stream</div>', unsafe_allow_html=True)

col_table, col_chart = st.columns([1.25, 1])

with col_table:
    if st.session_state.tx_log:
        df_disp = pd.DataFrame(st.session_state.tx_log[-60:])
        df_show = df_disp[["ts","type","amount","nameOrig","ip_address",
                            "cosine_dist","mahalanobis","tags"]].copy()
        df_show.columns = ["Time","Type","Amount","Account","IP",
                           "Cosine Δ","Mahal Δ","Status"]
        df_show["Amount"] = df_show["Amount"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(df_show[::-1].reset_index(drop=True),
                     use_container_width=True, height=390)
    else:
        st.info("▶ Click **Init** then **Stream** in the sidebar to start.")

with col_chart:
    st.plotly_chart(chart_timeline(st.session_state.tx_log), use_container_width=True)
    if bloom:
        bs = bloom.stats()
        st.markdown(
            f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:9px;'
            f'padding:8px 14px;font-size:0.76rem;color:{TXT_MUTED};">'
            f'<span style="color:{BLUE};font-weight:700;">Bloom Stats</span>'
            f'&nbsp;·&nbsp;m = {bs["bit_array_size_m"]:,} bits'
            f'&nbsp;·&nbsp;k = {bs["hash_functions_k"]} hashes'
            f'&nbsp;·&nbsp;FP ≈ {bs["estimated_fp_rate"]:.4%}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
#  ANALYTICS TABS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sp-section">🔬 Deep-Dive Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "  📊 Algorithm Inspection  ",
    "  📐 Distance Scatter  ",
    "  🎭 Fraud Simulation  ",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col_bf, col_dg = st.columns(2)

    with col_bf:
        st.markdown(f'<h4 style="color:{BLUE};">🔵 Bloom Filter</h4>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sp-algo-card">'
            f'<h4>Bloom Filter — Module 4: Stream Mining</h4>'
            f'<p>Space-efficient probabilistic membership test using <b>k</b> hash functions on an '
            f'<b>m</b>-bit array. Zero false negatives guaranteed. '
            f'False-positive rate: <b>P_fp ≈ (1 − e<sup>−kn/m</sup>)<sup>k</sup></b>. '
            f'Optimal params: <b>m = −n·ln(ε)/(ln2)²</b>, <b>k = (m/n)·ln2</b>.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if bloom:
            st.plotly_chart(chart_bloom(bloom), use_container_width=True)
            b1, b2, b3 = st.columns(3)
            bs = bloom.stats()
            b1.metric("Bit Array (m)", f'{bs["bit_array_size_m"]:,}')
            b2.metric("Hash Fns (k)",  bs["hash_functions_k"])
            b3.metric("Est. FP Rate",  f'{bs["estimated_fp_rate"]:.4%}')
        else:
            st.info("Initialise to view Bloom Filter state.")

    with col_dg:
        st.markdown(f'<h4 style="color:{AMBER};">🟡 DGIM Algorithm</h4>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sp-algo-card" style="border-left-color:{AMBER};">'
            f'<h4 style="color:{AMBER};">DGIM — Module 4: Stream Mining</h4>'
            f'<p>Counts high-value transactions (&gt;$5,000) in a rolling window of N events using '
            f'exponential buckets (1,2,4,8…). Invariant: <b>≤ 2 buckets per size</b>; '
            f'a third triggers a merge. Space: <b>O(log² N)</b>. Error: <b>≤ 50%</b>.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if dgim:
            st.plotly_chart(chart_dgim(dgim), use_container_width=True)
            ds = dgim.stats()
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Stream Pos",   f'{ds["stream_position"]:,}')
            d2.metric("DGIM Est.",    ds["dgim_estimate"])
            d3.metric("True Count",   ds["true_window_count"])
            d4.metric("Buckets",      ds["bucket_count"])
            if ds["buckets"]:
                st.markdown(f'<p style="font-size:0.8rem;font-weight:700;color:{TXT_MUTED};margin:8px 0 4px 0;">Bucket Detail</p>', unsafe_allow_html=True)
                df_b = pd.DataFrame(ds["buckets"])
                df_b.columns = ["Timestamp", "Bucket Size", "log₂(size)"]
                st.dataframe(df_b, use_container_width=True, height=190)
        else:
            st.info("Initialise to view DGIM state.")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<h4 style="color:{TXT};">📐 Cosine & Mahalanobis Distance Space</h4>', unsafe_allow_html=True)
    ci, cm = st.columns([2, 1])

    with ci:
        st.markdown(
            f'<div class="sp-algo-card" style="border-left-color:{EMERALD};">'
            f'<h4 style="color:{EMERALD};">Cosine Distance — Module 5</h4>'
            f'<p>Angular divergence between current tx vector and historical baseline: '
            f'<b>d(u,v) = 1 − (u·v)/(‖u‖‖v‖)</b>. Scale-invariant. '
            f'Range [0,2]: 0 = identical pattern, 1 = orthogonal, 2 = opposite.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sp-algo-card" style="border-left-color:{CRIMSON};">'
            f'<h4 style="color:{CRIMSON};">Mahalanobis Distance — Module 5</h4>'
            f'<p>Correlation-aware anomaly metric: <b>D_M = √[(x−μ)ᵀ Σ⁻¹ (x−μ)]</b>. '
            f'Uses precision matrix Σ⁻¹ to handle correlated features. '
            f'Under MVN: D_M² ~ χ²(d), so 3σ boundary ≈ D_M=3.7 for d=3.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with cm:
        dl = st.session_state.distance_log
        if dl:
            r = dl[-100:]
            st.metric("Avg Cosine Δ",   round(np.mean([d["cosine"]      for d in r]), 4))
            st.metric("Avg Mahal Δ",    round(np.mean([d["mahalanobis"] for d in r]), 3))
            st.metric("Peak Mahal Δ",   round(max(d["mahalanobis"]      for d in r), 2))
        else:
            st.info("Stream data to see metrics.")

    st.plotly_chart(chart_scatter(st.session_state.distance_log), use_container_width=True)

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<h4 style="color:{TXT};">🎭 Inject a Custom Transaction</h4>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{TXT_MUTED};font-size:0.84rem;">Craft a synthetic transaction and '
        f'push it through all four algorithms live to observe each detector\'s response.</p>',
        unsafe_allow_html=True,
    )

    with st.form("sim_form"):
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_type   = st.selectbox("Type", ["TRANSFER","CASH_OUT","PAYMENT","DEBIT","CASH_IN"])
            sim_amount = st.number_input("Amount ($)", 1.0, 1_000_000.0, 9999.99, 100.0)
        with s2:
            sim_orig    = st.text_input("Origin Account",  "C9999999999")
            sim_old_bal = st.number_input("Old Balance ($)", 0.0, 10_000_000.0, 10000.0)
            sim_new_bal = st.number_input("New Balance ($)", 0.0, 10_000_000.0, 0.01)
        with s3:
            sim_dest  = st.text_input("Dest Account",   "C1111111111")
            sim_ip    = st.text_input("Source IP",      "185.220.101.33")
            sim_fraud = st.checkbox("Mark as Known Fraud", value=True)

        submitted = st.form_submit_button("🚀 Inject Transaction", use_container_width=True)

    if submitted:
        if not st.session_state.initialized:
            st.error("Please click Init first.")
        else:
            sim_tx = build_synthetic_transaction(
                tx_type=sim_type, amount=sim_amount,
                name_orig=sim_orig, old_balance=sim_old_bal,
                new_balance=sim_new_bal, name_dest=sim_dest,
                ip_address=sim_ip, is_fraud=int(sim_fraud),
            )
            if sim_fraud:
                st.session_state.bloom.add(sim_ip)
                st.session_state.bloom.add(sim_orig)

            res = process_transaction(sim_tx)
            st.session_state.tx_log.append(res)
            st.session_state.total_streamed += 1
            if res["is_flagged"]:
                st.session_state.total_flagged += 1

            st.markdown(f'<hr style="border-color:{BORDER};">', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:0.9rem;font-weight:700;color:{TXT};">🔎 Algorithm Verdicts</p>', unsafe_allow_html=True)

            v1, v2, v3, v4 = st.columns(4)
            def verdict_card(col, title: str, flag: bool, val: str, on_color: str, off_color: str):
                c = on_color if flag else off_color
                lbl = "● TRIGGERED" if flag else "○ CLEAR"
                col.markdown(
                    f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                    f'border-top:3px solid {c};border-radius:10px;padding:0.9rem;text-align:center;">'
                    f'<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
                    f'color:{TXT_MUTED};letter-spacing:.08em;">{title}</div>'
                    f'<div style="font-size:1.2rem;font-weight:800;color:{c};margin:4px 0;">{lbl}</div>'
                    f'<div style="font-size:0.8rem;color:{TXT_MUTED};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            verdict_card(v1, "Bloom Filter",    res["is_blacklisted"], f'IP: {res["ip_address"]}',       CRIMSON, EMERALD)
            verdict_card(v2, "DGIM Surge",      res["dgim_surge"],     f'Est: {res["dgim_estimate"]}',   AMBER,   EMERALD)
            verdict_card(v3, "Cosine Dist.",    res["cos_anomaly"],    f'{res["cosine_dist"]:.4f}',      CRIMSON, EMERALD)
            verdict_card(v4, "Mahalanobis",     res["maha_anomaly"],   f'{res["mahalanobis"]:.4f}',      CRIMSON, EMERALD)

            oc = CRIMSON if res["is_flagged"] else EMERALD
            ol = "🚨 TRANSACTION FLAGGED — POTENTIAL FRAUD" if res["is_flagged"] else "✅ TRANSACTION CLEARED — NO ANOMALY DETECTED"
            st.markdown(
                f'<div class="sp-verdict" style="background:{oc}1A;border:1.5px solid {oc};color:{oc};">'
                f'{ol}</div>',
                unsafe_allow_html=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    time.sleep(0.35)
    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="sp-footer">'
    f'SentinelPay &nbsp;·&nbsp; University of Mumbai — BDA Semester 7 &nbsp;·&nbsp; '
    f'Bloom Filter &nbsp;·&nbsp; DGIM &nbsp;·&nbsp; Cosine Distance &nbsp;·&nbsp; Mahalanobis Distance'
    f'</div>',
    unsafe_allow_html=True,
)
