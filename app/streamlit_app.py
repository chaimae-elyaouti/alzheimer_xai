"""
AlzXAI — Explainable Alzheimer Stage Classifier
Design : Futuristic Clinical Dark Mode | INSEA Project 2026
Authors : EL YAOUTI Chaimae & ATIF Imane (INSEA)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import io
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap

# Import des modules locaux
from src.preprocessing import create_clinical_features

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG 
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AlzXAI · Alzheimer Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════
BG_DEEP    = "#06090F"
BG_PANEL   = "#0C1220"
BG_CARD    = "#101828"
BORDER     = "#1A2D45"
MUTED      = "#2E4460"
TEXT       = "#C4D4E8"
TEXT_DIM   = "#566F8F"
ACCENT     = "#38BDF8"

CN_CLR  = "#00FF88"   # neon green
MCI_CLR = "#FF9500"   # fluo orange
AD_CLR  = "#FF2D55"   # scarlet red

STAGE_META = {
    "CN":  {"label": "Cognitively Normal",        "color": CN_CLR,  "dim": "#002A18", "fa_icon": "fa-solid fa-circle-check"},
    "MCI": {"label": "Mild Cognitive Impairment", "color": MCI_CLR, "dim": "#291500", "fa_icon": "fa-solid fa-triangle-exclamation"},
    "AD":  {"label": "Alzheimer's Disease",       "color": AD_CLR,  "dim": "#270010", "fa_icon": "fa-solid fa-radiation"},
}
CLASSES       = ["CN", "MCI", "AD"]
CLASS_COLORS  = [CN_CLR, MCI_CLR, AD_CLR]

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CSS INJECTION & GITHUB MASKING
# ══════════════════════════════════════════════════════════════════════
def _css():
    st.markdown(f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap">

    <style>
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        background: {BG_DEEP};
        color: {TEXT};
    }}
    .stApp {{ background: {BG_DEEP}; }}
    .block-container {{ padding: 0 2rem 3rem; max-width: 1440px; }}


    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 4px; background: {BG_DEEP}; }}
    ::-webkit-scrollbar-thumb {{ background: {MUTED}; border-radius: 10px; }}

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {BG_PANEL};
        border-bottom: 1px solid {BORDER};
        padding: 0 1.5rem;
        gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_DIM};
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.85rem 1.6rem;
        border-bottom: 2px solid transparent;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom-color: {ACCENT} !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: {BG_DEEP};
        padding: 1.5rem 0 0;
    }}

    /* ── Sliders ── */
    div[data-baseweb="slider"] > div > div > div {{
        background: linear-gradient(90deg, {ACCENT}, #818CF8) !important;
    }}
    .stSlider label {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: {TEXT_DIM} !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    /* ── Number inputs ── */
    .stNumberInput input {{
        background: {BG_DEEP} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
    }}
    .stNumberInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 2px {ACCENT}22 !important;
    }}
    .stNumberInput label {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: {TEXT_DIM} !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    /* ── Selectbox ── */
    .stSelectbox [data-baseweb="select"] > div {{
        background: {BG_DEEP} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px;
    }}
    .stSelectbox label {{
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: {TEXT_DIM} !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    /* ── Checkboxes ── */
    .stCheckbox label p {{
        color: {TEXT} !important;
        font-size: 0.83rem;
    }}

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {{
        width: 100%;
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        border: none;
        color: white;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        cursor: pointer;
        box-shadow: 0 0 24px #0EA5E930;
        transition: all 0.25s ease;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 32px #6366F140;
    }}

    /* ── Metric ── */
    [data-testid="stMetricValue"] {{
        color: {ACCENT};
        font-weight: 800;
        font-size: 1.6rem !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_DIM};
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* ══ Custom components ══ */
    .alz-topbar {{
        background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_DEEP} 100%);
        border-bottom: 1px solid {BORDER};
        padding: 1.4rem 2rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        position: relative;
        overflow: hidden;
    }}

    .alz-section-label {{
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: {TEXT_DIM};
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 0.8rem;
    }}

    .alz-input-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.3rem 0.5rem;
        margin-bottom: 0.7rem;
    }}
    .alz-input-card-header {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {BORDER};
    }}

    .alz-idle-monitor {{
        height: 540px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: {BG_CARD};
        border: 1px dashed {BORDER};
        border-radius: 16px;
        gap: 0.8rem;
    }}

    .alz-banner {{
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin: 0.5rem 0 1rem;
        text-align: center;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }}

    .alz-prob-row {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }}
    .alz-prob-label {{
        min-width: 2.8rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }}
    .alz-prob-track {{
        flex: 1;
        height: 5px;
        background: {BG_DEEP};
        border-radius: 3px;
        overflow: hidden;
    }}
    .alz-prob-fill {{
        height: 100%;
        border-radius: 3px;
    }}
    .alz-prob-pct {{
        min-width: 3rem;
        text-align: right;
        font-size: 0.74rem;
        font-weight: 700;
    }}

    .alz-shap-wrapper {{
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}

    .alz-insight {{
        background: {BG_PANEL};
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.78rem;
        line-height: 1.65;
        color: {TEXT};
    }}

    .mini-tag {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        border: 1px solid;
    }}

    .alz-footer {{
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid {BORDER};
        text-align: center;
        font-size: 0.72rem;
        color: {TEXT_DIM};
        line-height: 1.8;
    }}
    </style>
    """, unsafe_allow_html=True)

_css()

# ══════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ══════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    mdl  = joblib.load(os.path.join(_MODELS, "best_pipeline.pkl"))
    fns  = joblib.load(os.path.join(_MODELS, "feature_names.pkl"))
    shex = joblib.load(os.path.join(_MODELS, "shap_explainer.pkl"))
    return mdl, fns, shex

_ARTIFACTS_OK = True
_LOAD_ERR = ""
try:
    pipeline, feature_names, shap_explainer = load_artifacts()
except Exception as _e:
    _ARTIFACTS_OK = False
    _LOAD_ERR = str(_e)
    pipeline = feature_names = shap_explainer = None

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def engineer_features(raw: dict) -> pd.DataFrame:
    df_raw = pd.DataFrame([raw])
    df_raw['CDR'] = 0.0
    df_raw['ASF'] = 1764.0 / (raw["eTIV"] + 1e-9)
    return create_clinical_features(df_raw)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG_PANEL, edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def get_clinical_insights(shap_values_matrix: np.ndarray, feat_names: list,
                          pred_label: str, raw: dict) -> list:
    cls_idx  = CLASSES.index(pred_label)
    clr      = STAGE_META[pred_label]["color"]
    
    abs_sv   = np.abs(shap_values_matrix[0, :, cls_idx])
    top_idxs = np.argsort(abs_sv)[::-1][:8]

    RULES = {
        "MMSE": lambda v: (
            "fa-solid fa-puzzle-piece", f"MMSE = {v:.0f} — Déclin cognitif global significatif (Seuil normal ≥ 24)."
            if v < 24 else
            "fa-solid fa-brain", f"MMSE = {v:.0f} — Performances cognitives globales préservées."
        ),
        "FAQ": lambda v: (
            "fa-solid fa-bolt", f"FAQ = {v:.0f} — Perte d'autonomie fonctionnelle quotidienne (Seuil d'alerte > 5)."
            if v > 5 else
            "fa-solid fa-circle-check", f"FAQ = {v:.0f} — Indépendance fonctionnelle quotidienne préservée."
        ),
        "nWBV": lambda v: (
            "fa-solid fa-gauge-high", f"nWBV = {v:.3f} — Volume cérébral normalisé faible; atrophie corticale active détectée."
            if v < 0.72 else
            "fa-solid fa-shield-halved", f"nWBV = {v:.3f} — Volume de la masse cérébrale conforme aux attentes."
        ),
        "Abeta42": lambda v: (
            "fa-solid fa-dna", f"CSF Aβ42 = {v:.0f} pg/mL — Concentration effondrée; agrégation active des plaques amyloïdes."
            if v < 400 else
            "fa-solid fa-dna", f"CSF Aβ42 = {v:.0f} pg/mL — Taux d'amyloïde soluble dans les normes bio-cliniques."
        ),
        "pTau181": lambda v: (
            "fa-solid fa-flask-vial", f"pTau181 = {v:.0f} pg/mL — Concentration élevée; propagation des enchevêtrements neurofibrillaires."
            if v > 50 else
            "fa-solid fa-flask", f"pTau181 = {v:.0f} pg/mL — Absence d'hyper-phosphorylation de la protéine Tau."
        ),
        "NfL": lambda v: (
            "fa-solid fa-chart-line", f"NfL = {v:.0f} pg/mL — Taux élevé; témoin direct d'une lyse neuronale en cours."
            if v > 40 else
            "fa-solid fa-arrow-trend-down", f"NfL = {v:.0f} pg/mL — Intégrité axonale respectée."
        ),
    }

    eng = engineer_features(raw).iloc[0].to_dict()
    eng.update(raw)

    insights = []
    seen = set()
    for idx in top_idxs:
        if idx >= len(feat_names):
            continue
        fname = feat_names[idx]
        if fname in RULES and fname not in seen:
            seen.add(fname)
            val = eng.get(fname, None)
            if val is not None:
                try:
                    icon, msg = RULES[fname](float(val))
                    insights.append({"fa_icon": icon, "color": clr, "feature": fname, "msg": msg})
                except Exception:
                    pass
        if len(insights) >= 3:
            break

    if not insights:
        insights.append({
            "fa_icon": "fa-solid fa-info", "color": clr, "feature": "Modèle",
            "msg": "Le diagnostic repose sur une convergence globale de la signature biologique."
        })
    return insights

# ══════════════════════════════════════════════════════════════════════
# TOP BAR 
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="alz-topbar">
  <div style="font-size:2rem; color:{ACCENT}; flex-shrink:0;"><i class="fa-solid fa-brain"></i></div>
  <div style="flex:1;">
    <div style="font-size:1.45rem; font-weight:900; letter-spacing:-0.02em;
                background:linear-gradient(90deg,{ACCENT} 0%,{CN_CLR} 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                line-height:1.15;">
      AlzXAI
    </div>
    <div style="font-size:0.68rem; color:{TEXT_DIM}; letter-spacing:0.12em;
                text-transform:uppercase; margin-top:3px;">
      Explainable Alzheimer Stage Classifier &nbsp;·&nbsp; LightGBM + SHAP TreeExplainer
    </div>
  </div>
  <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
    <span class="mini-tag" style="color:{CN_CLR}; border-color:{CN_CLR}44; background:{CN_CLR}0D;">CN · Normal</span>
    <span class="mini-tag" style="color:{MCI_CLR}; border-color:{MCI_CLR}44; background:{MCI_CLR}0D;">MCI · Léger</span>
    <span class="mini-tag" style="color:{AD_CLR}; border-color:{AD_CLR}44; background:{AD_CLR}0D;">AD · Alzheimer</span>
    <span style="font-size:0.62rem; color:{MUTED}; margin-left:0.5rem; letter-spacing:0.07em; text-transform:uppercase;">INSEA · 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not _ARTIFACTS_OK:
    st.error(f"**Artefacts de production introuvables.** Chemin attendu : `../models/`. Erreur : `{_LOAD_ERR}`")

# ══════════════════════════════════════════════════════════════════════
# MAIN TABS (Remplacement par icônes FontAwesome vectorielles)
# ══════════════════════════════════════════════════════════════════════
T_PATIENT, T_COHORT, T_GLOBAL, T_METHOD = st.tabs([
    "🩺 Patient Analysis",
    "📊 Cohort Screening",
    "🔬 Global Interpretability",
    "📑 Methodology",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PATIENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with T_PATIENT:
    col_L, col_R = st.columns([1.1, 1.3], gap="large")

    with col_L:
        st.markdown(f"<div class='alz-section-label'>Clinical Hub — Data Entry</div>", unsafe_allow_html=True)

        # A · Patient Demographics
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{ACCENT};'><i class='fa-solid fa-user-doctor'></i> &nbsp;Patient Demographics</div>", unsafe_allow_html=True)
        cA1, cA2 = st.columns(2)
        with cA1:
            age = st.slider("Age (years)", 55, 95, 72, key="age")
            education = st.slider("Education (years)", 0, 20, 13, key="edu")
        with cA2:
            sex = st.selectbox("Biological Sex", ["F", "M"], key="sex")
            apoe4 = st.selectbox("APOE ε4 Alleles", [0, 1, 2], key="apoe4")
        st.markdown("</div>", unsafe_allow_html=True)

        # B · Cognitive Assessment
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{MCI_CLR};'><i class='fa-solid fa-brain'></i> &nbsp;Cognitive Assessment</div>", unsafe_allow_html=True)
        cB1, cB2 = st.columns(2)
        with cB1:
            mmse = st.slider("MMSE Score (0–30)", 0, 30, 24, key="mmse")
        with cB2:
            faq = st.slider("FAQ Score (0–30)", 0, 30, 5, key="faq")
        st.markdown("</div>", unsafe_allow_html=True)

        # C · CSF / Blood Biomarkers
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{AD_CLR};'><i class='fa-solid fa-flask-vial'></i> &nbsp;CSF / Blood Biomarkers</div>", unsafe_allow_html=True)
        cC1, cC2, cC3 = st.columns(3)
        with cC1:
            abeta42 = st.number_input("Aβ42 (pg/mL)", 100, 800, 380, step=10, key="abeta42")
        with cC2:
            ptau181 = st.number_input("pTau181 (pg/mL)", 10, 150, 55, step=5, key="ptau181")
        with cC3:
            nfl = st.number_input("NfL (pg/mL)", 5, 120, 40, step=5, key="nfl")
        st.markdown("</div>", unsafe_allow_html=True)

        # D · Structural MRI
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{ACCENT};'><i class='fa-solid fa-wave-square'></i> &nbsp;Structural MRI</div>", unsafe_allow_html=True)
        cD1, cD2 = st.columns(2)
        with cD1:
            nwbv = st.slider("nWBV (ratio)", 0.60, 0.90, 0.74, step=0.005, key="nwbv")
        with cD2:
            etiv = st.slider("eTIV (cm³)", 900, 2100, 1480, key="etiv")
        st.markdown("</div>", unsafe_allow_html=True)

        # E · Comorbidities
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{TEXT_DIM};'><i class='fa-solid fa-notes-medical'></i> &nbsp;Comorbidities</div>", unsafe_allow_html=True)
        cE1, cE2, cE3 = st.columns(3)
        with cE1:
            htn = st.checkbox("Hypertension", key="htn")
        with cE2:
            diab = st.checkbox("Diabetes", key="diab")
        with cE3:
            depr = st.checkbox("Depression Hx", key="depr")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        run = st.button("Run Diagnostic Analysis", type="primary", key="run")

    # ── RIGHT : Decision Monitor ─────────────────────────────────────
    with col_R:
        st.markdown(f"<div class='alz-section-label'>Decision Monitor</div>", unsafe_allow_html=True)

        if not run:
            st.markdown(f"""
            <div class="alz-idle-monitor">
              <div style="font-size:3rem; color:{MUTED}; opacity:.3;"><i class="fa-solid fa-heart-pulse"></i></div>
              <div style="font-size:.78rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:{MUTED};">Awaiting Patient Data</div>
              <div style="font-size:.7rem; color:{TEXT_DIM}; max-width:240px; text-align:center; line-height:1.8;">
                Remplissez les informations cliniques à gauche puis déclenchez l'analyse prédictive.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if _ARTIFACTS_OK:
                raw_input = {
                    "Age": age, "Sex": sex, "Education": education, "MMSE": mmse,
                    "nWBV": nwbv, "eTIV": etiv, "Abeta42": abeta42, "pTau181": ptau181, "NfL": nfl,
                    "FAQ": faq, "APOE4": apoe4, "Hypertension": int(htn), "Diabetes": int(diab), "Depression_Hx": int(depr),
                }
                try:
                    df_pt = engineer_features(raw_input)
                    proba = pipeline.predict_proba(df_pt)[0]
                    pidx = int(np.argmax(proba))
                    plabel = CLASSES[pidx]
                    meta = STAGE_META[plabel]
                    pclr = meta["color"]

                    # 1. Diagnostic Banner
                    st.markdown(f"""
                    <div class="alz-banner" style="background:{meta['dim']}; border-color:{pclr}55;">
                      <div style="font-size:.62rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:{pclr}; margin-bottom:.4rem;">
                        <i class="{meta['fa_icon']}"></i> &nbsp;Classified Stage
                      </div>
                      <div style="font-size:2.2rem; font-weight:900; color:{pclr}; letter-spacing:-.02em; line-height:1.1;">{plabel}</div>
                      <div style="font-size:.82rem; color:{TEXT}; margin-top:.35rem; font-weight:500;">{meta['label']}</div>
                      <div style="font-size:.7rem; color:{TEXT_DIM}; margin-top:.5rem;">Confiance du modèle : <strong style="color:{pclr}; font-family:'JetBrains Mono',monospace;">{proba[pidx]*100:.1f}%</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 2. Concentric Ring Chart
                    fig_ring = go.Figure()
                    ring_sizes = [0.20, 0.14, 0.10]
                    ring_offsets = [0.0, 0.07, 0.12]
                    for i, (cls, clr_r, sz, off) in enumerate(zip(CLASSES, CLASS_COLORS, ring_sizes, ring_offsets)):
                        fig_ring.add_trace(go.Pie(
                            values=[proba[i], 1 - proba[i]], hole=0.62 + off, sort=False, direction="clockwise",
                            marker=dict(colors=[clr_r, "rgba(0,0,0,0)"]), showlegend=False, textinfo="none",
                            hovertemplate=f"<b>{cls}</b>: {proba[i]*100:.1f}%<extra></extra>",
                            domain={"x": [0.0 + off * 0.6, 1.0 - off * 0.6], "y": [0.0 + off * 0.6, 1.0 - off * 0.6]}
                        ))
                    fig_ring.update_layout(
                        height=210, margin=dict(t=8, b=8, l=8, r=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        annotations=[dict(text=f"<span style='font-size:24px; font-weight:900; color:{pclr};'>{proba[pidx]*100:.0f}%</span><br><span style='font-size:11px; color:{TEXT_DIM};'>{plabel}</span>", x=0.5, y=0.5, showarrow=False)]
                    )
                    st.plotly_chart(fig_ring, use_container_width=True, config={"displayModeBar": False})

                    # 3. Probability Bars
                    for cls, prob, clr_b in zip(CLASSES, CLASS_COLORS):
                        st.markdown(f"""
                        <div class="alz-prob-row">
                          <div class="alz-prob-label" style="color:{clr_b};">{cls}</div>
                          <div class="alz-prob-track"><div class="alz-prob-fill" style="width:{prob*100:.1f}%; background:{clr_b};"></div></div>
                          <div class="alz-prob-pct" style="color:{clr_b};">{prob*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 4. SHAP Waterfall Local
                    st.markdown("<div class='alz-section-label' style='margin-top:1.1rem;'>SHAP Local Analysis — Feature Contributions</div>", unsafe_allow_html=True)
                    try:
                        preproc = pipeline.named_steps["preprocessor"]
                        X_trans = preproc.transform(df_pt)
                        shap_vals = shap_explainer.shap_values(X_trans)
                        sv_local = shap_vals[0, :, pidx]

                        n_top = 10
                        s_idxs = np.argsort(np.abs(sv_local))[::-1][:n_top]
                        vals_plot = sv_local[s_idxs][::-1]
                        names_plot = [(feature_names[i] if i < len(feature_names) else f"f{i}") for i in s_idxs][::-1]

                        fig_wf, ax = plt.subplots(figsize=(6.2, 3.8))
                        fig_wf.patch.set_facecolor(BG_PANEL)
                        ax.set_facecolor(BG_PANEL)

                        bar_clrs = [AD_CLR if v > 0 else CN_CLR for v in vals_plot]
                        ax.barh(range(len(vals_plot)), vals_plot, color=bar_clrs, height=0.58, zorder=3)
                        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)

                        ax.set_yticks(range(len(names_plot)))
                        ax.set_yticklabels(names_plot, fontsize=8, color=TEXT)
                        ax.tick_params(axis="x", colors=TEXT_DIM, labelsize=7)
                        ax.spines[:].set_visible(False)
                        ax.grid(axis="x", color=BORDER, linewidth=0.4, alpha=0.5, zorder=1)
                        plt.tight_layout(pad=0.7)

                        st.markdown("<div class='alz-shap-wrapper'>", unsafe_allow_html=True)
                        st.image(f"data:image/png;base64,{fig_to_b64(fig_wf)}", use_column_width=True)
                        plt.close(fig_wf)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # 5. Clinical Insights (Remplacement des émojis par badges)
                        st.markdown("<div class='alz-section-label' style='margin-top:.9rem;'>Clinical Insights — Key Drivers Interpretation</div>", unsafe_allow_html=True)
                        insights = get_clinical_insights(shap_vals, feature_names, plabel, raw_input)
                        for ins in insights:
                            st.markdown(f"""
                            <div class="alz-insight" style="border-left-color:{ins['color']};">
                              <i class="{ins['fa_icon']}" style="color:{ins['color']}; margin-right:5px;"></i> &nbsp;<strong style="color:{ins['color']};">{ins['feature']}</strong> &nbsp;— {ins['msg']}
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as _se:
                        st.warning(f"Erreur de calcul SHAP: {_se}")

                    st.markdown(f"""
                    <div style="font-size:0.68rem; color:{TEXT_DIM}; line-height:1.6; margin-top:1.5rem; padding:0.8rem; border-top:1px solid {BORDER};">
                      <i class="fa-solid fa-triangle-exclamation"></i> &nbsp;<strong>Outil de recherche académique.</strong> Prototype d'aide à la décision. Tout résultat doit faire l'objet d'une validation clinique par un neurologue.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as _pe:
                    st.error(f"Erreur d'analyse: {_pe}")
            else:
                st.error("Les artefacts du modèle ne sont pas chargés correctement.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — COHORT SCREENING
# ══════════════════════════════════════════════════════════════════════
with T_COHORT:
    st.markdown("<div class='alz-section-label'>Batch Patient Screening — CSV Upload</div>", unsafe_allow_html=True)
    cH1, cH2 = st.columns([1.2, 1])
    with cH1:
        st.markdown(f"""
        <div class="alz-input-card" style="font-size:.8rem; color:{TEXT_DIM}; line-height:1.9;">
          <i class="fa-solid fa-file-csv" style="color:{ACCENT};"></i> &nbsp;Uploadez un fichier CSV de patients. Le pipeline applique automatiquement le pré-traitement et le feature engineering clinique.<br><br>
          <strong style="color:{TEXT};">Colonnes requises :</strong><br>
          <code style="color:{ACCENT}; font-size:.72rem; font-family:'JetBrains Mono',monospace;">Age, Sex, Education, MMSE, nWBV, eTIV, Abeta42, pTau181, NfL, FAQ, APOE4, Hypertension, Diabetes, Depression_Hx</code>
        </div>
        """, unsafe_allow_html=True)
    with cH2:
        uploaded = st.file_uploader("Drop cohort CSV here", type=["csv"], key="cohort_file")

    if uploaded and _ARTIFACTS_OK:
        try:
            df_raw_batch = pd.read_csv(uploaded)
            st.info(f" Patients chargés — exécution de l'inférence en lot…")

            rows = []
            for _, row in df_raw_batch.iterrows():
                rows.append(engineer_features(row.to_dict()).iloc[0])
            df_feat = pd.DataFrame(rows)

            batch_probas = pipeline.predict_proba(df_feat)
            batch_preds = pipeline.predict(df_feat)
            batch_labels = [CLASSES[p] for p in batch_preds]

            df_out = df_raw_batch.copy()
            df_out["Predicted_Stage"] = batch_labels
            df_out["Prob_CN"] = batch_probas[:, 0].round(3)
            df_out["Prob_MCI"] = batch_probas[:, 1].round(3)
            df_out["Prob_AD"] = batch_probas[:, 2].round(3)

            counts = pd.Series(batch_labels).value_counts()

            k1, k2, k3 = st.columns(3)
            for col, cls, clr_k in zip([k1, k2, k3], CLASSES, CLASS_COLORS):
                n = counts.get(cls, 0)
                col.metric(f"Patients {cls}", n, f"{n/len(df_out)*100:.1f}%")

            fig_pie = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.55,
                marker=dict(colors=[CN_CLR if l == "CN" else MCI_CLR if l == "MCI" else AD_CLR for l in counts.index]),
                textinfo="label+percent", textfont=dict(size=11, color="white"),
            ))
            fig_pie.update_layout(height=270, margin=dict(t=20, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", showlegend=False)

            pA, pB = st.columns([1, 2])
            with pA:
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            with pB:
                st.dataframe(df_out[["Predicted_Stage", "Prob_CN", "Prob_MCI", "Prob_AD"]].style.background_gradient(subset=["Prob_AD"], cmap="Reds").format(precision=3), use_container_width=True, height=255)

            st.download_button("📥  Download Full Results (CSV)", data=df_out.to_csv(index=False).encode("utf-8"), file_name="alzxai_cohort_predictions.csv", mime="text/csv")
        except Exception as _ce:
            st.error(f"Erreur de lot: {_ce}")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — GLOBAL INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════
with T_GLOBAL:
    st.markdown("<div class='alz-section-label'>Global Model Interpretability — SHAP Analysis</div>", unsafe_allow_html=True)
    _FIG = os.path.join(_HERE, "..", "reports", "figures")

    def _show(fname, caption):
        p = os.path.join(_FIG, fname)
        if os.path.exists(p):
            st.image(p, caption=caption, use_column_width=True)
        else:
            st.markdown(f"<div class='alz-input-card' style='text-align:center; color:{TEXT_DIM}; font-size:.78rem; padding:2rem 1rem;'><i class='fa-solid fa-folder-open'></i> &nbsp;Figure non générée : <code>{fname}</code><br><span style='opacity:.55;'>Exécutez le notebook 03_XAI.ipynb pour matérialiser ce graphique.</span></div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("<div class='alz-section-label'>SHAP Beeswarm — All Classes</div>", unsafe_allow_html=True)
        _show("05_shap_beeswarm.png", "Beeswarm plot — distribution de l'impact des features par classe")
    with g2:
        st.markdown("<div class='alz-section-label'>Global Feature Importance</div>", unsafe_allow_html=True)
        _show("06_shap_importance.png", "Mean |SHAP| — Classement mondial des caractéristiques les plus discriminantes")

    st.markdown("<div class='alz-section-label' style='margin-top:1.5rem;'>Dependence Plots — Clinical Interactions</div>", unsafe_allow_html=True)
    _show("07_shap_dependence.png", "Interactions non-linéaires : MMSE × APOE4 (gauche) · Aβ42 × pTau181 (droite)")

    st.markdown("<div class='alz-section-label' style='margin-top:1.5rem;'>Model Performance Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="alz-input-card">
      <table style="width:100%; border-collapse:collapse; font-size:.78rem; color:{TEXT}; text-align:left;">
        <thead>
          <tr style="border-bottom:1px solid {BORDER}; color:{TEXT_DIM}; font-size:.66rem; text-transform:uppercase; letter-spacing:.1em;">
            <th style="padding:.65rem .8rem;">Métrique d'Évaluation</th>
            <th style="padding:.65rem .8rem; text-align:center;">Validation Croisée (5-Fold)</th>
            <th style="padding:.65rem .8rem; text-align:center;">Test Set Hold-Out</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid {BORDER}22;">
            <td style="padding:.6rem .8rem; color:{TEXT}; font-weight:500;">F1-Score Macro</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{ACCENT}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9122</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{CN_CLR}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9070</td>
          </tr>
          <tr style="border-bottom:1px solid {BORDER}22;">
            <td style="padding:.6rem .8rem; color:{TEXT}; font-weight:500;">F1-Score Pondéré</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{ACCENT}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9190</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{CN_CLR}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9120</td>
          </tr>
          <tr style="border-bottom:1px solid {BORDER}22;">
            <td style="padding:.6rem .8rem; color:{TEXT}; font-weight:500;">AUC-ROC Multiclasse</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{ACCENT}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9671</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{CN_CLR}; font-family:'JetBrains Mono',monospace; font-weight:600;">0.9630</td>
          </tr>
          <tr style="border-bottom:1px solid {BORDER}22;">
            <td style="padding:.6rem .8rem; color:{TEXT}; font-weight:500;">Précision Globale (Accuracy)</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{ACCENT}; font-family:'JetBrains Mono',monospace; font-weight:600;">91.90%</td>
            <td style="padding:.6rem .8rem; text-align:center; color:{CN_CLR}; font-family:'JetBrains Mono',monospace; font-weight:600;">97.00%</td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════
with T_METHOD:
    st.markdown("<div class='alz-section-label'>Technical Architecture & References</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="alz-input-card">
          <div class="alz-input-card-header" style="color:{ACCENT};"><i class="fa-solid fa-gears"></i> &nbsp;Pipeline Architecture</div>
          <table style="width:100%; font-size:.78rem; color:{TEXT}; border-collapse:collapse; line-height:2.0;">
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Classifier</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">LightGBM (multiclass)</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Preprocessing</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">Scikit-Learn Pipeline</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Imputation</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">KNN Imputer (biomarkers)</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Imbalance</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">class_weight='balanced'</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">HPO Engine</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">Optuna TPE Sampler</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Validation</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">Stratified K-Fold (k=5)</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">XAI Global</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">SHAP TreeExplainer</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">XAI Local</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">SHAP Waterfall Framework</td></tr>
            <tr style="border-bottom:1px solid {BORDER}22;"><td style="padding:.5rem; color:{TEXT_DIM}; text-transform:uppercase; font-size:.65rem; letter-spacing:0.05em;">Deployment</td><td style="padding:.5rem; color:{ACCENT}; font-weight:600; font-family:'JetBrains Mono',monospace;">Streamlit UI / WSL Linux</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        fn_list = feature_names if _ARTIFACTS_OK and feature_names else ["Age","Education","MMSE","nWBV","eTIV","ASF","Abeta42","pTau181","NfL","FAQ","APOE4","Amyloid_Tau_Ratio","Cog_Functional_Score","Brain_Age_Ratio","APOE4_Risk","Comorbidity_Index","NfL_Age_Adjusted","Sex","Hypertension","Diabetes","Depression_Hx"]
        tags_html = " ".join(f'<span style="background:{BG_DEEP}; border:1px solid {BORDER}; border-radius:4px; padding:.12rem .48rem; color:{TEXT}; font-family:\'JetBrains Mono\',monospace; font-size:.68rem; display:inline-block; margin:.18rem .12rem;">{f}</span>' for f in fn_list)
        st.markdown(f"<div class='alz-input-card'><div class='alz-input-card-header' style='color:{MCI_CLR};'><i class='fa-solid fa-list-check'></i> &nbsp;Feature Set ({len(fn_list)} features — CDR excluded)</div><div style='line-height:2.2;'>{tags_html}</div></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alz-input-card" style="margin-top:.5rem;">
      <div class="alz-input-card-header" style="color:{CN_CLR};"><i class="fa-solid fa-book-bookmark"></i> &nbsp;Key Bibliographic References</div>
      <div style="font-size:.77rem; color:{TEXT_DIM}; line-height:2.3;">
        <div><i class="fa-solid fa-file-pdf"></i> &nbsp;Marcus et al. (2010). <em>Open Access Series of Imaging Studies (OASIS)</em>. <em>J. Cognitive Neuroscience</em>.</div>
        <div><i class="fa-solid fa-file-pdf"></i> &nbsp;Jack et al. (2018). NIA-AA Research Framework — biological definition of AD. <em>Alzheimer's &amp; Dementia</em>.</div>
        <div><i class="fa-solid fa-file-pdf"></i> &nbsp;Lundberg &amp; Lee (2017). A unified approach to interpreting model predictions (SHAP). <em>NeurIPS</em>.</div>
        <div><i class="fa-solid fa-file-pdf"></i> &nbsp;Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. <em>NeurIPS</em>.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# FOOTER 
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="alz-footer">
  AlzXAI &nbsp;·&nbsp; Explainable Alzheimer Stage Classifier<br>
  <strong>EL YAOUTI Chaimae &amp; ATIF Imane</strong> — Élèves Ingénieures en Data Science<br>
  INSEA — Institut National de Statistique et d'Économie Appliquée &nbsp;·&nbsp; Rabat, Maroc &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)