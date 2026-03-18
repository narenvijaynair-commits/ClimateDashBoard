"""
8_Country_Lookup.py
════════════════════
Free-text country search — retrieves all environmental indices for any country.
Imports all data/logic from country_utils.py (project root).

Place this file in:   pages/8_Country_Lookup.py
country_utils.py must sit in the project root next to Home.py.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from country_utils import (
    INDICES, OBS_KEYS, FUT_KEYS,
    COUNTRY_COORDS, ALIASES, CONTINENT_MAPPING,
    get_all_indices, load_hdi, fuzzy_resolve, hdi_tier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Country Lookup",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e8e8; }

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebarNav"] { display: none; }

/* big search bar */
div[data-testid="stTextInput"] input {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important;
    background: #161b22 !important;
    border: 1px solid rgba(0,200,180,0.35) !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
    padding: 0.65rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #00c8b4 !important;
    box-shadow: 0 0 0 3px rgba(0,200,180,0.12) !important;
}
div[data-testid="stTextInput"] label {
    font-size: 0.72rem !important;
    color: #8b949e !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* section headers */
.sec-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin: 28px 0 14px 0;
}

/* index cards */
.ic {
    background: linear-gradient(145deg,rgba(0,200,180,.06),rgba(15,40,60,.55));
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
    transition: border-color .2s;
}
.ic:hover { border-color: rgba(0,200,180,.4); }
.ic-lbl  { font-size:.68rem; color:#8b949e; letter-spacing:.1em;
            text-transform:uppercase; margin-bottom:4px; }
.ic-val  { font-family:'IBM Plex Mono',monospace; font-size:1.25rem;
            font-weight:600; margin-bottom:2px; }
.ic-dlt  { font-size:.7rem; margin-bottom:6px; }
.ic-unit { font-size:.68rem; color:#8b949e; }
.ic-src  { font-size:.65rem; color:#3d6a7a; margin-top:6px; }

/* hero */
.hero {
    background: linear-gradient(135deg,rgba(0,200,180,.09) 0%,rgba(56,178,255,.07) 100%);
    border: 1px solid rgba(0,200,180,.2);
    border-radius: 16px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.2rem;
}

/* pill */
.pill {
    display:inline-block; font-family:'IBM Plex Mono',monospace;
    font-size:.72rem; padding:3px 12px; border-radius:20px;
}

/* empty-state prompt */
.prompt-wrap {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding: 4rem 1rem; gap: 0.6rem;
}
.prompt-title {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.1rem; color:#4a7a8a; letter-spacing:.05em;
}
.prompt-sub { font-size:.82rem; color:#30363d; }

/* suggestion chips */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:.5rem; justify-content:center; }
.chip {
    font-family:'IBM Plex Mono',monospace; font-size:.72rem;
    padding:4px 14px; border-radius:20px;
    background:rgba(0,200,180,.07); border:1px solid rgba(0,200,180,.2);
    color:#7ec8d8;
}

#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar nav
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("Home.py",                    label="Home",                 icon="🏠")
    st.page_link("pages/1_Comparative_Analysis.py",     label="Comparative Analysis", icon="🌍")
    st.page_link("pages/2_Precipitation.py",         label="Precipitation",        icon="🌧️")
    st.page_link("pages/3_Water_Storage.py",       label="Water Storage",        icon="💧")
    st.page_link("pages/4_Air_Quality.py",        label="Air Quality",          icon="💨")
    st.page_link("pages/5_Temperature.py",    label="Temperature",          icon="🌡️")
    st.page_link("pages/6_Future_Temperature.py",        label="Future Temperature",   icon="🔮")
    st.page_link("pages/7_Future_Precipitation.py",      label="Future Precipitation", icon="🌦️")
    st.page_link("pages/8_Country_Lookup.py",  label="Country Lookup",       icon="🌐")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:.76rem;color:#4a7a8a;'>"
        "Source: NASA GPM · GRACE · GISTEMP · CMIP6 · UNDP HDI 2025"
        "</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🌐 Country Lookup")
st.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:11px;color:#8b949e;'>"
    "Type any country name, alias, or abbreviation to retrieve all environmental indices"
    "</span>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Search input
# ─────────────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Country",
    placeholder="e.g.  Brazil · South Korea · DRC · USA · Ivory Coast · Türkiye …",
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Resolve query → canonical country
# ─────────────────────────────────────────────────────────────────────────────
canonical, display_name, alternatives, err = fuzzy_resolve(query)

# ─────────────────────────────────────────────────────────────────────────────
# Empty-state
# ─────────────────────────────────────────────────────────────────────────────
if not query.strip():
    st.markdown("""
    <div class="prompt-wrap">
        <div class="prompt-title">↑ Search for any country above</div>
        <div class="prompt-sub">Accepts common names, abbreviations, and alternate spellings</div>
        <div class="chips">
            <span class="chip">India</span>
            <span class="chip">Brazil</span>
            <span class="chip">USA</span>
            <span class="chip">DRC</span>
            <span class="chip">Russia</span>
            <span class="chip">South Korea</span>
            <span class="chip">Ivory Coast</span>
            <span class="chip">UK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Error state — no match found
# ─────────────────────────────────────────────────────────────────────────────
if err:
    st.warning(err, icon="⚠️")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy-match notice when resolved name differs from raw query
# ─────────────────────────────────────────────────────────────────────────────
typed_lower = query.strip().lower()
if typed_lower != canonical and typed_lower not in ALIASES:
    msg = f"🔄 Showing results for **{display_name}**"
    if alternatives:
        msg += "  ·  Also found: " + " · ".join(f"**{a}**" for a in alternatives)
    st.info(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading indices for {display_name}…"):
    hdi_lookup = load_hdi()
    hdi_val    = hdi_lookup.get(canonical)
    if hdi_val is None:
        for alias, canon in ALIASES.items():
            if canon == canonical:
                hdi_val = hdi_lookup.get(alias)
                if hdi_val:
                    break

    tier_label, tier_color, tier_bg = hdi_tier(hdi_val)
    index_results = get_all_indices(canonical)

coords    = COUNTRY_COORDS.get(canonical)
continent = CONTINENT_MAPPING.get(canonical, "Unknown")
lat_str   = f"{abs(coords[0]):.1f}°{'N' if coords[0]>=0 else 'S'}" if coords else "—"
lon_str   = f"{abs(coords[1]):.1f}°{'E' if coords[1]>=0 else 'W'}" if coords else "—"

# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:flex-start;flex-wrap:wrap;gap:2.5rem;">
    <div>
      <div style="font-size:.7rem;color:#8b949e;letter-spacing:.12em;
                  text-transform:uppercase;margin-bottom:4px;">Country</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                  font-weight:600;color:#e8e8e8;line-height:1.1;">{display_name}</div>
      <div style="font-size:.82rem;color:#7ec8d8;margin-top:6px;">
          {continent} &nbsp;·&nbsp; {lat_str}, {lon_str}
      </div>
    </div>
    <div style="border-left:1px solid #30363d;padding-left:2.5rem;">
      <div style="font-size:.7rem;color:#8b949e;letter-spacing:.12em;
                  text-transform:uppercase;margin-bottom:4px;">HDI 2022</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                  font-weight:600;color:{tier_color};">
          {f"{hdi_val:.3f}" if hdi_val else "—"}
      </div>
      <span class="pill"
            style="background:{tier_bg};color:{tier_color};
                   border:1px solid {tier_color}44;margin-top:4px;">
          {tier_label} Human Development
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Index card renderer
# ─────────────────────────────────────────────────────────────────────────────
def _card(col, key):
    meta     = INDICES[key]
    val, gm  = index_results.get(key, (None, None))
    label    = meta['label']
    units    = meta['units']
    icon     = meta['icon']
    src      = meta['source']
    pos_good = meta['pos_good']

    with col:
        if val is None:
            st.markdown(f"""
            <div class="ic">
              <div class="ic-lbl">{icon} {label}</div>
              <div class="ic-val" style="color:#3d5a66;">—</div>
              <div class="ic-unit">Data not available</div>
              <div class="ic-src">{src}</div>
            </div>""", unsafe_allow_html=True)
            return

        # Green if the value's direction is favourable, red if not
        val_is_good = (val > 0) == pos_good
        vc = "#2ecc71" if val_is_good else "#e74c3c"

        if gm is not None and gm != 0:
            delta   = val - gm
            pct     = delta / abs(gm) * 100
            arrow   = "▲" if delta >= 0 else "▼"
            is_good = (delta >= 0) == pos_good
            dc      = "#2ecc71" if is_good else "#e74c3c"
            dstr    = f"{arrow} {abs(pct):.1f}% vs global ({gm:+.4f} {units})"
        else:
            dstr, dc = "", "#8b949e"

        st.markdown(f"""
        <div class="ic">
          <div class="ic-lbl">{icon} {label}</div>
          <div class="ic-val" style="color:{vc};">{val:+.4f}</div>
          <div class="ic-dlt" style="color:{dc};">{dstr}</div>
          <div class="ic-unit">{units}</div>
          <div class="ic-src">{src}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Observed index cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📡 Observed Trends</div>', unsafe_allow_html=True)
for col, key in zip(st.columns(4), OBS_KEYS):
    _card(col, key)

# ─────────────────────────────────────────────────────────────────────────────
# Future projection cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔭 CMIP6 Future Projections</div>', unsafe_allow_html=True)
for col, key in zip(st.columns(4), FUT_KEYS):
    _card(col, key)

# ─────────────────────────────────────────────────────────────────────────────
# Deviation bar chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 Deviation from Global Mean</div>', unsafe_allow_html=True)

labels, vals, colors, hovers = [], [], [], []
for key in OBS_KEYS + FUT_KEYS:
    v, gm = index_results.get(key, (None, None))
    if v is None or gm is None or gm == 0:
        continue
    pct     = (v - gm) / abs(gm) * 100
    is_good = (pct >= 0) == INDICES[key]['pos_good']
    labels.append(INDICES[key]['label'])
    vals.append(pct)
    colors.append("#2ecc71" if is_good else "#e74c3c")
    hovers.append(
        f"{INDICES[key]['label']}<br>"
        f"Value: {v:+.5f} {INDICES[key]['units']}<br>"
        f"Global mean: {gm:+.5f} {INDICES[key]['units']}<br>"
        f"Deviation: {pct:+.2f}%"
    )

if labels:
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors, marker_line_width=0,
        customdata=hovers, hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dash", line_width=1)
    fig.update_layout(
        xaxis=dict(color="#8b949e", tickfont=dict(size=10), tickangle=20),
        yaxis=dict(title="% deviation from global mean", color="#8b949e",
                   tickfont=dict(size=10), gridcolor="#21262d"),
        paper_bgcolor="#0f1117", plot_bgcolor="#161b22",
        height=320, margin=dict(l=60, r=20, t=10, b=110),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green = favourable vs global mean · Red = unfavourable · "
        "Bars show % deviation from the global country-average for each index."
    )
else:
    st.info("No index data available — check that all source files are present.")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table + download
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📋 Full Summary Table</div>', unsafe_allow_html=True)

rows = []
for key in OBS_KEYS + FUT_KEYS:
    v, gm = index_results.get(key, (None, None))
    m     = INDICES[key]
    pct   = ""
    if v is not None and gm is not None and gm != 0:
        pct = f"{(v - gm) / abs(gm) * 100:+.2f}%"
    rows.append({
        "Index"       : f"{m['icon']} {m['label']}",
        "Value"       : f"{v:+.5f}" if v is not None else "—",
        "Units"       : m['units'],
        "Global Mean" : f"{gm:+.5f}" if gm is not None else "—",
        "vs Global"   : pct,
        "Source"      : m['source'],
    })

df_out = pd.DataFrame(rows)
st.dataframe(df_out, use_container_width=True, hide_index=True)

st.download_button(
    label     = f"⬇️  Download {display_name} profile as CSV",
    data      = df_out.to_csv(index=False).encode("utf-8"),
    file_name = f"lookup_{canonical.replace(' ', '_')}.csv",
    mime      = "text/csv",
)
