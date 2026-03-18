"""
Dashboard Landing Page
======================
Analysis of Environmental Stresses and Human Development Index

Folder structure expected:
    your_project/
    ├── landing_page.py
    ├── EnvironmentalStress.png          ← optional hero image
    ├── pages/
    │   ├── 1_HDI_Climate.py            → Comparative Analysis
    │   ├── 2_GPM_Map.py                → Precipitation
    │   ├── 3_GRACE_Map.py              → Water Storage
    │   ├── 4_PM25_Map.py               → Air Quality
    │   ├── 5_GISStemp_Map.py           → Temperature
    │   ├── 6_Temp_SSP.py               → Future Temperature
    │   └── 7_Precip_SSP.py             → Future Precipitation
    └── <data files …>

Run:
    streamlit run landing_page.py
"""

import streamlit as st


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Environmental Stresses & HDI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: radial-gradient(ellipse at 20% 10%, #0d1f2d 0%, #071318 60%, #020d10 100%);
        color: #e8f4f8;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: rgba(10, 30, 40, 0.95);
        border-right: 1px solid rgba(0,200,180,0.15);
    }

    /* ── hero ── */
    .hero-title {
        font-family: 'Calibri', 'Candara', 'Segoe UI', sans-serif;
        font-size: 3.5rem;    /* Adjust size as needed (e.g., 48px or 3rem) */
        text-align: center;   /* Use 'center', 'left', or 'right' */
        font-weight: bold;
        line-height: 1.2;     /* Improves readability for multi-line titles */
        margin-bottom: 0.5rem;
    }
    
    .hero-sub {
        font-size: 1rem;
        color: #7ec8d8;
        font-weight: 300;
        letter-spacing: 0.03em;
        margin-bottom: 0;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #00e5cc33, #38b2ff88, #a78bfa33);
        border: none;
        border-radius: 2px;
        margin: 1.8rem 0;
    }

    /* ── section labels ── */
    .section-label {
        font-family: 'Calibri', 'Candara', 'Segoe UI', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a8a9a;
        margin: 1.6rem 0 1rem 0;
    }

    /* ── card grids ── */
    .card-grid-3 {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1.4rem;
    }
    .card-grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.4rem;
    }
    @media (max-width: 900px) {
        .card-grid-3, .card-grid-2 { grid-template-columns: 1fr; }
    }

    /* ── cards ── */
    .dash-card {
        position: relative;
        background: linear-gradient(145deg, rgba(0,200,180,0.07) 0%, rgba(20,50,70,0.6) 100%);
        border: 1px solid rgba(0,200,180,0.18);
        border-radius: 16px;
        padding: 1.8rem 1.6rem 1.5rem;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        overflow: hidden;
        height: 100%;
    }
    .dash-card::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 180px; height: 180px;
        border-radius: 50%;
        opacity: 0.07;
        filter: blur(40px);
    }
    .dash-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,200,180,0.45);
        box-shadow: 0 12px 40px rgba(0,200,180,0.12);
    }

    /* per-card accent blobs */
    .card-hdi::before     { background: #38b2ff; }
    .card-precip::before  { background: #38bdf8; }
    .card-water::before   { background: #818cf8; }
    .card-pm25::before    { background: #f97316; }
    .card-temp::before    { background: #f85149; }
    .card-ftemp::before   { background: #fb923c; }
    .card-fprecip::before { background: #34d399; }

    .card-icon  { font-size: 2.4rem; margin-bottom: 0.65rem; display: block; }
    .card-tag {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 0.85rem;
    }
    .tag-hdi     { background: rgba(56,178,255,0.15);  color: #38b2ff;  border: 1px solid #38b2ff44; }
    .tag-precip  { background: rgba(56,189,248,0.15);  color: #38bdf8;  border: 1px solid #38bdf844; }
    .tag-water   { background: rgba(129,140,248,0.15); color: #818cf8;  border: 1px solid #818cf844; }
    .tag-pm25    { background: rgba(249,115,22,0.15);  color: #f97316;  border: 1px solid #f9731644; }
    .tag-temp    { background: rgba(248,81,73,0.15);   color: #f85149;  border: 1px solid #f8514944; }
    .tag-ftemp   { background: rgba(251,146,60,0.15);  color: #fb923c;  border: 1px solid #fb923c44; }
    .tag-fprecip { background: rgba(52,211,153,0.15);  color: #34d399;  border: 1px solid #34d39944; }

    .card-title {
        font-family: 'Calibri', 'Candara', 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e8f4f8;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        font-size: 0.88rem;
        color: #93b8c8;
        line-height: 1.6;
        margin-bottom: 1.2rem;
    }
    .card-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .chip {
        font-size: 0.72rem;
        color: #7ec8d8;
        background: rgba(126,200,216,0.1);
        border: 1px solid rgba(126,200,216,0.2);
        padding: 2px 9px;
        border-radius: 20px;
    }

    /* ── image panel ── */
    .image-panel {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(0,200,180,0.18);
        background: rgba(10,30,40,0.6);
    }
    .image-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 320px;
        border-radius: 18px;
        border: 1px dashed rgba(0,200,180,0.25);
        background: rgba(10,30,40,0.4);
        color: #4a7a8a;
        font-size: 0.85rem;
        font-family: 'Calibri', 'Candara', 'Segoe UI', sans-serif;
        gap: 0.5rem;
    }

    /* ── info strip ── */
    .info-strip {
        margin-top: 2.5rem;
        padding: 1rem 1.4rem;
        background: rgba(0,200,180,0.05);
        border: 1px solid rgba(0,200,180,0.12);
        border-radius: 12px;
        font-size: 0.84rem;
        color: #7ec8d8;
        line-height: 1.7;
    }
    .info-strip strong { color: #00e5cc; }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    pages = [
        ("🏠", "Home",                    None),
        ("🌍", "Comparative Analysis",    "pages/1_Comparative_Analysis.py"),
        ("🌧️", "Precipitation",           "pages/2_Precipitation.py"),
        ("💧", "Water Storage",           "pages/3_Water_Storage.py"),
        ("💨", "Air Quality",             "pages/4_Air_Quality.py"),
        ("🌡️", "Temperature",             "pages/5_Temperature.py"),
        ("🔮", "Future Temperature",      "pages/6_Future_Temperature.py"),
        ("🌦️", "Future Precipitation",    "pages/7_Future_Precipitation.py"),
    ]



# ═══════════════════════════════════════════════════════════════════════════════
# Hero — title left, image right
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-title">Analysis of Environmental Stresses<br>&amp; Human Development Index</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub" style="margin-top:1rem;">'
    'Interactive dashboards linking environmental stress indicators — '
    'precipitation, water storage, air quality, and temperature — '
    'with global human development outcomes across 190+ countries.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 1 — Observational dashboards (3 cards)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📡 Observational &amp; Historical</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-grid-3">

      <div class="dash-card card-hdi">
        <span class="card-icon">🌍</span>
        <span class="card-tag tag-hdi">Comparative Analysis</span>
        <div class="card-title">Comparative Analysis</div>
        <div class="card-desc">
          Explore how Human Development Index scores correlate with long-term
          environmental stress signals — temperature anomalies, precipitation
          shifts, water storage, and air quality — across 190+ countries.
        </div>
        <div class="card-chips">
          <span class="chip">Temperature</span>
          <span class="chip">Precipitation</span>
          <span class="chip">Water storage</span>
          <span class="chip">PM2.5</span>
          <span class="chip">HDI</span>
        </div>
      </div>

      <div class="dash-card card-precip">
        <span class="card-icon">🌧️</span>
        <span class="card-tag tag-precip">Precipitation</span>
        <div class="card-title">Precipitation</div>
        <div class="card-desc">
          Visualise long-term precipitation trends from NASA GPM IMERG
          (2001–2023) on an interactive world map. Drill into regional patterns
          and examine zonal moisture shifts at pixel level.
        </div>
        <div class="card-chips">
          <span class="chip">GPM IMERG</span>
          <span class="chip">Precipitation</span>
          <span class="chip">2001–2023</span>
        </div>
      </div>

      <div class="dash-card card-water">
        <span class="card-icon">💧</span>
        <span class="card-tag tag-water">Water Storage</span>
        <div class="card-title">Water Storage</div>
        <div class="card-desc">
          Analyse terrestrial water storage trends derived from NASA GRACE and
          GRACE-FO satellite gravimetry. Identify regions facing long-term
          groundwater depletion or accumulation.
        </div>
        <div class="card-chips">
          <span class="chip">GRACE</span>
          <span class="chip">GRACE-FO</span>
          <span class="chip">Water storage</span>
          <span class="chip">2003-2023</span>
        </div>
      </div>

    </div>
    """,
    unsafe_allow_html=True,
)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Comparative Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/1_HDI_Climate.py")
with r1c2:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Precipitation Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/2_GPM_Map.py")
with r1c3:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Water Storage Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/3_GRACE_Map.py")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 2 — Air quality & temperature (2 cards)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🛰️ Satellite &amp; Surface Observations</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-grid-2">

      <div class="dash-card card-pm25">
        <span class="card-icon">💨</span>
        <span class="card-tag tag-pm25">Air Quality</span>
        <div class="card-title">Air Quality</div>
        <div class="card-desc">
          Visualise fine-particulate-matter (PM2.5) trends from NASA models using satelite data on an interactive world map. Identify pollution hotspots
          and examine long-term air quality trajectories by region.
        </div>
        <div class="card-chips">
          <span class="chip">PM2.5</span>
          <span class="chip">Satellite</span>
        </div>
      </div>

      <div class="dash-card card-temp">
        <span class="card-icon">🌡️</span>
        <span class="card-tag tag-temp">Temperature</span>
        <div class="card-title">Temperature</div>
        <div class="card-desc">
          Explore NASA surface temperature change from 1980–2025 on an
          interactive heatmap. Analyse warming patterns by latitude, longitude,
          and region against the long-term baseline.
        </div>
        <div class="card-chips">
          <span class="chip">NASA GISS</span>
          <span class="chip">Surface temperature</span>
          <span class="chip">1980–2025</span>
        </div>
      </div>

    </div>
    """,
    unsafe_allow_html=True,
)

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Air Quality Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/4_PM25_Map.py")
with r2c2:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Temperature Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/5_GISStemp_Map.py")

# ═══════════════════════════════════════════════════════════════════════════════
# Row 3 — CMIP6 future projections (2 cards)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🔭 CMIP6 Future Projections — SSP2-4.5 vs SSP5-8.5</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-grid-2">

      <div class="dash-card card-ftemp">
        <span class="card-icon">🔮</span>
        <span class="card-tag tag-ftemp">Future Temperature · CMIP6</span>
        <div class="card-title">Future Temperature</div>
        <div class="card-desc">
          Compare projected near-surface air temperature trends between
          moderate and high emission scenarios. Difference maps highlight regions facing
          accelerated warming under the higher-emission pathway.
        </div>
        <div class="card-chips">
          <span class="chip">SSP2</span>
          <span class="chip">SSP5</span>
          <span class="chip">CMIP6</span>
        </div>
      </div>

      <div class="dash-card card-fprecip">
        <span class="card-icon">🌦️</span>
        <span class="card-tag tag-fprecip">Future Precipitation · CMIP6</span>
        <div class="card-title">Future Precipitation</div>
        <div class="card-desc">
          Compare projected precipitation trend slopes between the moderate 
          and high-emission scenarios. Includes difference maps, zonal
          analysis, and wet/dry pixel breakdowns.
        </div>
        <div class="card-chips">
          <span class="chip">SSP2</span>
          <span class="chip">SSP5</span>
          <span class="chip">CMIP6</span>
        </div>
      </div>

    </div>
    """,
    unsafe_allow_html=True,
)

r3c1, r3c2 = st.columns(2)
with r3c1:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Future Temperature Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/6_Temp_SSP.py")
with r3c2:
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("↗ Open Future Precipitation Dashboard", use_container_width=True, type="primary"):
        st.switch_page("pages/7_Precip_SSP.py")

# ── info strip ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="info-strip">
        <strong>Data sources:</strong>
        Precipitation — NASA GPM IMERG (2001–2023) &nbsp;·&nbsp;
        Water Storage — NASA GRACE / GRACE-FO &nbsp;·&nbsp;
        Air Quality — Global Satellite PM2.5 (GEE) &nbsp;·&nbsp;
        Temperature — NASA GISTEMP &nbsp;·&nbsp;
        Future Projections — CMIP6 via GEE &nbsp;·&nbsp;
        HDI — UNDP Human Development Report 2025
        <br>
        <strong>Future scenarios:</strong>
        SSP2 (intermediate emissions) &nbsp;·&nbsp; SSP5 (high emissions / fossil-fuel-intensive pathway)
    </div>
    """,
    unsafe_allow_html=True,
)
