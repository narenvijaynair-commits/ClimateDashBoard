"""
GISS Surface Temperature Trend — Interactive NetCDF Dashboard
=============================================================
Install:
    pip install streamlit plotly netCDF4 numpy pandas matplotlib scipy folium streamlit-folium

Run (standalone):
    streamlit run 3_GISS_Temperature.py

As a multi-page app, place in pages/ next to landing_page.py.

Required input file (same directory as the main app):
    - GISS_temperature_trend_1980_2025.nc

NetCDF structure:
    dimensions : lat=90, lon=180
    variable   : TEMPTREND(lat, lon)  [°C change 1980–2025, JJA season]
    source     : NASA GISS / GISTEMP
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
import warnings
warnings.filterwarnings('ignore')

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GISS Temperature Trend",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS (same dark theme as PM2.5 dashboard) ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background: #0f1117; color: #e8e8e8; }

    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #e8e8e8;
    }
    .metric-unit {
        font-size: 12px;
        color: #8b949e;
        margin-left: 4px;
    }
    .metric-pos { color: #f85149; }   /* warming = red  */
    .metric-neg { color: #58a6ff; }   /* cooling = blue */

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 24px !important;
         color: #e8e8e8 !important; letter-spacing: -0.5px; }
    h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #e8e8e8 !important; }

    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-bottom: 1px solid #30363d; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { color: #f85149 !important; border-bottom-color: #f85149 !important; }

    .stSlider > div > div > div { background: #30363d; }
    .stSelectbox > div > div { background: #161b22; border-color: #30363d; }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; font-size: 12px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8e8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
NC_FILE = "GISS_temperature_trend_1980_2025.nc"
MISSING = 9999.0

COLORMAPS = {
    "RdBu (Diverging)"   : "RdBu_r",
    "Coolwarm"           : "coolwarm",
    "Seismic"            : "seismic",
    "RdYlBu (Diverging)" : "RdYlBu_r",
    "PiYG"               : "PiYG",
    "Inferno"            : "inferno",
    "Plasma"             : "plasma",
    "Viridis"            : "viridis",
    "Turbo"              : "turbo",
}

REGIONS = {
    "Global"         : (-180, -90, 180, 90),
    "Arctic (>60°N)" : (-180,  60, 180, 90),
    "South Asia"     : (60,    5,  100, 40),
    "East Asia"      : (100,  15,  145, 55),
    "Southeast Asia" : (95,  -10,  141, 28),
    "Middle East"    : (25,   15,   65, 42),
    "Africa"         : (-20,  -40,   55, 38),
    "Europe"         : (-25,   35,   45, 72),
    "North America"  : (-170,  15,  -50, 75),
    "South America"  : (-82,  -60,  -30, 15),
    "Antarctica"     : (-180, -90,  180, -60),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading NetCDF…")
def load_netcdf(path: str):
    """Read TEMPTREND from NetCDF, return data + coord arrays."""
    ds   = nc.Dataset(path)
    lats = ds.variables["lat"][:]          # shape (90,)
    lons = ds.variables["lon"][:]          # shape (180,)
    raw  = ds.variables["TEMPTREND"][:]    # shape (90, 180)
    ds.close()

    # Convert MaskedArray → plain ndarray, apply missing-value mask
    data = np.array(raw, dtype=float)
    data = np.where(np.abs(data - MISSING) < 1.0, np.nan, data)
    data = np.where(np.abs(data) > 500, np.nan, data)   # catch any other sentinels

    return {
        "data" : data,
        "lats" : np.array(lats, dtype=float),
        "lons" : np.array(lons, dtype=float),
    }


def crop_to_region(dataset, region_bounds):
    """Crop data dict to a lon/lat bounding box."""
    lon_min, lat_min, lon_max, lat_max = region_bounds
    lons, lats, data = dataset["lons"], dataset["lats"], dataset["data"]

    col_mask = (lons >= lon_min) & (lons <= lon_max)
    row_mask = (lats >= lat_min) & (lats <= lat_max)

    cropped = data[np.ix_(row_mask, col_mask)]
    return cropped, lons[col_mask], lats[row_mask]

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════
def make_heatmap(data, lons, lats, cmap_name, vmin, vmax, title=""):
    """Build a Plotly choropleth-style heatmap."""
    fig = go.Figure(go.Heatmap(
        z        = data,
        x        = lons,
        y        = lats,
        zmin     = vmin,
        zmax     = vmax,
        colorscale = cmap_name,
        colorbar = dict(
            title      = dict(text="°C", side="right", font=dict(size=11, color="#8b949e")),
            tickfont   = dict(size=10, color="#8b949e"),
            thickness  = 14,
            len        = 0.75,
            bgcolor    = "rgba(22,27,34,0.8)",
            bordercolor= "#30363d",
            borderwidth= 1,
        ),
        hovertemplate = (
            "Lon: %{x:.1f}°<br>"
            "Lat: %{y:.1f}°<br>"
            "Temp change: %{z:+.2f} °C<extra></extra>"
        ),
    ))

    fig.update_layout(
        title      = dict(text=title, font=dict(family="IBM Plex Mono", size=14, color="#e8e8e8"), x=0.01),
        xaxis      = dict(title="Longitude", showgrid=False, color="#8b949e",
                          tickfont=dict(size=10), zeroline=False),
        yaxis      = dict(title="Latitude",  showgrid=False, color="#8b949e",
                          tickfont=dict(size=10), zeroline=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#0f1117",
        height        = 540,
        margin        = dict(l=50, r=20, t=50, b=40),
    )
    return fig


def make_histogram(data_flat, vmin, vmax, cmap_name, bins=80):
    """Colour-coded histogram of temperature change values."""
    valid = data_flat[~np.isnan(data_flat)]
    valid = valid[(valid >= vmin) & (valid <= vmax)]

    counts, edges = np.histogram(valid, bins=bins, range=(vmin, vmax))
    centres = (edges[:-1] + edges[1:]) / 2

    cmap   = plt.get_cmap(cmap_name)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = [f"rgba{tuple(int(c*255) for c in cmap(norm(v))[:3]) + (0.85,)}" for v in centres]

    fig = go.Figure(go.Bar(
        x          = centres,
        y          = counts,
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "ΔT: %{x:+.2f} °C<br>Count: %{y:,}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color="#f85149", line_dash="dash", line_width=1.5,
                  annotation_text="0 °C", annotation_font_color="#f85149",
                  annotation_font_size=10)

    fig.update_layout(
        xaxis = dict(title="Temperature Change (°C)", color="#8b949e",
                     tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis = dict(title="Grid Cell Count", color="#8b949e",
                     tickfont=dict(size=10), gridcolor="#21262d"),
        bargap         = 0.02,
        paper_bgcolor  = "#0f1117",
        plot_bgcolor   = "#161b22",
        height         = 320,
        margin         = dict(l=50, r=20, t=20, b=40),
        showlegend     = False,
    )
    return fig


def make_zonal_bar(data, lats, title="Zonal Mean Temperature Change"):
    """Mean temperature change by latitude band."""
    zonal = np.nanmean(data, axis=1)
    valid = ~np.isnan(zonal)
    colors = ["#f85149" if v > 0 else "#58a6ff" for v in zonal[valid]]

    fig = go.Figure(go.Bar(
        x          = zonal[valid],
        y          = lats[valid],
        orientation= "h",
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "Lat: %{y:.1f}°<br>Mean ΔT: %{x:+.2f} °C<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#8b949e", line_dash="dash", line_width=1)
    fig.update_layout(
        title      = dict(text=title, font=dict(family="IBM Plex Mono", size=12, color="#8b949e")),
        xaxis      = dict(title="Mean ΔT (°C)", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis      = dict(title="Latitude", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#161b22",
        height        = 460,
        margin        = dict(l=60, r=20, t=40, b=40),
    )
    return fig


def make_longitudinal_bar(data, lons, title="Longitudinal Mean Temperature Change"):
    """Mean temperature change by longitude band."""
    longitudinal = np.nanmean(data, axis=0)
    valid        = ~np.isnan(longitudinal)
    colors = ["#f85149" if v > 0 else "#58a6ff" for v in longitudinal[valid]]

    fig = go.Figure(go.Bar(
        x          = lons[valid],
        y          = longitudinal[valid],
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "Lon: %{x:.1f}°<br>Mean ΔT: %{y:+.2f} °C<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dash", line_width=1)
    fig.update_layout(
        title      = dict(text=title, font=dict(family="IBM Plex Mono", size=12, color="#8b949e")),
        xaxis      = dict(title="Longitude", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis      = dict(title="Mean ΔT (°C)", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#161b22",
        height        = 320,
        margin        = dict(l=60, r=20, t=40, b=40),
    )
    return fig


def make_scatter_map(data, lons, lats, cmap_name, vmin, vmax, title=""):
    """Plotly scatter_geo for an alternative geographic projection view."""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_vals = data.ravel()
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()

    mask = ~np.isnan(flat_vals)
    flat_vals = flat_vals[mask]
    flat_lons = flat_lons[mask]
    flat_lats = flat_lats[mask]

    fig = go.Figure(go.Scattergeo(
        lon  = flat_lons,
        lat  = flat_lats,
        mode = "markers",
        marker = dict(
            color      = flat_vals,
            colorscale = cmap_name,
            cmin       = vmin,
            cmax       = vmax,
            size       = 4,
            opacity    = 0.85,
            colorbar   = dict(
                title    = dict(text="ΔT (°C)", font=dict(color="#8b949e", size=11)),
                tickfont = dict(color="#8b949e", size=10),
                bgcolor  = "rgba(22,27,34,0.8)",
                bordercolor="#30363d",
                borderwidth=1,
            ),
        ),
        hovertemplate = "Lon: %{lon:.1f}°<br>Lat: %{lat:.1f}°<br>ΔT: %{marker.color:+.2f} °C<extra></extra>",
    ))
    fig.update_geos(
        showland   = True, landcolor  = "#1c2128",
        showocean  = True, oceancolor = "#0f1117",
        showframe  = False,
        showcountries = True, countrycolor = "#30363d",
        showcoastlines= True, coastlinecolor = "#30363d",
        projection_type = "natural earth",
        bgcolor    = "#0f1117",
    )
    fig.update_layout(
        title         = dict(text=title, font=dict(family="IBM Plex Mono", size=13, color="#e8e8e8"), x=0.01),
        paper_bgcolor = "#0f1117",
        height        = 500,
        margin        = dict(l=0, r=0, t=45, b=0),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌡️ GISS Temperature Trend Explorer")
st.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:12px;color:#8b949e;'>"
    "NASA GISTEMP · JJA Surface Temperature Change 1980–2025 &nbsp;·&nbsp; °C total change"
    "</span>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Guard: check dependencies and file
# ═══════════════════════════════════════════════════════════════════════════════
if not HAS_NC:
    st.error("**netCDF4** is required. Install with: `pip install netCDF4`")
    st.stop()

if not Path(NC_FILE).exists():
    st.error(
        f"**{NC_FILE}** not found in the current directory.\n\n"
        "Ensure the NetCDF file is in the same folder as the main app (landing_page.py)."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar controls
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    st.markdown("---")

    # Region
    region_name   = st.selectbox("Region / Zoom", list(REGIONS.keys()), index=0)
    region_bounds = REGIONS[region_name]

    st.markdown("---")

    # Colourmap
    cmap_label = st.selectbox("Colour palette", list(COLORMAPS.keys()), index=0)
    cmap_name  = COLORMAPS[cmap_label]

    reverse_cmap = st.checkbox("Reverse palette", value=False)
    if reverse_cmap:
        cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name[:-2]

    st.markdown("---")

    # Value clipping
    st.markdown("**Colour range clipping**")
    use_percentile = st.checkbox("Clip by percentile", value=True)
    if use_percentile:
        pct_low  = st.slider("Lower percentile", 0,  49,  2)
        pct_high = st.slider("Upper percentile", 51, 100, 98)
    else:
        manual_min = st.number_input("Min value (°C)", value=-3.0, step=0.5, format="%.1f")
        manual_max = st.number_input("Max value (°C)", value= 5.0, step=0.5, format="%.1f")

    st.markdown("---")

    # Smoothing
    smooth = st.checkbox("Apply Gaussian smoothing", value=False)
    if smooth:
        sigma = st.slider("Smoothing sigma (grid cells)", 1, 8, 2)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:IBM Plex Mono;font-size:10px;color:#8b949e;'>"
        "Source: NASA GISS / GISTEMP<br>JJA L-OTI · 1200 km smoothing radius<br>1980–2025"
        "</span>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Load and process data
# ═══════════════════════════════════════════════════════════════════════════════
dataset = load_netcdf(NC_FILE)
data_crop, lons_crop, lats_crop = crop_to_region(dataset, region_bounds)

# Optional smoothing
if smooth and HAS_SCIPY:
    data_smooth = ndimage.gaussian_filter(
        np.where(np.isnan(data_crop), 0, data_crop), sigma=sigma
    )
    data_display = np.where(np.isnan(data_crop), np.nan, data_smooth)
elif smooth and not HAS_SCIPY:
    st.sidebar.warning("scipy not installed — smoothing unavailable.")
    data_display = data_crop
else:
    data_display = data_crop

# Colour range
valid_flat = data_display[~np.isnan(data_display)].ravel()
if use_percentile and len(valid_flat) > 0:
    vmin = float(np.percentile(valid_flat, pct_low))
    vmax = float(np.percentile(valid_flat, pct_high))
else:
    vmin = manual_min
    vmax = manual_max

make_symmetric = st.sidebar.checkbox("Force symmetric colour range", value=True)
if make_symmetric:
    extreme = max(abs(vmin), abs(vmax))
    vmin, vmax = -extreme, extreme

# ═══════════════════════════════════════════════════════════════════════════════
# Summary metrics
# ═══════════════════════════════════════════════════════════════════════════════
valid_all   = dataset["data"][~np.isnan(dataset["data"])].ravel()
valid_crop  = data_crop[~np.isnan(data_crop)].ravel()

pct_warming = 100 * np.mean(valid_all > 0)
pct_cooling = 100 * np.mean(valid_all < 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mean ΔT",         f"{np.nanmean(valid_all):+.2f}",   "°C")
col2.metric("Median ΔT",       f"{np.nanmedian(valid_all):+.2f}", "°C")
col3.metric("Std dev",         f"{np.nanstd(valid_all):.2f}",     "°C")
col4.metric("Max warming",     f"{np.nanmax(valid_all):+.2f}",    "°C")
col5.metric("Max cooling",     f"{np.nanmin(valid_all):+.2f}",    "°C")
col6.metric("Warming cells",   f"{pct_warming:.1f}%",             f"vs {pct_cooling:.1f}% cooling")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Main tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_map, tab_geo, tab_hist, tab_zonal, tab_stats, tab_export = st.tabs([
    "🗺️  Heatmap", "🌍  Globe View", "📊  Histogram",
    "🌐  Zonal Analysis", "📋  Statistics", "💾  Export"
])

# ── Tab 1: Heatmap ────────────────────────────────────────────────────────────
with tab_map:
    map_title = f"GISS JJA Temperature Change 1980–2025 — {region_name}"
    fig_map   = make_heatmap(data_display, lons_crop, lats_crop,
                             cmap_name, vmin, vmax, title=map_title)
    st.plotly_chart(fig_map, use_container_width=True)

    if region_name != "Global":
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(f"{region_name} — Mean",   f"{np.nanmean(valid_crop):+.2f} °C")
        rc2.metric(f"{region_name} — Median", f"{np.nanmedian(valid_crop):+.2f} °C")
        rc3.metric(f"{region_name} — Max",    f"{np.nanmax(valid_crop):+.2f} °C")
        rc4.metric(f"{region_name} — Min",    f"{np.nanmin(valid_crop):+.2f} °C")

    st.caption(
        f"Grid: {dataset['data'].shape[1]}×{dataset['data'].shape[0]} (180×90) · "
        f"Resolution: 2° × 2° · "
        f"Colour range: [{vmin:.2f}, {vmax:.2f}] °C"
    )

# ── Tab 2: Globe view ─────────────────────────────────────────────────────────
with tab_geo:
    st.markdown('<div class="section-header">Natural Earth Projection</div>', unsafe_allow_html=True)
    fig_geo = make_scatter_map(
        data_display, lons_crop, lats_crop, cmap_name, vmin, vmax,
        title=f"GISS Temperature Change — {region_name}"
    )
    st.plotly_chart(fig_geo, use_container_width=True)
    st.caption(
        "Each marker represents one 2°×2° grid cell. "
        "Missing-value cells (ocean/ice gaps) are omitted."
    )

# ── Tab 3: Histogram ─────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="section-header">Value Distribution</div>', unsafe_allow_html=True)

    bins     = st.slider("Number of bins", 20, 200, 60, key="hist_bins")
    fig_hist = make_histogram(valid_crop, vmin, vmax, cmap_name, bins=bins)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Percentile table
    st.markdown('<div class="section-header">Percentile Summary</div>', unsafe_allow_html=True)
    pcts     = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.nanpercentile(valid_crop, pcts)
    pct_df   = pd.DataFrame({
        "Percentile"  : [f"P{p}" for p in pcts],
        "Value (°C)"  : [f"{v:+.3f}" for v in pct_vals],
        "Direction"   : ["🔴 Warming" if v > 0 else "🔵 Cooling" for v in pct_vals],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

# ── Tab 4: Zonal analysis ─────────────────────────────────────────────────────
with tab_zonal:
    st.markdown('<div class="section-header">Zonal Mean (by Latitude)</div>', unsafe_allow_html=True)
    fig_zonal = make_zonal_bar(
        data_display, lats_crop,
        title="Mean Temperature Change by Latitude Band"
    )
    st.plotly_chart(fig_zonal, use_container_width=True)

    st.markdown('<div class="section-header">Longitudinal Mean (by Longitude)</div>', unsafe_allow_html=True)
    fig_lon = make_longitudinal_bar(
        data_display, lons_crop,
        title="Mean Temperature Change by Longitude Band"
    )
    st.plotly_chart(fig_lon, use_container_width=True)

# ── Tab 5: Statistics ─────────────────────────────────────────────────────────
with tab_stats:
    st.markdown('<div class="section-header">Global Statistics</div>', unsafe_allow_html=True)

    stat_rows = [
        ("Mean ΔT",              f"{np.nanmean(valid_all):+.4f}",    "°C"),
        ("Median ΔT",            f"{np.nanmedian(valid_all):+.4f}",  "°C"),
        ("Std deviation",        f"{np.nanstd(valid_all):.4f}",      "°C"),
        ("Min (max cooling)",    f"{np.nanmin(valid_all):+.4f}",     "°C"),
        ("Max (max warming)",    f"{np.nanmax(valid_all):+.4f}",     "°C"),
        ("% cells warming",      f"{pct_warming:.2f}",               "%"),
        ("% cells cooling",      f"{pct_cooling:.2f}",               "%"),
        ("Valid cells",          f"{len(valid_all):,}",              "cells"),
        ("Grid resolution",      "2° × 2°",                          ""),
        ("Grid shape (lon×lat)", f"{dataset['data'].shape[1]}×{dataset['data'].shape[0]}", ""),
        ("Period",               "1980 – 2025",                      ""),
        ("Season",               "JJA (Jun–Jul–Aug)",                ""),
        ("Source",               "NASA GISS GISTEMP",                ""),
        ("Smoothing radius",     "1200 km",                          ""),
    ]

    stats_df = pd.DataFrame(stat_rows, columns=["Statistic", "Value", "Unit"])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    if region_name != "Global":
        st.markdown(
            f'<div class="section-header">{region_name} Region Statistics</div>',
            unsafe_allow_html=True,
        )
        region_rows = [
            ("Mean ΔT",   f"{np.nanmean(valid_crop):+.4f}",   "°C"),
            ("Median ΔT", f"{np.nanmedian(valid_crop):+.4f}", "°C"),
            ("Std",       f"{np.nanstd(valid_crop):.4f}",     "°C"),
            ("Min",       f"{np.nanmin(valid_crop):+.4f}",    "°C"),
            ("Max",       f"{np.nanmax(valid_crop):+.4f}",    "°C"),
            ("N cells",   f"{len(valid_crop):,}",             "cells"),
        ]
        st.dataframe(
            pd.DataFrame(region_rows, columns=["Statistic", "Value", "Unit"]),
            use_container_width=True, hide_index=True,
        )

# ── Tab 6: Export ─────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export Current View</div>', unsafe_allow_html=True)
    st.markdown("Download the **currently displayed region** as a CSV or PNG.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📄 CSV — grid cell values**")
        st.caption("Exports a flat table of lon, lat, temperature change for the current region.")
        lon_grid, lat_grid = np.meshgrid(lons_crop, lats_crop)
        export_df = pd.DataFrame({
            "longitude"        : lon_grid.ravel(),
            "latitude"         : lat_grid.ravel(),
            "temp_change_degC" : data_display.ravel(),
        }).dropna()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download CSV",
            data      = csv_bytes,
            file_name = f"giss_temp_trend_{region_name.lower().replace(' ','_')}.csv",
            mime      = "text/csv",
        )
        st.metric("Rows in export", f"{len(export_df):,}")

    with col_b:
        st.markdown("**🖼️ PNG — map image**")
        st.caption("Exports the current heatmap as a static high-resolution PNG.")

        dpi = st.select_slider("DPI", [72, 150, 300], value=150, key="dpi_giss")

        if st.button("Generate PNG"):
            fig_png, ax = plt.subplots(figsize=(14, 7), facecolor="#0f1117")
            ax.set_facecolor("#0f1117")
            cmap_obj = plt.get_cmap(cmap_name)
            cmap_obj.set_bad(color="#0f1117")
            im = ax.imshow(
                data_display,
                extent = [lons_crop[0], lons_crop[-1], lats_crop[-1], lats_crop[0]],
                cmap   = cmap_obj,
                vmin   = vmin,
                vmax   = vmax,
                aspect = "auto",
                origin = "upper",
            )
            cb = fig_png.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label("ΔT (°C)", color="#8b949e", fontsize=10)
            cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

            ax.set_title(
                f"GISS JJA Temperature Change 1980–2025 — {region_name}",
                color="#e8e8e8", fontsize=13, pad=12, fontfamily="monospace"
            )
            ax.set_xlabel("Longitude", color="#8b949e", fontsize=9)
            ax.set_ylabel("Latitude",  color="#8b949e", fontsize=9)
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            buf = io.BytesIO()
            fig_png.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                            facecolor="#0f1117")
            buf.seek(0)
            plt.close(fig_png)

            st.download_button(
                label     = "⬇️ Download PNG",
                data      = buf,
                file_name = f"giss_temp_trend_{region_name.lower().replace(' ','_')}.png",
                mime      = "image/png",
            )
            st.success("PNG ready — click the button above to save.")
