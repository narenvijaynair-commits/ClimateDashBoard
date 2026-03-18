"""
Precipitation Slope SSP5-8.5 — Interactive GeoTIFF Dashboard
=============================================================
Install:
    pip install streamlit plotly rasterio numpy pandas matplotlib scipy

Run:
    streamlit run pr_slope_dashboard.py

Required input file (same directory):
    - pr_slope_ssp585.tif
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from pathlib import Path
import io
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Precipitation Slope SSP5-8.5",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #0a1628; color: #cdd9e5; }

    section[data-testid="stSidebar"] {
        background: #0d1f3c;
        border-right: 1px solid #1b3358;
    }

    h1 { font-family: 'Space Mono', monospace !important;
         font-size: 22px !important; color: #cdd9e5 !important;
         letter-spacing: -0.5px; }
    h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #cdd9e5 !important; }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: #6e8aab;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1b3358;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    .stTabs [data-baseweb="tab-list"] { background: #0d1f3c; border-bottom: 1px solid #1b3358; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        font-size: 11px; color: #6e8aab;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { color: #4fc3f7 !important; border-bottom-color: #4fc3f7 !important; }

    div[data-testid="stMetric"] {
        background: #0d1f3c;
        border: 1px solid #1b3358;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #6e8aab !important; font-size: 11px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #cdd9e5 !important; }

    .stSelectbox > div > div { background: #0d1f3c; border-color: #1b3358; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
TIFF_FILE = "pr_slope_ssp585.tif"

COLORMAPS = {
    "BrBG (Diverging)"    : "BrBG",
    "RdBu (Diverging)"    : "RdBu",
    "PRGn (Diverging)"    : "PRGn",
    "PiYG (Diverging)"    : "PiYG",
    "Coolwarm"            : "coolwarm",
    "Blues"               : "Blues",
    "Viridis"             : "viridis",
    "Turbo"               : "turbo",
    "Plasma"              : "plasma",
}

REGIONS = {
    "Global"            : (-180, -90,  180,  90),
    "South Asia"        : (60,    5,   100,  40),
    "East Asia"         : (100,  15,   145,  55),
    "Southeast Asia"    : (95,  -10,   141,  28),
    "Middle East & N Africa": (20, 10,  65,  42),
    "Sub-Saharan Africa": (-20, -40,    55,  20),
    "Europe"            : (-25,  35,    45,  72),
    "North America"     : (-170, 15,   -50,  75),
    "South America"     : (-82, -60,   -30,  15),
    "Australia & Oceania": (110, -50,  180,  10),
    "Arctic (>60°N)"    : (-180, 60,   180,  90),
    "Tropics (±23.5°)"  : (-180, -23.5, 180, 23.5),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading GeoTIFF…")
def load_tiff(path: str, downsample: int = 4):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(
                max(1, src.height // downsample),
                max(1, src.width  // downsample),
            ),
            resampling=rasterio.enums.Resampling.average,
        ).astype(float)

        bounds     = src.bounds
        nodata     = src.nodata
        crs        = str(src.crs)
        orig_shape = (src.height, src.width)
        res        = src.res
        tags       = src.tags()

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.abs(data) > 1e10, np.nan, data)

    lons = np.linspace(bounds.left,  bounds.right,  data.shape[1])
    lats = np.linspace(bounds.top,   bounds.bottom, data.shape[0])

    return {
        "data"       : data,
        "lons"       : lons,
        "lats"       : lats,
        "bounds"     : bounds,
        "crs"        : crs,
        "orig_shape" : orig_shape,
        "res"        : res,
        "tags"       : tags,
    }

def crop_to_region(raster, region_bounds):
    lon_min, lat_min, lon_max, lat_max = region_bounds
    lons, lats, data = raster["lons"], raster["lats"], raster["data"]
    col_mask = (lons >= lon_min) & (lons <= lon_max)
    row_mask = (lats >= lat_min) & (lats <= lat_max)
    return data[np.ix_(row_mask, col_mask)], lons[col_mask], lats[row_mask]

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════
BG      = "#0a1628"
BG2     = "#0d1f3c"
BORDER  = "#1b3358"
TEXT    = "#cdd9e5"
MUTED   = "#6e8aab"
WET     = "#4fc3f7"   # wetter  → blue
DRY     = "#ef9a9a"   # drier   → red/coral

def make_heatmap(data, lons, lats, cmap_name, vmin, vmax, units, title=""):
    fig = go.Figure(go.Heatmap(
        z        = data,
        x        = lons,
        y        = lats,
        zmin     = vmin,
        zmax     = vmax,
        colorscale = cmap_name,
        colorbar = dict(
            title      = dict(text=units, side="right",
                              font=dict(size=11, color=MUTED)),
            tickfont   = dict(size=10, color=MUTED),
            thickness  = 14,
            len        = 0.75,
            bgcolor    = f"rgba(13,31,60,0.85)",
            bordercolor= BORDER,
            borderwidth= 1,
        ),
        hovertemplate = (
            f"Lon: %{{x:.2f}}°<br>Lat: %{{y:.2f}}°<br>"
            f"Slope: %{{z:.5f}} {units}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title      = dict(text=title, font=dict(family="Space Mono", size=13, color=TEXT), x=0.01),
        xaxis      = dict(title="Longitude", showgrid=False, color=MUTED,
                          tickfont=dict(size=10), zeroline=False),
        yaxis      = dict(title="Latitude",  showgrid=False, color=MUTED,
                          tickfont=dict(size=10), zeroline=False,
                          scaleanchor="x", scaleratio=1),
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        height        = 540,
        margin        = dict(l=50, r=20, t=50, b=40),
    )
    return fig

def make_histogram(data_flat, vmin, vmax, cmap_name, units, bins=80):
    valid = data_flat[~np.isnan(data_flat)]
    valid = valid[(valid >= vmin) & (valid <= vmax)]
    counts, edges = np.histogram(valid, bins=bins, range=(vmin, vmax))
    centres = (edges[:-1] + edges[1:]) / 2

    cmap   = plt.get_cmap(cmap_name)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = [f"rgba{tuple(int(c*255) for c in cmap(norm(v))[:3]) + (0.85,)}" for v in centres]

    fig = go.Figure(go.Bar(
        x=centres, y=counts, marker_color=colors, marker_line_width=0,
        hovertemplate=f"Slope: %{{x:.5f}} {units}<br>Count: %{{y:,}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=DRY, line_dash="dash", line_width=1.5,
                  annotation_text="0", annotation_font_color=DRY, annotation_font_size=10)
    fig.update_layout(
        xaxis = dict(title=f"Precipitation Slope ({units})", color=MUTED,
                     tickfont=dict(size=10), gridcolor="#142038"),
        yaxis = dict(title="Pixel Count", color=MUTED,
                     tickfont=dict(size=10), gridcolor="#142038"),
        bargap=0.02, paper_bgcolor=BG, plot_bgcolor=BG2,
        height=320, margin=dict(l=50, r=20, t=20, b=40), showlegend=False,
    )
    return fig

def make_zonal_bar(data, lats, units):
    zonal = np.nanmean(data, axis=1)
    valid = ~np.isnan(zonal)
    colors = [WET if v > 0 else DRY for v in zonal[valid]]
    fig = go.Figure(go.Bar(
        x=zonal[valid], y=lats[valid], orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate=f"Lat: %{{y:.1f}}°<br>Mean slope: %{{x:.5f}} {units}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        title  = dict(text="Mean Precipitation Slope by Latitude",
                      font=dict(family="Space Mono", size=12, color=MUTED)),
        xaxis  = dict(title=f"Mean Slope ({units})", color=MUTED,
                      tickfont=dict(size=10), gridcolor="#142038"),
        yaxis  = dict(title="Latitude", color=MUTED,
                      tickfont=dict(size=10), gridcolor="#142038"),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        height=430, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig

def make_longitudinal_bar(data, lons, units):
    longitudinal = np.nanmean(data, axis=0)
    valid        = ~np.isnan(longitudinal)
    colors = [WET if v > 0 else DRY for v in longitudinal[valid]]
    fig = go.Figure(go.Bar(
        x=lons[valid], y=longitudinal[valid],
        marker_color=colors, marker_line_width=0,
        hovertemplate=f"Lon: %{{x:.1f}}°<br>Mean slope: %{{y:.5f}} {units}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        title  = dict(text="Mean Precipitation Slope by Longitude",
                      font=dict(family="Space Mono", size=12, color=MUTED)),
        xaxis  = dict(title="Longitude", color=MUTED,
                      tickfont=dict(size=10), gridcolor="#142038"),
        yaxis  = dict(title=f"Mean Slope ({units})", color=MUTED,
                      tickfont=dict(size=10), gridcolor="#142038"),
        paper_bgcolor=BG, plot_bgcolor=BG2,
        height=320, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig

def make_wetdry_pie(data_flat):
    valid   = data_flat[~np.isnan(data_flat)]
    n_wet   = int(np.sum(valid > 0))
    n_dry   = int(np.sum(valid < 0))
    n_zero  = int(np.sum(valid == 0))
    fig = go.Figure(go.Pie(
        labels  = ["Wetter (↑)", "Drier (↓)", "No change"],
        values  = [n_wet, n_dry, n_zero],
        marker  = dict(colors=[WET, DRY, MUTED]),
        textfont= dict(size=12, color=TEXT),
        hole    = 0.45,
        hovertemplate="%{label}: %{value:,} px (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG, height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(font=dict(color=MUTED, size=11), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Guards
# ═══════════════════════════════════════════════════════════════════════════════
if not HAS_RASTERIO:
    st.error("**rasterio** is required. Run: `pip install rasterio`")
    st.stop()

if not Path(TIFF_FILE).exists():
    st.error(
        f"**{TIFF_FILE}** not found in the current directory.\n\n"
        "Make sure the file has been exported from GEE and placed alongside this script."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌧️ Precipitation Slope — SSP5-8.5")
st.markdown(
    "<span style='font-family:Space Mono;font-size:11px;color:#6e8aab;'>"
    "Interactive exploration of projected precipitation trend slope &nbsp;·&nbsp; SSP5-8.5 scenario"
    "</span>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    downsample = st.select_slider(
        "Resolution (lower = faster)",
        options=[1, 2, 4, 8, 16], value=4,
        help="Downsampling factor — 1 = native resolution (may be slow)."
    )

    st.markdown("---")

    region_name   = st.selectbox("Region / Zoom", list(REGIONS.keys()), index=0)
    region_bounds = REGIONS[region_name]

    st.markdown("---")

    cmap_label = st.selectbox("Colour palette", list(COLORMAPS.keys()), index=0)
    cmap_name  = COLORMAPS[cmap_label]
    if st.checkbox("Reverse palette", value=False):
        cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name[:-2]

    st.markdown("---")

    st.markdown("**Colour range clipping**")
    use_percentile = st.checkbox("Clip by percentile", value=True)
    if use_percentile:
        pct_low  = st.slider("Lower percentile", 0, 49,  2)
        pct_high = st.slider("Upper percentile", 51, 100, 98)
    else:
        manual_min = st.number_input("Min value", value=-0.001, step=0.0001, format="%.5f")
        manual_max = st.number_input("Max value", value= 0.001, step=0.0001, format="%.5f")

    make_symmetric = st.checkbox("Force symmetric colour range", value=True)

    st.markdown("---")

    smooth = st.checkbox("Apply Gaussian smoothing", value=False)
    if smooth:
        sigma = st.slider("Smoothing sigma (pixels)", 1, 10, 3)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:Space Mono;font-size:9px;color:#6e8aab;'>"
        "Scenario: SSP5-8.5<br>Variable: Precipitation slope<br>"
        "Source: CMIP6 / GEE export"
        "</span>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
raster = load_tiff(TIFF_FILE, downsample=downsample)

# Auto-detect units from magnitude
global_vals  = raster["data"][~np.isnan(raster["data"])].ravel()
median_abs   = np.nanmedian(np.abs(global_vals))
if median_abs < 0.01:
    units = "mm/day/yr"
elif median_abs < 10:
    units = "mm/month/yr"
else:
    units = "mm/yr"

data_crop, lons_crop, lats_crop = crop_to_region(raster, region_bounds)

# Smoothing
if smooth and HAS_SCIPY:
    smoothed = ndimage.gaussian_filter(np.where(np.isnan(data_crop), 0, data_crop), sigma=sigma)
    data_display = np.where(np.isnan(data_crop), np.nan, smoothed)
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
    vmin, vmax = manual_min, manual_max

if make_symmetric:
    extreme = max(abs(vmin), abs(vmax))
    vmin, vmax = -extreme, extreme

# ═══════════════════════════════════════════════════════════════════════════════
# Metrics row
# ═══════════════════════════════════════════════════════════════════════════════
valid_all    = global_vals
pct_wetter   = 100 * np.mean(valid_all > 0)
pct_drier    = 100 * np.mean(valid_all < 0)
valid_crop_f = data_crop[~np.isnan(data_crop)].ravel()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Global mean slope",   f"{np.nanmean(valid_all):+.5f}",   units)
c2.metric("Global median",       f"{np.nanmedian(valid_all):+.5f}", units)
c3.metric("Std deviation",       f"{np.nanstd(valid_all):.5f}",     units)
c4.metric("Max (wettest trend)", f"{np.nanmax(valid_all):+.5f}",    units)
c5.metric("Min (driest trend)",  f"{np.nanmin(valid_all):+.5f}",    units)
c6.metric("Wetter pixels",       f"{pct_wetter:.1f}%",              f"vs {pct_drier:.1f}% drier")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_map, tab_hist, tab_zonal, tab_breakdown, tab_stats, tab_export = st.tabs([
    "🗺️  Map", "📊  Histogram", "🌐  Zonal Analysis",
    "🔵  Wet / Dry Breakdown", "📋  Statistics", "💾  Export"
])

# ── Map ───────────────────────────────────────────────────────────────────────
with tab_map:
    fig_map = make_heatmap(
        data_display, lons_crop, lats_crop, cmap_name, vmin, vmax, units,
        title=f"Precipitation Slope (SSP5-8.5) — {region_name}",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    if region_name != "Global" and len(valid_crop_f) > 0:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(f"{region_name} — Mean",   f"{np.nanmean(valid_crop_f):+.5f} {units}")
        rc2.metric(f"{region_name} — Median", f"{np.nanmedian(valid_crop_f):+.5f} {units}")
        rc3.metric(f"{region_name} — Max",    f"{np.nanmax(valid_crop_f):+.5f} {units}")
        rc4.metric(f"{region_name} — Min",    f"{np.nanmin(valid_crop_f):+.5f} {units}")

    st.caption(
        f"CRS: {raster['crs']} · "
        f"Native: {raster['orig_shape'][1]}×{raster['orig_shape'][0]} px · "
        f"Display ×{downsample} downsampled · "
        f"Colour range: [{vmin:.5f}, {vmax:.5f}] {units}"
    )

# ── Histogram ─────────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="section-header">Value Distribution</div>', unsafe_allow_html=True)
    bins = st.slider("Number of bins", 20, 200, 80, key="hist_bins")
    st.plotly_chart(make_histogram(valid_flat, vmin, vmax, cmap_name, units, bins=bins),
                    use_container_width=True)

    st.markdown('<div class="section-header">Percentile Summary</div>', unsafe_allow_html=True)
    pcts     = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.nanpercentile(valid_flat, pcts)
    pct_df   = pd.DataFrame({
        "Percentile"  : [f"P{p}" for p in pcts],
        f"Value ({units})": [f"{v:+.6f}" for v in pct_vals],
        "Direction"   : ["↑ Wetter" if v > 0 else "↓ Drier" for v in pct_vals],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

# ── Zonal analysis ────────────────────────────────────────────────────────────
with tab_zonal:
    st.markdown('<div class="section-header">Zonal Mean (by Latitude)</div>', unsafe_allow_html=True)
    st.plotly_chart(make_zonal_bar(data_display, lats_crop, units), use_container_width=True)

    st.markdown('<div class="section-header">Longitudinal Mean (by Longitude)</div>', unsafe_allow_html=True)
    st.plotly_chart(make_longitudinal_bar(data_display, lons_crop, units), use_container_width=True)

# ── Wet / Dry breakdown ───────────────────────────────────────────────────────
with tab_breakdown:
    st.markdown('<div class="section-header">Wetter vs Drier Pixel Breakdown</div>',
                unsafe_allow_html=True)

    b1, b2 = st.columns([1, 2])
    with b1:
        st.plotly_chart(make_wetdry_pie(valid_flat), use_container_width=True)

    with b2:
        n_wet  = int(np.sum(valid_flat > 0))
        n_dry  = int(np.sum(valid_flat < 0))
        n_zero = int(np.sum(valid_flat == 0))
        total  = len(valid_flat)

        breakdown_df = pd.DataFrame([
            {"Category": "Wetter (↑)",   "Pixels": n_wet,  "Percent": f"{100*n_wet/total:.2f}%",
             "Mean slope": f"{np.mean(valid_flat[valid_flat > 0]):+.6f}" if n_wet > 0 else "N/A"},
            {"Category": "Drier (↓)",    "Pixels": n_dry,  "Percent": f"{100*n_dry/total:.2f}%",
             "Mean slope": f"{np.mean(valid_flat[valid_flat < 0]):+.6f}" if n_dry > 0 else "N/A"},
            {"Category": "No change",    "Pixels": n_zero, "Percent": f"{100*n_zero/total:.2f}%",
             "Mean slope": "0.000000"},
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(
            f"**Wetter–Drier ratio:** `{n_wet/max(n_dry,1):.2f}` "
            f"({'more wetter pixels' if n_wet > n_dry else 'more drier pixels'})"
        )
        st.markdown(
            f"**Mean slope (wetter pixels):** "
            f"`{np.mean(valid_flat[valid_flat > 0]):+.6f} {units}`" if n_wet > 0 else ""
        )
        st.markdown(
            f"**Mean slope (drier pixels):** "
            f"`{np.mean(valid_flat[valid_flat < 0]):+.6f} {units}`" if n_dry > 0 else ""
        )

# ── Statistics ────────────────────────────────────────────────────────────────
with tab_stats:
    st.markdown('<div class="section-header">Global Statistics</div>', unsafe_allow_html=True)
    stat_rows = [
        ("Mean slope",              f"{np.nanmean(valid_all):+.7f}",    units),
        ("Median slope",            f"{np.nanmedian(valid_all):+.7f}",  units),
        ("Std deviation",           f"{np.nanstd(valid_all):.7f}",      units),
        ("Min slope (driest)",      f"{np.nanmin(valid_all):+.7f}",     units),
        ("Max slope (wettest)",     f"{np.nanmax(valid_all):+.7f}",     units),
        ("% pixels wetter",         f"{pct_wetter:.3f}",                "%"),
        ("% pixels drier",          f"{pct_drier:.3f}",                 "%"),
        ("Valid pixels (global)",   f"{len(valid_all):,}",              "px"),
        ("Native resolution",       f"{raster['orig_shape'][1]}×{raster['orig_shape'][0]}", "px"),
        ("CRS",                     raster["crs"],                      ""),
        ("Bounds W",                f"{raster['bounds'].left:.4f}",     "°"),
        ("Bounds E",                f"{raster['bounds'].right:.4f}",    "°"),
        ("Bounds S",                f"{raster['bounds'].bottom:.4f}",   "°"),
        ("Bounds N",                f"{raster['bounds'].top:.4f}",      "°"),
        ("Auto-detected units",     units,                              ""),
    ]
    st.dataframe(pd.DataFrame(stat_rows, columns=["Statistic", "Value", "Unit"]),
                 use_container_width=True, hide_index=True)

    if region_name != "Global" and len(valid_crop_f) > 0:
        st.markdown(f'<div class="section-header">{region_name} Statistics</div>',
                    unsafe_allow_html=True)
        region_rows = [
            ("Mean",     f"{np.nanmean(valid_crop_f):+.7f}",   units),
            ("Median",   f"{np.nanmedian(valid_crop_f):+.7f}", units),
            ("Std",      f"{np.nanstd(valid_crop_f):.7f}",     units),
            ("Min",      f"{np.nanmin(valid_crop_f):+.7f}",    units),
            ("Max",      f"{np.nanmax(valid_crop_f):+.7f}",    units),
            ("N pixels", f"{len(valid_crop_f):,}",             "px"),
            ("% wetter", f"{100*np.mean(valid_crop_f > 0):.2f}", "%"),
            ("% drier",  f"{100*np.mean(valid_crop_f < 0):.2f}", "%"),
        ]
        st.dataframe(pd.DataFrame(region_rows, columns=["Statistic", "Value", "Unit"]),
                     use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export Current View</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📄 CSV — pixel values**")
        st.caption("Flat table: longitude, latitude, precipitation slope for the current region.")
        lon_grid, lat_grid = np.meshgrid(lons_crop, lats_crop)
        export_df = pd.DataFrame({
            "longitude"        : lon_grid.ravel(),
            "latitude"         : lat_grid.ravel(),
            f"pr_slope_{units.replace('/','_').replace(' ','_')}": data_display.ravel(),
        }).dropna()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data      = csv_bytes,
            file_name = f"pr_slope_ssp585_{region_name.lower().replace(' ','_')}.csv",
            mime      = "text/csv",
        )
        st.metric("Rows in export", f"{len(export_df):,}")

    with col_b:
        st.markdown("**🖼️ PNG — map image**")
        st.caption("High-resolution static map of the current view.")
        dpi = st.select_slider("DPI", [72, 150, 300], value=150)

        if st.button("Generate PNG"):
            fig_png, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
            ax.set_facecolor(BG)
            cmap_obj = plt.get_cmap(cmap_name)
            cmap_obj.set_bad(color=BG)
            im = ax.imshow(
                data_display,
                extent = [lons_crop[0], lons_crop[-1], lats_crop[-1], lats_crop[0]],
                cmap   = cmap_obj, vmin=vmin, vmax=vmax,
                aspect = "auto", origin="upper",
            )
            cb = fig_png.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label(f"Precipitation Slope ({units})", color=MUTED, fontsize=10)
            cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

            ax.set_title(
                f"Precipitation Slope SSP5-8.5 — {region_name}",
                color=TEXT, fontsize=13, pad=12, fontfamily="monospace"
            )
            ax.set_xlabel("Longitude", color=MUTED, fontsize=9)
            ax.set_ylabel("Latitude",  color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

            buf = io.BytesIO()
            fig_png.savefig(buf, format="png", dpi=dpi,
                            bbox_inches="tight", facecolor=BG)
            buf.seek(0)
            plt.close(fig_png)

            st.download_button(
                "⬇️ Download PNG", data=buf,
                file_name=f"pr_slope_ssp585_{region_name.lower().replace(' ','_')}.png",
                mime="image/png",
            )
            st.success("PNG ready — click above to save.")

st.markdown("---")
st.caption("Scenario: SSP5-8.5 · Variable: Precipitation slope · Source: CMIP6 via GEE")
