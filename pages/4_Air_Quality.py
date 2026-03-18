"""
PM2.5 Monthly Trend — Interactive GeoTIFF Dashboard
====================================================
Install:
    pip install streamlit plotly rasterio numpy pandas matplotlib scipy folium streamlit-folium

Run:
    streamlit run pm25_tif_dashboard.py

Required input file (same directory):
    - pm25_monthly_trend_slope.tif
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
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PM2.5 Trend Explorer",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
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
    .metric-pos { color: #f85149; }
    .metric-neg { color: #3fb950; }

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
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

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
TIFF_FILE = "pm25_monthly_trend_slope.tif"

COLORMAPS = {
    "RdBu (Diverging)"    : "RdBu_r",
    "RdYlGn (Diverging)"  : "RdYlGn",
    "Coolwarm"            : "coolwarm",
    "Seismic"             : "seismic",
    "PiYG"                : "PiYG",
    "Viridis"             : "viridis",
    "Plasma"              : "plasma",
    "Inferno"             : "inferno",
    "Turbo"               : "turbo",
}

REGIONS = {
    "Global"         : (-180, -90, 180, 90),
    "South Asia"     : (60,    5,  100, 40),
    "East Asia"      : (100,  15,  145, 55),
    "Southeast Asia" : (95,  -10,  141, 28),
    "Middle East"    : (25,   15,   65, 42),
    "Africa"         : (-20,  -40,   55, 38),
    "Europe"         : (-25,   35,   45, 72),
    "North America"  : (-170,  15,  -50, 75),
    "South America"  : (-82,  -60,  -30, 15),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading GeoTIFF…")
def load_tiff(path: str, downsample: int = 4):
    """Load raster, return data array + geographic metadata."""
    if not HAS_RASTERIO:
        st.error("rasterio not installed. Run: pip install rasterio")
        return None

    with rasterio.open(path) as src:
        # Read at reduced resolution for performance
        data = src.read(
            1,
            out_shape=(
                max(1, src.height // downsample),
                max(1, src.width  // downsample),
            ),
            resampling=rasterio.enums.Resampling.average,
        ).astype(float)

        bounds  = src.bounds          # left, bottom, right, top
        nodata  = src.nodata
        crs     = str(src.crs)
        orig_shape = (src.height, src.width)
        res     = src.res             # (x_res, y_res)

    # Mask nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.abs(data) > 1e10, np.nan, data)   # catch sentinel values

    # Build lat/lon arrays
    lons = np.linspace(bounds.left,  bounds.right, data.shape[1])
    lats = np.linspace(bounds.top,   bounds.bottom, data.shape[0])

    return {
        "data"       : data,
        "lons"       : lons,
        "lats"       : lats,
        "bounds"     : bounds,
        "crs"        : crs,
        "orig_shape" : orig_shape,
        "res"        : res,
        "nodata"     : nodata,
    }

def crop_to_region(raster, region_bounds):
    """Crop raster dict to a lon/lat bounding box."""
    lon_min, lat_min, lon_max, lat_max = region_bounds
    lons, lats, data = raster["lons"], raster["lats"], raster["data"]

    col_mask = (lons >= lon_min) & (lons <= lon_max)
    row_mask = (lats >= lat_min) & (lats <= lat_max)

    cropped = data[np.ix_(row_mask, col_mask)]
    return cropped, lons[col_mask], lats[row_mask]

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════
def make_heatmap(data, lons, lats, cmap_name, vmin, vmax, title=""):
    """Build a Plotly heatmap figure."""
    fig = go.Figure(go.Heatmap(
        z        = data,
        x        = lons,
        y        = lats,
        zmin     = vmin,
        zmax     = vmax,
        colorscale = cmap_name,
        colorbar = dict(
            title      = dict(text="µg/m³/yr", side="right", font=dict(size=11, color="#8b949e")),
            tickfont   = dict(size=10, color="#8b949e"),
            thickness  = 14,
            len        = 0.75,
            bgcolor    = "rgba(22,27,34,0.8)",
            bordercolor= "#30363d",
            borderwidth= 1,
        ),
        hovertemplate = "Lon: %{x:.2f}°<br>Lat: %{y:.2f}°<br>Trend: %{z:.4f} µg/m³/yr<extra></extra>",
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
    """Build a colour-coded histogram of trend values."""
    valid = data_flat[~np.isnan(data_flat)]
    valid = valid[(valid >= vmin) & (valid <= vmax)]

    counts, edges = np.histogram(valid, bins=bins, range=(vmin, vmax))
    centres = (edges[:-1] + edges[1:]) / 2

    # Colour each bar by its position on the colourmap
    cmap   = plt.get_cmap(cmap_name)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = [f"rgba{tuple(int(c*255) for c in cmap(norm(v))[:3]) + (0.85,)}" for v in centres]

    fig = go.Figure(go.Bar(
        x          = centres,
        y          = counts,
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "Trend: %{x:.4f} µg/m³/yr<br>Count: %{y:,}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color="#f85149", line_dash="dash", line_width=1.5,
                  annotation_text="0", annotation_font_color="#f85149",
                  annotation_font_size=10)

    fig.update_layout(
        xaxis = dict(title="PM2.5 Trend (µg/m³/yr)", color="#8b949e",
                     tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis = dict(title="Pixel Count", color="#8b949e",
                     tickfont=dict(size=10), gridcolor="#21262d"),
        bargap         = 0.02,
        paper_bgcolor  = "#0f1117",
        plot_bgcolor   = "#161b22",
        height         = 320,
        margin         = dict(l=50, r=20, t=20, b=40),
        showlegend     = False,
    )
    return fig

def make_zonal_bar(data, lats, title="Zonal Mean Trend"):
    """Mean trend by latitude band."""
    zonal = np.nanmean(data, axis=1)          # average across longitudes
    valid = ~np.isnan(zonal)

    colors = ["#f85149" if v > 0 else "#3fb950" for v in zonal[valid]]

    fig = go.Figure(go.Bar(
        x          = zonal[valid],
        y          = lats[valid],
        orientation= "h",
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "Lat: %{y:.1f}°<br>Mean trend: %{x:.4f} µg/m³/yr<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#8b949e", line_dash="dash", line_width=1)
    fig.update_layout(
        title      = dict(text=title, font=dict(family="IBM Plex Mono", size=12, color="#8b949e")),
        xaxis      = dict(title="Mean PM2.5 Trend (µg/m³/yr)", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis      = dict(title="Latitude", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#161b22",
        height        = 420,
        margin        = dict(l=60, r=20, t=40, b=40),
    )
    return fig

def make_longitudinal_bar(data, lons, title="Longitudinal Mean Trend"):
    """Mean trend by longitude band."""
    longitudinal = np.nanmean(data, axis=0)
    valid        = ~np.isnan(longitudinal)

    colors = ["#f85149" if v > 0 else "#3fb950" for v in longitudinal[valid]]

    fig = go.Figure(go.Bar(
        x          = lons[valid],
        y          = longitudinal[valid],
        marker_color = colors,
        marker_line_width = 0,
        hovertemplate = "Lon: %{x:.1f}°<br>Mean trend: %{y:.4f} µg/m³/yr<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dash", line_width=1)
    fig.update_layout(
        title      = dict(text=title, font=dict(family="IBM Plex Mono", size=12, color="#8b949e")),
        xaxis      = dict(title="Longitude", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        yaxis      = dict(title="Mean PM2.5 Trend (µg/m³/yr)", color="#8b949e",
                          tickfont=dict(size=10), gridcolor="#21262d"),
        paper_bgcolor = "#0f1117",
        plot_bgcolor  = "#161b22",
        height        = 320,
        margin        = dict(l=60, r=20, t=40, b=40),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌫️ PM2.5 Trend Explorer")
st.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:12px;color:#8b949e;'>"
    "Interactive analysis of monthly PM2.5 long-term trend slope &nbsp;·&nbsp; µg/m³/year"
    "</span>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Guard: check dependencies and file
# ═══════════════════════════════════════════════════════════════════════════════
if not HAS_RASTERIO:
    st.error("**rasterio** is required. Install with: `pip install rasterio`")
    st.stop()

if not Path(TIFF_FILE).exists():
    st.error(
        f"**{TIFF_FILE}** not found in the current directory.\n\n"
        "Export it from GEE first using the PM2.5 trend analysis script."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar controls
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    # Resolution
    downsample = st.select_slider(
        "Resolution (lower = faster)",
        options=[1, 2, 4, 8, 16],
        value=4,
        help="Downsampling factor. 1 = native resolution (may be slow)."
    )

    st.markdown("---")

    # Region
    region_name = st.selectbox("Region / Zoom", list(REGIONS.keys()), index=0)
    region_bounds = REGIONS[region_name]

    st.markdown("---")

    # Colourmap
    cmap_label = st.selectbox("Colour palette", list(COLORMAPS.keys()), index=0)
    cmap_name  = COLORMAPS[cmap_label]

    # Reverse palette
    reverse_cmap = st.checkbox("Reverse palette", value=False)
    if reverse_cmap:
        cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name[:-2]

    st.markdown("---")

    # Value clipping
    st.markdown("**Colour range clipping**")
    use_percentile = st.checkbox("Clip by percentile", value=True)
    if use_percentile:
        pct_low  = st.slider("Lower percentile", 0, 49,  2)
        pct_high = st.slider("Upper percentile", 51, 100, 98)
    else:
        manual_min = st.number_input("Min value (µg/m³/yr)", value=-0.10, step=0.01, format="%.3f")
        manual_max = st.number_input("Max value (µg/m³/yr)", value= 0.10, step=0.01, format="%.3f")

    st.markdown("---")

    # Smoothing
    smooth = st.checkbox("Apply Gaussian smoothing", value=False)
    if smooth:
        sigma = st.slider("Smoothing sigma (pixels)", 1, 10, 3)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:IBM Plex Mono;font-size:10px;color:#8b949e;'>"
        "Source: Global Satellite PM2.5<br>GEE Collection · Monthly trend slope"
        "</span>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Load and process data
# ═══════════════════════════════════════════════════════════════════════════════
raster = load_tiff(TIFF_FILE, downsample=downsample)
data_crop, lons_crop, lats_crop = crop_to_region(raster, region_bounds)

# Optional smoothing
if smooth and HAS_SCIPY:
    data_smooth = ndimage.gaussian_filter(
        np.where(np.isnan(data_crop), 0, data_crop), sigma=sigma
    )
    mask = np.isnan(data_crop)
    data_display = np.where(mask, np.nan, data_smooth)
elif smooth and not HAS_SCIPY:
    st.sidebar.warning("scipy not installed — smoothing unavailable.")
    data_display = data_crop
else:
    data_display = data_crop

# Compute colour range
valid_flat = data_display[~np.isnan(data_display)].ravel()
if use_percentile and len(valid_flat) > 0:
    vmin = float(np.percentile(valid_flat, pct_low))
    vmax = float(np.percentile(valid_flat, pct_high))
else:
    vmin = manual_min
    vmax = manual_max

# Ensure symmetric around 0 for diverging maps (optional toggle)
make_symmetric = st.sidebar.checkbox("Force symmetric colour range", value=True)
if make_symmetric:
    extreme = max(abs(vmin), abs(vmax))
    vmin, vmax = -extreme, extreme

# ═══════════════════════════════════════════════════════════════════════════════
# Summary metrics
# ═══════════════════════════════════════════════════════════════════════════════
valid_all   = raster["data"][~np.isnan(raster["data"])].ravel()
valid_crop  = data_crop[~np.isnan(data_crop)].ravel()
pct_worsening = 100 * np.mean(valid_all > 0)
pct_improving = 100 * np.mean(valid_all < 0)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mean trend",    f"{np.nanmean(valid_all):+.4f}",  "µg/m³/yr")
col2.metric("Median trend",  f"{np.nanmedian(valid_all):+.4f}", "µg/m³/yr")
col3.metric("Std dev",       f"{np.nanstd(valid_all):.4f}",    "µg/m³/yr")
col4.metric("Max (worst)",   f"{np.nanmax(valid_all):+.4f}",   "µg/m³/yr")
col5.metric("Min (best)",    f"{np.nanmin(valid_all):+.4f}",   "µg/m³/yr")
col6.metric("Worsening pixels", f"{pct_worsening:.1f}%",       f"vs {pct_improving:.1f}% improving")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Main tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_map, tab_hist, tab_zonal, tab_stats, tab_export = st.tabs([
    "🗺️  Map", "📊  Histogram", "🌐  Zonal Analysis", "📋  Statistics", "💾  Export"
])

# ── Tab 1: Map ────────────────────────────────────────────────────────────────
with tab_map:
    map_title = f"PM2.5 Monthly Trend Slope — {region_name}"
    fig_map   = make_heatmap(data_display, lons_crop, lats_crop,
                             cmap_name, vmin, vmax, title=map_title)
    st.plotly_chart(fig_map, use_container_width=True)

    # Region stats below map
    if region_name != "Global":
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(f"{region_name} — Mean",   f"{np.nanmean(valid_crop):+.4f} µg/m³/yr")
        rc2.metric(f"{region_name} — Median", f"{np.nanmedian(valid_crop):+.4f} µg/m³/yr")
        rc3.metric(f"{region_name} — Max",    f"{np.nanmax(valid_crop):+.4f} µg/m³/yr")
        rc4.metric(f"{region_name} — Min",    f"{np.nanmin(valid_crop):+.4f} µg/m³/yr")

    st.caption(
        f"CRS: {raster['crs']} · "
        f"Native resolution: {raster['orig_shape'][1]}×{raster['orig_shape'][0]} px · "
        f"Display downsampled ×{downsample} · "
        f"Colour range: [{vmin:.4f}, {vmax:.4f}] µg/m³/yr"
    )

# ── Tab 2: Histogram ─────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="section-header">Value Distribution</div>', unsafe_allow_html=True)

    bins = st.slider("Number of bins", 20, 200, 80, key="hist_bins")
    fig_hist = make_histogram(valid_crop, vmin, vmax, cmap_name, bins=bins)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Percentile table
    st.markdown('<div class="section-header">Percentile Summary</div>', unsafe_allow_html=True)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.nanpercentile(valid_crop, pcts)
    pct_df = pd.DataFrame({
        "Percentile"       : [f"P{p}" for p in pcts],
        "Value (µg/m³/yr)" : [f"{v:+.5f}" for v in pct_vals],
        "Direction"        : ["↑ Worsening" if v > 0 else "↓ Improving" for v in pct_vals],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

# ── Tab 3: Zonal analysis ─────────────────────────────────────────────────────
with tab_zonal:
    st.markdown('<div class="section-header">Zonal Mean (by Latitude)</div>', unsafe_allow_html=True)
    fig_zonal = make_zonal_bar(data_display, lats_crop, title="Mean PM2.5 Trend by Latitude Band")
    st.plotly_chart(fig_zonal, use_container_width=True)

    st.markdown('<div class="section-header">Longitudinal Mean (by Longitude)</div>', unsafe_allow_html=True)
    fig_lon = make_longitudinal_bar(data_display, lons_crop, title="Mean PM2.5 Trend by Longitude Band")
    st.plotly_chart(fig_lon, use_container_width=True)

# ── Tab 4: Statistics ─────────────────────────────────────────────────────────
with tab_stats:
    st.markdown('<div class="section-header">Global Statistics</div>', unsafe_allow_html=True)

    stat_rows = [
        ("Mean",               f"{np.nanmean(valid_all):+.6f}",     "µg/m³/yr"),
        ("Median",             f"{np.nanmedian(valid_all):+.6f}",    "µg/m³/yr"),
        ("Std deviation",      f"{np.nanstd(valid_all):.6f}",        "µg/m³/yr"),
        ("Min",                f"{np.nanmin(valid_all):+.6f}",       "µg/m³/yr"),
        ("Max",                f"{np.nanmax(valid_all):+.6f}",       "µg/m³/yr"),
        ("% pixels worsening", f"{pct_worsening:.2f}",               "%"),
        ("% pixels improving", f"{pct_improving:.2f}",               "%"),
        ("Valid pixels",       f"{len(valid_all):,}",                "px"),
        ("Native resolution",  f"{raster['orig_shape'][1]}×{raster['orig_shape'][0]}", "px"),
        ("CRS",                raster["crs"],                        ""),
        ("Bounds (W)",         f"{raster['bounds'].left:.4f}",       "°"),
        ("Bounds (E)",         f"{raster['bounds'].right:.4f}",      "°"),
        ("Bounds (S)",         f"{raster['bounds'].bottom:.4f}",     "°"),
        ("Bounds (N)",         f"{raster['bounds'].top:.4f}",        "°"),
    ]

    stats_df = pd.DataFrame(stat_rows, columns=["Statistic", "Value", "Unit"])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    if region_name != "Global":
        st.markdown(f'<div class="section-header">{region_name} Region Statistics</div>',
                    unsafe_allow_html=True)
        region_rows = [
            ("Mean",     f"{np.nanmean(valid_crop):+.6f}",   "µg/m³/yr"),
            ("Median",   f"{np.nanmedian(valid_crop):+.6f}",  "µg/m³/yr"),
            ("Std",      f"{np.nanstd(valid_crop):.6f}",      "µg/m³/yr"),
            ("Min",      f"{np.nanmin(valid_crop):+.6f}",     "µg/m³/yr"),
            ("Max",      f"{np.nanmax(valid_crop):+.6f}",     "µg/m³/yr"),
            ("N pixels", f"{len(valid_crop):,}",              "px"),
        ]
        st.dataframe(pd.DataFrame(region_rows, columns=["Statistic", "Value", "Unit"]),
                     use_container_width=True, hide_index=True)

# ── Tab 5: Export ─────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export Current View</div>', unsafe_allow_html=True)
    st.markdown("Download the **currently displayed region** as a CSV or PNG.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📄 CSV — pixel values**")
        st.caption("Exports a flat table of lon, lat, trend value for the current region.")
        # Build long-form dataframe
        lon_grid, lat_grid = np.meshgrid(lons_crop, lats_crop)
        export_df = pd.DataFrame({
            "longitude"             : lon_grid.ravel(),
            "latitude"              : lat_grid.ravel(),
            "pm25_trend_ug_m3_yr"   : data_display.ravel(),
        }).dropna()
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download CSV",
            data      = csv_bytes,
            file_name = f"pm25_trend_{region_name.lower().replace(' ','_')}.csv",
            mime      = "text/csv",
        )
        st.metric("Rows in export", f"{len(export_df):,}")

    with col_b:
        st.markdown("**🖼️ PNG — map image**")
        st.caption("Exports the current map as a static high-resolution PNG.")

        dpi = st.select_slider("DPI", [72, 150, 300], value=150)

        if st.button("Generate PNG"):
            fig_png, ax = plt.subplots(figsize=(14, 7), facecolor="#0f1117")
            ax.set_facecolor("#0f1117")
            cmap_obj = plt.get_cmap(cmap_name)
            cmap_obj.set_bad(color="#0f1117")
            im = ax.imshow(
                data_display,
                extent   = [lons_crop[0], lons_crop[-1], lats_crop[-1], lats_crop[0]],
                cmap     = cmap_obj,
                vmin     = vmin,
                vmax     = vmax,
                aspect   = "auto",
                origin   = "upper",
            )
            cb = fig_png.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label("PM2.5 Trend (µg/m³/yr)", color="#8b949e", fontsize=10)
            cb.ax.yaxis.set_tick_params(color="#8b949e", labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

            ax.set_title(f"PM2.5 Monthly Trend — {region_name}", color="#e8e8e8",
                         fontsize=13, pad=12, fontfamily="monospace")
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
                file_name = f"pm25_trend_{region_name.lower().replace(' ','_')}.png",
                mime      = "image/png",
            )
            st.success("PNG ready — click the button above to save.")
