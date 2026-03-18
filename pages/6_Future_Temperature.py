"""
Temperature Slope — SSP2-4.5 vs SSP5-8.5 Interactive Dashboard
=================================================================
Install:
    pip install streamlit plotly rasterio numpy pandas matplotlib scipy

Run:
    streamlit run temp_ssp_dashboard.py


    - tas_slope_ssp245.tif
    - tas_slope_ssp585.tif
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
    page_title="Temperature Slope — SSP2-4.5 vs SSP5-8.5",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b0f19; color: #d4dce8; }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2d45;
    }

    h1 { font-family: 'JetBrains Mono', monospace !important;
         font-size: 20px !important; color: #d4dce8 !important; }
    h2, h3 { font-family: 'Inter', sans-serif !important; color: #d4dce8 !important; }

    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px; color: #5a7a9e;
        text-transform: uppercase; letter-spacing: 2px;
        border-bottom: 1px solid #1f2d45;
        padding-bottom: 6px; margin: 20px 0 14px 0;
    }

    .pill-ssp245 { background:#1a2e3b; color:#38bdf8; border:1px solid #0284c7;
                   border-radius:12px; padding:2px 10px; font-size:11px;
                   font-family:'JetBrains Mono',monospace; }
    .pill-ssp585 { background:#2d1e1e; color:#f87171; border:1px solid #dc2626;
                   border-radius:12px; padding:2px 10px; font-size:11px;
                   font-family:'JetBrains Mono',monospace; }

    .stTabs [data-baseweb="tab-list"] { background:#111827; border-bottom:1px solid #1f2d45; gap:0; }
    .stTabs [data-baseweb="tab"] {
        font-family:'JetBrains Mono',monospace; font-size:11px; color:#5a7a9e;
        border-bottom:2px solid transparent; padding:10px 18px;
    }
    .stTabs [aria-selected="true"] { color:#a5f3fc !important; border-bottom-color:#a5f3fc !important; }

    div[data-testid="stMetric"] {
        background:#111827; border:1px solid #1f2d45;
        border-radius:8px; padding:12px 14px;
    }
    div[data-testid="stMetric"] label
        { color:#5a7a9e !important; font-size:11px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"]
        { color:#d4dce8 !important; font-size:18px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DATASETS = {
    "ssp245": {
        "file"      : "tas_slope_ssp245.tif",
        "label"     : "SSP2-4.5",
        "long_label": "Temperature Slope (tas) — SSP2-4.5",
        "color"     : "#38bdf8",
        "units"     : "°C/yr",
    },
    "ssp585": {
        "file"      : "tas_slope_ssp585.tif",
        "label"     : "SSP5-8.5",
        "long_label": "Temperature Slope (tas) — SSP5-8.5",
        "color"     : "#f87171",
        "units"     : "°C/yr",
    },
}

COLORMAPS = {
    "RdYlBu (diverging)": "RdYlBu_r",
    "RdBu (diverging)"  : "RdBu_r",
    "PRGn (diverging)" : "PRGn",
    "PiYG (diverging)" : "PiYG",
    "Coolwarm"         : "coolwarm",
    "Blues"            : "Blues",
    "Viridis"          : "viridis",
    "Turbo"            : "turbo",
}

REGIONS = {
    "Global"              : (-180, -90,  180,  90),
    "South Asia"          : (60,    5,   100,  40),
    "East Asia"           : (100,  15,   145,  55),
    "Southeast Asia"      : (95,  -10,   141,  28),
    "Middle East"         : (25,   15,    65,  42),
    "Africa"              : (-20, -40,    55,  38),
    "Europe"              : (-25,  35,    45,  72),
    "North America"       : (-170, 15,   -50,  75),
    "South America"       : (-82, -60,   -30,  15),
    "Australia & Oceania" : (110, -50,   180,  10),
    "Arctic (>60°N)"      : (-180, 60,   180,  90),
    "Tropics (±23.5°)"    : (-180,-23.5, 180,  23.5),
}

BG, BG2, BORDER, TEXT, MUTED = "#0b0f19", "#111827", "#1f2d45", "#d4dce8", "#5a7a9e"
C245, C585 = "#38bdf8", "#f87171"

# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_tiff(path: str, downsample: int = 4):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(max(1, src.height // downsample),
                       max(1, src.width  // downsample)),
            resampling=rasterio.enums.Resampling.average,
        ).astype(float)
        bounds     = src.bounds
        nodata     = src.nodata
        crs        = str(src.crs)
        orig_shape = (src.height, src.width)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.abs(data) > 1e10, np.nan, data)
    lons = np.linspace(bounds.left,  bounds.right,  data.shape[1])
    lats = np.linspace(bounds.top,   bounds.bottom, data.shape[0])
    return {"data": data, "lons": lons, "lats": lats,
            "bounds": bounds, "crs": crs, "orig_shape": orig_shape}

def crop(raster, bounds):
    lo, la, hi, ha = bounds
    lm = (raster["lons"] >= lo) & (raster["lons"] <= hi)
    rm = (raster["lats"] >= la) & (raster["lats"] <= ha)
    return (raster["data"][np.ix_(rm, lm)],
            raster["lons"][lm],
            raster["lats"][rm])

def apply_smooth(data, sigma):
    if not HAS_SCIPY:
        return data
    s = ndimage.gaussian_filter(np.where(np.isnan(data), 0, data), sigma=sigma)
    return np.where(np.isnan(data), np.nan, s)

def clamp(data, use_pct, pct_lo, pct_hi, man_min, man_max, sym):
    v = data[~np.isnan(data)].ravel()
    if len(v) == 0:
        return -1.0, 1.0
    vmin = float(np.percentile(v, pct_lo))  if use_pct else man_min
    vmax = float(np.percentile(v, pct_hi))  if use_pct else man_max
    if sym:
        e = max(abs(vmin), abs(vmax))
        vmin, vmax = -e, e
    return vmin, vmax

# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════
def layout_base(height=480):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2, height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(font=dict(color=MUTED, size=11), bgcolor="rgba(0,0,0,0)"),
    )

def ax_style(title="", size=10, showgrid=True):
    return dict(title=title, color=MUTED, tickfont=dict(size=size),
                gridcolor="#1a2638", zeroline=False, showgrid=showgrid)

def heatmap_fig(data, lons, lats, cmap, vmin, vmax, title, units, height=480):
    fig = go.Figure(go.Heatmap(
        z=data, x=lons, y=lats, zmin=vmin, zmax=vmax,
        colorscale=cmap,
        colorbar=dict(
            title=dict(text=units, side="right", font=dict(size=10, color=MUTED)),
            tickfont=dict(size=9, color=MUTED), thickness=12, len=0.72,
            bgcolor="rgba(17,24,39,0.9)", bordercolor=BORDER, borderwidth=1,
        ),
        hovertemplate=f"Lon:%{{x:.2f}}° Lat:%{{y:.2f}}°<br>Slope: %{{z:.6f}} {units}<extra></extra>",
    ))
    fig.update_layout(
        **layout_base(height),
        title=dict(text=title, font=dict(family="JetBrains Mono", size=12, color=TEXT), x=0.01),
        xaxis=dict(**ax_style("Longitude", showgrid=False)),
        yaxis=dict(**ax_style("Latitude",  showgrid=False),
                   scaleanchor="x", scaleratio=1),
    )
    return fig

def diff_fig(d245, d585, lons, lats, units, height=480):
    r = min(d245.shape[0], d585.shape[0])
    c = min(d245.shape[1], d585.shape[1])
    diff  = d585[:r, :c] - d245[:r, :c]
    lc, lrc = lons[:c], lats[:r]
    v     = diff[~np.isnan(diff)].ravel()
    if len(v) == 0:
        return go.Figure()
    e = max(abs(np.percentile(v, 2)), abs(np.percentile(v, 98)))
    fig = go.Figure(go.Heatmap(
        z=diff, x=lc, y=lrc, zmin=-e, zmax=e,
        colorscale="RdBu_r",
        colorbar=dict(
            title=dict(text=units, side="right", font=dict(size=10, color=MUTED)),
            tickfont=dict(size=9, color=MUTED), thickness=12, len=0.72,
            bgcolor="rgba(17,24,39,0.9)", bordercolor=BORDER, borderwidth=1,
        ),
        hovertemplate=f"Lon:%{{x:.2f}}° Lat:%{{y:.2f}}°<br>SSP5-8.5 − SSP2-4.5: %{{z:.6f}} {units}<extra></extra>",
    ))
    fig.update_layout(
        **layout_base(height),
        title=dict(text="Difference: SSP5-8.5 − SSP2-4.5  (red = warmer, blue = cooler)",
                   font=dict(family="JetBrains Mono", size=12, color=TEXT), x=0.01),
        xaxis=dict(**ax_style("Longitude", showgrid=False)),
        yaxis=dict(**ax_style("Latitude",  showgrid=False),
                   scaleanchor="x", scaleratio=1),
    )
    return fig, diff

def hist_fig(v245, v585, vmin, vmax, cmap_name, units, bins=70):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = go.Figure()
    for v, label, color in [(v245, "SSP2-4.5", C245), (v585, "SSP5-8.5", C585)]:
        clipped = v[(v >= vmin) & (v <= vmax)]
        counts, edges = np.histogram(clipped, bins=bins, range=(vmin, vmax))
        centres = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=centres, y=counts, name=label,
            marker_color=color, opacity=0.6, marker_line_width=0,
            hovertemplate=f"{label}: %{{x:.6f}} {units}<br>Count: %{{y:,}}<extra></extra>",
        ))
    fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash", line_width=1.5,
                  annotation_text="0", annotation_font_color="#94a3b8", annotation_font_size=9)
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(**ax_style(f"Temperature Slope ({units})")),
        yaxis=dict(**ax_style("Pixel Count")),
        **layout_base(320),
    )
    return fig

def zonal_fig(d245, d585, lats):
    z245 = np.nanmean(d245, axis=1)
    z585 = np.nanmean(d585, axis=1)
    n    = min(len(z245), len(z585), len(lats))

    fig = go.Figure()
    for z, label, color in [(z245, "SSP2-4.5", C245), (z585, "SSP5-8.5", C585)]:
        valid = ~np.isnan(z[:n])
        fig.add_trace(go.Scatter(
            x=z[:n][valid], y=lats[:n][valid], mode="lines",
            name=label, line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{x:.6f}}<br>Lat: %{{y:.1f}}°<extra></extra>",
        ))

    # Shaded difference
    z245c, z585c = z245[:n], z585[:n]
    lats_n = lats[:n]
    valid  = ~(np.isnan(z245c) | np.isnan(z585c))
    fig.add_trace(go.Scatter(
        x=np.concatenate([z245c[valid], z585c[valid][::-1]]),
        y=np.concatenate([lats_n[valid], lats_n[valid][::-1]]),
        fill="toself", fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_vline(x=0, line_color=MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        **layout_base(440),
        title=dict(text="Zonal Mean by Latitude", font=dict(family="JetBrains Mono", size=12, color=MUTED)),
        xaxis=dict(**ax_style("Mean Slope (°C/yr)")),
        yaxis=dict(**ax_style("Latitude")),
    )
    return fig

def longitudinal_fig(d245, d585, lons):
    l245 = np.nanmean(d245, axis=0)
    l585 = np.nanmean(d585, axis=0)
    n    = min(len(l245), len(l585), len(lons))
    fig  = go.Figure()
    for l, label, color in [(l245, "SSP2-4.5", C245), (l585, "SSP5-8.5", C585)]:
        valid = ~np.isnan(l[:n])
        fig.add_trace(go.Scatter(
            x=lons[:n][valid], y=l[:n][valid], mode="lines",
            name=label, line=dict(color=color, width=2),
            hovertemplate=f"{label}: %{{y:.6f}}<br>Lon: %{{x:.1f}}°<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        **layout_base(300),
        title=dict(text="Longitudinal Mean", font=dict(family="JetBrains Mono", size=12, color=MUTED)),
        xaxis=dict(**ax_style("Longitude")),
        yaxis=dict(**ax_style("Mean Slope (°C/yr)")),
    )
    return fig

def zonal_diff_fig(d245, d585, lats):
    z245 = np.nanmean(d245, axis=1)
    z585 = np.nanmean(d585, axis=1)
    n    = min(len(z245), len(z585), len(lats))
    diff = z585[:n] - z245[:n]
    valid = ~np.isnan(diff)
    colors = [C585 if d > 0 else C245 for d in diff[valid]]
    fig = go.Figure(go.Bar(
        x=diff[valid], y=lats[:n][valid], orientation="h",
        marker_color=colors, marker_line_width=0,
        hovertemplate="Lat: %{y:.1f}°<br>SSP585−SSP245: %{x:.6f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_dash="dash", line_width=1)
    fig.update_layout(
        **layout_base(440),
        title=dict(text="Zonal Difference: SSP5-8.5 − SSP2-4.5",
                   font=dict(family="JetBrains Mono", size=12, color=MUTED)),
        xaxis=dict(**ax_style("Slope difference (°C/yr)")),
        yaxis=dict(**ax_style("Latitude")),
    )
    return fig

def scatter_fig(v245, v585, sample=6000):
    mask = ~(np.isnan(v245) | np.isnan(v585))
    a, b = v245[mask], v585[mask]
    if len(a) > sample:
        idx  = np.random.choice(len(a), sample, replace=False)
        a, b = a[idx], b[idx]
    r = np.corrcoef(a, b)[0, 1] if len(a) > 1 else 0
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a, y=b, mode="markers",
        marker=dict(size=3, color="#a78bfa", opacity=0.35),
        hovertemplate="SSP245: %{x:.6f}<br>SSP585: %{y:.6f}<extra></extra>",
        name="Grid cells",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=1.2),
        name="1:1 line", showlegend=True,
    ))
    # OLS line
    m, b_int = np.polyfit(a, b, 1)
    x_fit    = np.linspace(lo, hi, 200)
    fig.add_trace(go.Scatter(
        x=x_fit, y=m * x_fit + b_int, mode="lines",
        line=dict(color="#fbbf24", width=1.5),
        name=f"OLS fit (r={r:.3f})",
    ))
    fig.update_layout(
        **layout_base(420),
        title=dict(text=f"Pixel-wise scatter: SSP2-4.5 vs SSP5-8.5  (r = {r:.4f})",
                   font=dict(family="JetBrains Mono", size=12, color=TEXT), x=0.01),
        xaxis=dict(**ax_style("SSP2-4.5 slope (°C/yr)")),
        yaxis=dict(**ax_style("SSP5-8.5 slope (°C/yr)")),
    )
    return fig, r

def warmcool_comparison_fig(v245, v585):
    """Grouped bar: % warmer / cooler / unchanged for each scenario."""
    rows = []
    for v, label in [(v245, "SSP2-4.5"), (v585, "SSP5-8.5")]:
        n = len(v)
        rows.append({
            "Scenario" : label,
            "Warmer ↑" : 100 * np.mean(v > 0),
            "Cooler ↓"  : 100 * np.mean(v < 0),
            "No change": 100 * np.mean(v == 0),
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for col, color in [("Warmer ↑", "#4ade80"), ("Cooler ↓", "#f87171"), ("No change", MUTED)]:
        fig.add_trace(go.Bar(
            x=df["Scenario"], y=df[col], name=col,
            marker_color=color, marker_line_width=0,
            hovertemplate=f"{col}: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(**ax_style("Scenario")),
        yaxis=dict(**ax_style("% of pixels")),
        **layout_base(320),
        title=dict(text="Warmer / Cooler / Unchanged Pixel Split by Scenario",
                   font=dict(family="JetBrains Mono", size=12, color=MUTED)),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Guards
# ═══════════════════════════════════════════════════════════════════════════════
if not HAS_RASTERIO:
    st.error("**rasterio** is required. Run: `pip install rasterio`")
    st.stop()

missing = [k for k, v in DATASETS.items() if not Path(v["file"]).exists()]
for k in missing:
    st.error(f"**{DATASETS[k]['file']}** not found in the current directory.")
if len(missing) == len(DATASETS):
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌡️ Temperature Slope — SSP2-4.5 vs SSP5-8.5")
st.markdown(
    "<span class='pill-ssp245'>SSP2-4.5</span>"
    " &nbsp; vs &nbsp; "
    "<span class='pill-ssp585'>SSP5-8.5</span>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    downsample = st.select_slider(
        "Resolution (lower = faster)", options=[1, 2, 4, 8, 16], value=4)

    region_name   = st.selectbox("Region / Zoom", list(REGIONS.keys()), index=0)
    region_bounds = REGIONS[region_name]

    st.markdown("---")
    cmap_label = st.selectbox("Colour palette", list(COLORMAPS.keys()), index=0)
    cmap_name  = COLORMAPS[cmap_label]
    if st.checkbox("Reverse palette", value=False):
        cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name[:-2]

    st.markdown("---")
    st.markdown("**Colour range**")
    use_pct  = st.checkbox("Clip by percentile", value=True)
    pct_lo   = st.slider("Lower %",   0, 49,   2) if use_pct else 2
    pct_hi   = st.slider("Upper %",  51, 100, 98) if use_pct else 98
    man_min  = st.number_input("Min", value=-0.001, step=0.0001, format="%.5f") if not use_pct else -0.001
    man_max  = st.number_input("Max", value= 0.001, step=0.0001, format="%.5f") if not use_pct else  0.001
    sym      = st.checkbox("Symmetric around 0", value=True)

    st.markdown("---")
    smooth = st.checkbox("Gaussian smoothing", value=False)
    sigma  = st.slider("Sigma", 1, 10, 3) if smooth else 1

    st.markdown("---")
    st.markdown(
        "<span style='font-family:JetBrains Mono;font-size:9px;color:#5a7a9e;'>"
        "Source: CMIP6 via GEE<br>Scenario: SSP2-4.5 · SSP5-8.5<br>Variable: Temperature slope"
        "</span>", unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading rasters…"):
    available = {k: v for k, v in DATASETS.items() if Path(v["file"]).exists()}
    rasters = {k: load_tiff(v["file"], downsample) for k, v in available.items()}

    cropped = {}
    for k, r in rasters.items():
        dc, lc, lac = crop(r, region_bounds)
        if smooth:
            dc = apply_smooth(dc, sigma)
        cropped[k] = (dc, lc, lac)

d245, lons245, lats245 = cropped.get("ssp245", (None, None, None))
d585, lons585, lats585 = cropped.get("ssp585", (None, None, None))

# Use ssp245 grid as reference (or whichever is available)
ref_key        = "ssp245" if "ssp245" in cropped else "ssp585"
_, lons_ref, lats_ref = cropped[ref_key]

v245 = d245[~np.isnan(d245)].ravel() if d245 is not None else np.array([])
v585 = d585[~np.isnan(d585)].ravel() if d585 is not None else np.array([])

# Shared colour range (computed from whichever datasets exist)
all_vals = np.concatenate([v for v in [v245, v585] if len(v) > 0])
vmin, vmax = clamp(all_vals.reshape(-1,1), use_pct, pct_lo, pct_hi, man_min, man_max, sym)

units = "°C/yr"

# ═══════════════════════════════════════════════════════════════════════════════
# Metrics strip
# ═══════════════════════════════════════════════════════════════════════════════
m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

if len(v245):
    m1.metric("SSP2-4.5 Mean",   f"{np.nanmean(v245):+.5f}",   units)
    m2.metric("SSP2-4.5 Warmer", f"{100*np.mean(v245>0):.1f}%",
              f"{100*np.mean(v245<0):.1f}% cooler")
    m3.metric("SSP2-4.5 Max",    f"{np.nanmax(v245):+.5f}",    units)
    m4.metric("SSP2-4.5 Min",    f"{np.nanmin(v245):+.5f}",    units)

if len(v585):
    m5.metric("SSP5-8.5 Mean",   f"{np.nanmean(v585):+.5f}",   units)
    m6.metric("SSP5-8.5 Warmer", f"{100*np.mean(v585>0):.1f}%",
              f"{100*np.mean(v585<0):.1f}% cooler")
    m7.metric("SSP5-8.5 Max",    f"{np.nanmax(v585):+.5f}",    units)
    m8.metric("SSP5-8.5 Min",    f"{np.nanmin(v585):+.5f}",    units)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_maps, tab_diff, tab_hist, tab_zonal, tab_scatter, tab_wetdry, tab_stats, tab_export = st.tabs([
    "🗺️  Side-by-Side",
    "➕  Difference",
    "📊  Histogram",
    "🌐  Zonal",
    "🔵  Scatter",
    "🌡️  Warm / Cool",
    "📋  Statistics",
    "💾  Export",
])

# ── Maps ──────────────────────────────────────────────────────────────────────
with tab_maps:
    if d245 is not None:
        st.plotly_chart(
            heatmap_fig(d245, lons245, lats245, cmap_name, vmin, vmax,
                        f"SSP2-4.5 — {region_name}", units),
            use_container_width=True,
        )
    if d585 is not None:
        st.plotly_chart(
            heatmap_fig(d585, lons585, lats585, cmap_name, vmin, vmax,
                        f"SSP5-8.5 — {region_name}", units),
            use_container_width=True,
        )

    # Both maps share the same colour scale — add a note
    st.caption(
        f"Both maps use the same colour scale [{vmin:.5f}, {vmax:.5f}] {units} "
        f"for direct visual comparison. "
        f"CRS: {rasters[ref_key]['crs']} · "
        f"Display downsampled ×{downsample}."
    )

    if region_name != "Global":
        st.markdown('<div class="section-header">Region statistics</div>',
                    unsafe_allow_html=True)
        rc = st.columns(8)
        for vals, label, cols in [(v245, "SSP2-4.5", rc[:4]), (v585, "SSP5-8.5", rc[4:])]:
            if len(vals):
                cols[0].metric(f"{label} mean",   f"{np.nanmean(vals):+.5f}")
                cols[1].metric(f"{label} median", f"{np.nanmedian(vals):+.5f}")
                cols[2].metric(f"{label} max",    f"{np.nanmax(vals):+.5f}")
                cols[3].metric(f"{label} min",    f"{np.nanmin(vals):+.5f}")

# ── Difference map ────────────────────────────────────────────────────────────
with tab_diff:
    if d245 is not None and d585 is not None:
        fig_diff, diff_arr = diff_fig(d245, d585, lons_ref, lats_ref, units)
        st.plotly_chart(fig_diff, use_container_width=True)

        diff_flat = diff_arr[~np.isnan(diff_arr)].ravel()
        st.markdown('<div class="section-header">Difference statistics</div>',
                    unsafe_allow_html=True)
        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
        dc1.metric("Mean diff",       f"{np.nanmean(diff_flat):+.6f} {units}")
        dc2.metric("Median diff",     f"{np.nanmedian(diff_flat):+.6f} {units}")
        dc3.metric("Std diff",        f"{np.nanstd(diff_flat):.6f} {units}")
        dc4.metric("SSP585 > SSP245", f"{100*np.mean(diff_flat > 0):.1f}%", "of pixels")
        dc5.metric("SSP585 < SSP245", f"{100*np.mean(diff_flat < 0):.1f}%", "of pixels")
    else:
        st.info("Both SSP2-4.5 and SSP5-8.5 files are needed for the difference map.")

# ── Histogram ─────────────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="section-header">Overlaid Value Distributions</div>',
                unsafe_allow_html=True)
    bins = st.slider("Bins", 20, 200, 70, key="hist_bins")
    st.plotly_chart(hist_fig(v245, v585, vmin, vmax, cmap_name, units, bins=bins),
                    use_container_width=True)
    st.caption("Both distributions clipped to the shared colour range for comparability.")

    st.markdown('<div class="section-header">Percentile Tables</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for col, vals, label in [(pc1, v245, "SSP2-4.5"), (pc2, v585, "SSP5-8.5")]:
        if len(vals):
            pv = np.nanpercentile(vals, pcts)
            with col:
                st.markdown(f"**{label}** ({units})")
                st.dataframe(pd.DataFrame({
                    "Pct"  : [f"P{p}" for p in pcts],
                    "Value": [f"{x:+.7f}" for x in pv],
                    "Dir"  : ["↑ Warmer" if x > 0 else "↓ Cooler" for x in pv],
                }), use_container_width=True, hide_index=True)

# ── Zonal analysis ────────────────────────────────────────────────────────────
with tab_zonal:
    if d245 is not None and d585 is not None:
        st.markdown('<div class="section-header">Zonal Mean by Latitude</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(zonal_fig(d245, d585, lats_ref), use_container_width=True)

        st.markdown('<div class="section-header">Longitudinal Mean</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(longitudinal_fig(d245, d585, lons_ref), use_container_width=True)

        st.markdown('<div class="section-header">Zonal Difference: SSP5-8.5 − SSP2-4.5</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(zonal_diff_fig(d245, d585, lats_ref), use_container_width=True)
        st.caption("Red = SSP5-8.5 warmer than SSP2-4.5 · Blue = SSP5-8.5 cooler than SSP2-4.5")
    else:
        st.info("Both files needed for zonal comparison.")

# ── Pixel scatter ─────────────────────────────────────────────────────────────
with tab_scatter:
    if d245 is not None and d585 is not None:
        n = min(len(d245.ravel()), len(d585.ravel()))
        fig_sc, r = scatter_fig(d245.ravel()[:n], d585.ravel()[:n])
        st.plotly_chart(fig_sc, use_container_width=True)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Pixel correlation r", f"{r:.4f}")
        sc2.metric("OLS slope (SSP585/SSP245)", f"{np.polyfit(d245.ravel()[:n][~(np.isnan(d245.ravel()[:n])|np.isnan(d585.ravel()[:n]))], d585.ravel()[:n][~(np.isnan(d245.ravel()[:n])|np.isnan(d585.ravel()[:n]))], 1)[0]:.4f}")
        sc3.metric("Sampled pixels", "6,000")
        st.caption("Yellow line = OLS fit · Dashed grey = 1:1 reference.")
    else:
        st.info("Both files needed for scatter plot.")

# ── Wet / Dry breakdown ───────────────────────────────────────────────────────
with tab_wetdry:
    st.markdown('<div class="section-header">Warmer / Cooler / Unchanged by Scenario</div>',
                unsafe_allow_html=True)
    if len(v245) and len(v585):
        st.plotly_chart(warmcool_comparison_fig(v245, v585), use_container_width=True)

        st.markdown('<div class="section-header">Pixel Counts</div>', unsafe_allow_html=True)
        wdc1, wdc2 = st.columns(2)
        for col, vals, label in [(wdc1, v245, "SSP2-4.5"), (wdc2, v585, "SSP5-8.5")]:
            n_wet  = int(np.sum(vals > 0))
            n_dry  = int(np.sum(vals < 0))
            n_zero = int(np.sum(vals == 0))
            total  = len(vals)
            with col:
                st.markdown(f"**{label}**")
                df_wd = pd.DataFrame([
                    {"Category": "Warmer ↑",  "Pixels": n_wet,  "Pct": f"{100*n_wet/total:.2f}%",
                     "Mean slope": f"{np.mean(vals[vals>0]):+.7f}" if n_wet else "N/A"},
                    {"Category": "Cooler ↓",   "Pixels": n_dry,  "Pct": f"{100*n_dry/total:.2f}%",
                     "Mean slope": f"{np.mean(vals[vals<0]):+.7f}" if n_dry else "N/A"},
                    {"Category": "No change", "Pixels": n_zero, "Pct": f"{100*n_zero/total:.2f}%",
                     "Mean slope": "0.0000000"},
                ])
                st.dataframe(df_wd, use_container_width=True, hide_index=True)

# ── Statistics ────────────────────────────────────────────────────────────────
with tab_stats:
    for k, vals in [("ssp245", v245), ("ssp585", v585)]:
        if k not in rasters:
            continue
        m  = DATASETS[k]
        r  = rasters[k]
        vc = cropped[k][0][~np.isnan(cropped[k][0])].ravel()
        st.markdown(f'<div class="section-header">{m["long_label"]}</div>',
                    unsafe_allow_html=True)
        rows = [
            ("Mean",          f"{np.nanmean(vals):+.8f}",    units),
            ("Median",        f"{np.nanmedian(vals):+.8f}",  units),
            ("Std deviation", f"{np.nanstd(vals):.8f}",      units),
            ("Min",           f"{np.nanmin(vals):+.8f}",     units),
            ("Max",           f"{np.nanmax(vals):+.8f}",     units),
            ("% Warmer ↑",    f"{100*np.mean(vals>0):.3f}",  "%"),
            ("% Cooler ↓",     f"{100*np.mean(vals<0):.3f}",  "%"),
            ("Valid pixels",  f"{len(vals):,}",               "px"),
            ("Native size",   f"{r['orig_shape'][1]}×{r['orig_shape'][0]}", "px"),
            ("CRS",           r["crs"],                       ""),
            ("Bounds W/E",    f"{r['bounds'].left:.4f} / {r['bounds'].right:.4f}", "°"),
            ("Bounds S/N",    f"{r['bounds'].bottom:.4f} / {r['bounds'].top:.4f}", "°"),
        ]
        if region_name != "Global":
            rows += [
                (f"[{region_name}] Mean",   f"{np.nanmean(vc):+.8f}",  units),
                (f"[{region_name}] Median", f"{np.nanmedian(vc):+.8f}", units),
                (f"[{region_name}] N px",   f"{len(vc):,}",             "px"),
                (f"[{region_name}] Warmer", f"{100*np.mean(vc>0):.2f}", "%"),
                (f"[{region_name}] Cooler",  f"{100*np.mean(vc<0):.2f}", "%"),
            ]
        st.dataframe(pd.DataFrame(rows, columns=["Statistic", "Value", "Unit"]),
                     use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

    exp_scenario = st.radio("Dataset", ["SSP2-4.5", "SSP5-8.5", "Difference (SSP5-8.5 − SSP2-4.5)"],
                             horizontal=True)

    if exp_scenario == "SSP2-4.5":
        exp_data, exp_lons, exp_lats, exp_key = d245, lons245, lats245, "ssp245"
    elif exp_scenario == "SSP5-8.5":
        exp_data, exp_lons, exp_lats, exp_key = d585, lons585, lats585, "ssp585"
    else:
        if d245 is not None and d585 is not None:
            r = min(d245.shape[0], d585.shape[0])
            c = min(d245.shape[1], d585.shape[1])
            exp_data = d585[:r, :c] - d245[:r, :c]
            exp_lons, exp_lats, exp_key = lons_ref[:c], lats_ref[:r], "diff"
        else:
            st.info("Both files required for difference export.")
            st.stop()

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**📄 CSV**")
        lon_g, lat_g = np.meshgrid(exp_lons, exp_lats)
        export_df = pd.DataFrame({
            "longitude"        : lon_g.ravel(),
            "latitude"         : lat_g.ravel(),
            f"tas_slope_{units.replace('/','_')}": exp_data.ravel(),
        }).dropna()
        st.download_button(
            "⬇️ Download CSV",
            data      = export_df.to_csv(index=False).encode("utf-8"),
            file_name = f"tas_{exp_key}_{region_name.lower().replace(' ','_')}.csv",
            mime      = "text/csv",
        )
        st.metric("Rows", f"{len(export_df):,}")

    with ec2:
        st.markdown("**🖼️ PNG**")
        dpi = st.select_slider("DPI", [72, 150, 300], value=150)
        if st.button("Render PNG"):
            fig_p, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
            ax.set_facecolor(BG)
            ev = exp_data[~np.isnan(exp_data)].ravel()
            ev_min = float(np.percentile(ev, 2)) if len(ev) else vmin
            ev_max = float(np.percentile(ev, 98)) if len(ev) else vmax
            if sym:
                e = max(abs(ev_min), abs(ev_max))
                ev_min, ev_max = -e, e
            cmap_obj = plt.get_cmap("RdBu_r" if exp_key == "diff" else cmap_name)
            cmap_obj.set_bad(color=BG)
            im = ax.imshow(exp_data,
                           extent=[exp_lons[0], exp_lons[-1], exp_lats[-1], exp_lats[0]],
                           cmap=cmap_obj, vmin=ev_min, vmax=ev_max,
                           aspect="auto", origin="upper")
            cb = fig_p.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label(f"Temperature Slope ({units})", color=MUTED, fontsize=10)
            cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
            ax.set_title(f"{exp_scenario} — {region_name}", color=TEXT,
                         fontsize=13, pad=12, fontfamily="monospace")
            ax.set_xlabel("Longitude", color=MUTED, fontsize=9)
            ax.set_ylabel("Latitude",  color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            buf = io.BytesIO()
            fig_p.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
            buf.seek(0); plt.close(fig_p)
            st.download_button(
                "⬇️ Download PNG", data=buf,
                file_name=f"tas_{exp_key}_{region_name.lower().replace(' ','_')}.png",
                mime="image/png",
            )

st.markdown("---")
st.caption("Temperature slope · CMIP6 · SSP2-4.5 & SSP5-8.5 · Source: GEE export")
