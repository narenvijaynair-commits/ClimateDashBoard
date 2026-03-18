import ee
import geemap.foliumap as geemap
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Initialize Earth Engine
# -----------------------------------------------------------------------------
ee.Initialize(project='gee-dashboard-12345')

# -----------------------------------------------------------------------------
# Streamlit UI setup
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Earth Data Dashboard 🌍")
st.title("🌍 Earth Data Dashboard")

st.markdown("""
This dashboard visualizes long-term hydrologic trends from NASA Earth datasets:
- **GRACE:** Total Water Storage (TWS) anomalies  
- **IMERG:** Precipitation (monthly accumulation)  
Use the dataset selector and date range below to explore regional trends.
""")

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["GRACE - Total Water Storage (TWS)", "IMERG - Precipitation"]
)

col1, col2 = st.columns(2)
start_date = col1.date_input("Start Date", value=pd.to_datetime("2002-04-01"))
end_date = col2.date_input("End Date", value=pd.to_datetime("2023-12-31"))
start_date, end_date = str(start_date), str(end_date)

# -----------------------------------------------------------------------------
# Load and process datasets
# -----------------------------------------------------------------------------
if "GRACE" in dataset_option:
    dataset_id = "NASA/GRACE/MASS_GRIDS/LAND"
    var = "lwe_thickness"
    label = "GRACE TWS Trend (mm/yr)"
    desc = "Total water storage anomalies estimated from GRACE & GRACE-FO"
    scale_factor = 1.0
    palette = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf',
               '#d9ef8b','#a6d96a','#66bd63','#1a9850']
elif "IMERG" in dataset_option:
    dataset_id = "NASA/GPM_L3/IMERG_MONTHLY_V06"
    var = "precipitation"
    label = "IMERG Precipitation Trend (mm/yr)"
    desc = "Monthly mean precipitation estimates from GPM IMERG"
    scale_factor = 1.0
    palette = ['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']

col = ee.ImageCollection(dataset_id).select(var).filterDate(start_date, end_date)

st.write(f"Number of images in selected range: {col.size().getInfo()}")

# -----------------------------------------------------------------------------
# Trend computation
# -----------------------------------------------------------------------------
def addTime(image):
    """Add a time band (years since 2000)."""
    t = ee.Number(image.date().difference(ee.Date('2000-01-01'), 'year'))
    return image.addBands(ee.Image.constant(t).rename('t')).float()

col_time = col.map(addTime)
trend = col_time.select(['t', var]).reduce(ee.Reducer.linearFit())

# -----------------------------------------------------------------------------
# Visualization parameters
# -----------------------------------------------------------------------------
trend_vis = {
    'bands': ['scale'],
    'min': -5,
    'max': 5,
    'palette': palette
}

# -----------------------------------------------------------------------------
# Map visualization
# -----------------------------------------------------------------------------
m = geemap.Map(center=[20, 0], zoom=2)
m.addLayer(trend, trend_vis, label)
m.add_colorbar(
    vis_params=trend_vis,
    label=label,
    orientation='horizontal',
    transparent_bg=True
)

# -----------------------------------------------------------------------------
# Streamlit output
# -----------------------------------------------------------------------------
st.write(f"### {label}")
st.caption(desc)
m.to_streamlit(height=600)

st.markdown(f"""
**Interpretation:**  
- **Blue/green regions** → Increasing trend  
- **Red/orange regions** → Decreasing trend  

Dataset source: {dataset_id}
""")
