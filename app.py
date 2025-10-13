import streamlit as st
import ee
import geemap.foliumap as geemap

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

st.set_page_config(page_title="Global Precipitation Trend Dashboard", layout="wide")

st.title("🌧️ Global Precipitation Trend Dashboard")
st.markdown("Visualizing IMERG monthly precipitation trends (2001–2023)")

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Settings")
    dataset_choice = st.selectbox("Dataset:", ["IMERG Monthly", "CHIRPS Pentad"])
    start_date = st.date_input("Start Date", value=pd.to_datetime("2001-06-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    min_trend = st.slider("Min trend (mm/yr)", -0.01, 0.0, -0.005, 0.001)
    max_trend = st.slider("Max trend (mm/yr)", 0.0, 0.01, 0.005, 0.001)

# Load dataset
if dataset_choice == "IMERG Monthly":
    dataset = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V06').select('precipitation')
else:
    dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD').select('precipitation')

dataset = dataset.filterDate(str(start_date), str(end_date))

# Define region
region = ee.Geometry.Rectangle([-180, -60, 180, 85])

# Land mask
land_mask = ee.Image('MODIS/051/MCD12Q1/2001_01_01').select('Land_Cover_Type_1').gt(0)

# Add time band
def add_time_band(image):
    date = ee.Date(image.get('system:time_start'))
    years = date.difference(ee.Date(str(start_date)), 'year')
    return image.addBands(ee.Image(years).rename('time').float())

dataset = dataset.map(add_time_band)

# Linear regression
linear_fit = dataset.select(['time', 'precipitation']).reduce(ee.Reducer.linearFit())
trend = linear_fit.select('scale').updateMask(land_mask)

# Visualization params
trend_vis = {
    'min': min_trend,
    'max': max_trend,
    'palette': ['#d73027', '#f46d43', '#fee090', '#e0f3f8', '#74add1', '#4575b4']
}

# Create map
m = geemap.Map(center=[10, 0], zoom=2)
m.addLayer(trend, trend_vis, "Precipitation Trend (mm/yr)")
m.add_colorbar(trend_vis['palette'], vmin=min_trend, vmax=max_trend, label="Trend (mm/yr)")

# Display map
m.to_streamlit(height=600)

# Optional: Add chatbot section (simple example)
st.divider()
st.header("💬 Ask the data assistant")

user_input = st.text_input("Ask a question about precipitation trends:")
if user_input:
    # Placeholder chatbot (could be connected to GPT or a custom model)
    st.write(f"🤖: Regions with strong negative trends may indicate drying over time. "
             f"Try zooming into the affected areas for more insight!")
