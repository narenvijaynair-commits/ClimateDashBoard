import ee
import geemap.foliumap as geemap
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------------------
# Initialize Earth Engine
# -----------------------------------------------------------------------------
try:
    ee.Initialize(project='calm-vehicle-450421-v2')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='calm-vehicle-450421-v2')

# -----------------------------------------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🌍 Earth Data Dashboard", layout="wide")
st.title("🌍 Earth Data Dashboard")

st.sidebar.header("Settings")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset:",
    ["IMERG Precipitation", "GRACE TWS (Mascon)"]
)

start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 1, 1))

# -----------------------------------------------------------------------------
# Create Map
# -----------------------------------------------------------------------------
m = geemap.Map(center=[0, 0], zoom=2)

# -----------------------------------------------------------------------------
# IMERG Dataset (Default)
# -----------------------------------------------------------------------------
if dataset_choice == "IMERG Precipitation":
    st.subheader("🌧️ IMERG Monthly Precipitation")

    imerg = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")
        .select("precipitation")
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    if imerg.size().getInfo() == 0:
        st.warning("No IMERG data found for this period.")
    else:
        imerg_mean = imerg.mean()
        vis = {"min": 0, "max": 300, "palette": ["#f7fbff", "#6baed6", "#08306b"]}
        m.addLayer(imerg_mean, vis, "IMERG Precipitation (mm/month)")

        # Colorbar
        try:
            m.add_colorbar(vis, label="IMERG Precipitation (mm/month)")
        except Exception:
            m.add_colorbar_branca(
                colors=vis["palette"], vmin=vis["min"], vmax=vis["max"],
                caption="IMERG Precipitation (mm/month)"
            )

# -----------------------------------------------------------------------------
# GRACE Dataset (Mascon)
# -----------------------------------------------------------------------------
elif dataset_choice == "GRACE TWS (Mascon)":
    st.subheader("💧 GRACE Total Water Storage (Mascon)")

    grace = (
        ee.ImageCollection("NASA/GRACE/MASS_GRIDS/MASCON")
        .select("lwe_thickness")
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    if grace.size().getInfo() == 0:
        st.warning("No GRACE data found for this period.")
    else:
        grace_mean = grace.mean().clip(ee.Geometry.BBox(-180, -90, 180, 90))
        vis = {"min": -20, "max": 20, "palette": ["red", "white", "blue"]}
        m.addLayer(grace_mean, vis, "GRACE LWE Thickness (cm)")

        # Colorbar
        try:
            m.add_colorbar(vis, label="GRACE LWE Thickness (cm)")
        except Exception:
            m.add_colorbar_branca(
                colors=vis["palette"], vmin=vis["min"], vmax=vis["max"],
                caption="GRACE LWE Thickness (cm)"
            )

# -----------------------------------------------------------------------------
# Display Map
# -----------------------------------------------------------------------------
m.to_streamlit(height=650)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
---
**Data Sources:**
- IMERG Monthly Precipitation: [NASA/GPM_L3/IMERG_MONTHLY_V06](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_MONTHLY_V06)  
- GRACE Total Water Storage (Mascon): [NASA/GRACE/MASS_GRIDS/MASCON](https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_MASCON)
""")
