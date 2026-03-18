"""
HDI vs Climate Trends — Multi-Dataset Interactive Streamlit Dashboard
======================================================================
Install:
    pip install streamlit plotly pandas numpy netCDF4 rasterio

Run:
    streamlit run hdi_climate_dashboard.py

Required input files (same directory):
    - HDR25_Statistical_Annex_HDI_Table.csv
    - PM25_Monthly_Trend_By_Country.csv
    - GPM_Precip_Trend_By_Country.csv
    - GRACE_TWS_Trend_By_Country_Global.csv
    - GISS_temperature_trend_1980_2025.nc
    - pr_slope_ssp245.tif
    - pr_slope_ssp585.tif
    - tas_slope_ssp245.tif
    - tas_slope_ssp585.tif
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="HDI vs Climate Trends",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# Shared constants
# ============================================================
CONTINENT_COLORS = {
    'Africa'  : '#e74c3c',
    'Asia'    : '#3498db',
    'Europe'  : '#2ecc71',
    'Americas': '#f39c12',
    'Oceania' : '#9b59b6',
    'Unknown' : '#95a5a6',
}

CONTINENT_MARKERS = {
    'Africa': 'circle', 'Asia': 'square', 'Europe': 'triangle-up',
    'Americas': 'triangle-down', 'Oceania': 'diamond', 'Unknown': 'pentagon',
}

CONTINENT_MAPPING = {
    # Africa
    'algeria': 'Africa', 'angola': 'Africa', 'benin': 'Africa', 'botswana': 'Africa',
    'burkina faso': 'Africa', 'burundi': 'Africa', 'cabo verde': 'Africa', 'cape verde': 'Africa',
    'cameroon': 'Africa', 'central african republic': 'Africa', 'central african rep.': 'Africa',
    'central african rep': 'Africa', 'chad': 'Africa', 'comoros': 'Africa', 'congo': 'Africa',
    'congo (democratic republic of the)': 'Africa', "côte d'ivoire": 'Africa', "cote d'ivoire": 'Africa',
    'djibouti': 'Africa', 'egypt': 'Africa', 'equatorial guinea': 'Africa',
    'eritrea': 'Africa', 'eswatini': 'Africa', 'eswatini (kingdom of)': 'Africa',
    'ethiopia': 'Africa', 'gabon': 'Africa', 'gambia': 'Africa', 'ghana': 'Africa',
    'guinea': 'Africa', 'guinea-bissau': 'Africa', 'kenya': 'Africa', 'lesotho': 'Africa',
    'liberia': 'Africa', 'libya': 'Africa', 'madagascar': 'Africa', 'malawi': 'Africa',
    'mali': 'Africa', 'mauritania': 'Africa', 'mauritius': 'Africa', 'morocco': 'Africa',
    'mozambique': 'Africa', 'namibia': 'Africa', 'niger': 'Africa', 'nigeria': 'Africa',
    'rwanda': 'Africa', 'sao tome and principe': 'Africa', 'senegal': 'Africa',
    'seychelles': 'Africa', 'sierra leone': 'Africa', 'somalia': 'Africa',
    'somaliland': 'Africa', 'south africa': 'Africa', 'south sudan': 'Africa',
    'sudan': 'Africa', 'tanzania': 'Africa', 'tanzania (united republic of)': 'Africa',
    'togo': 'Africa', 'tunisia': 'Africa', 'uganda': 'Africa', 'zambia': 'Africa',
    'zimbabwe': 'Africa',
    # Asia
    'afghanistan': 'Asia', 'armenia': 'Asia', 'azerbaijan': 'Asia', 'bahrain': 'Asia',
    'bangladesh': 'Asia', 'bhutan': 'Asia', 'brunei darussalam': 'Asia', 'cambodia': 'Asia',
    'china': 'Asia', 'cyprus': 'Asia', 'georgia': 'Asia', 'india': 'Asia',
    'indonesia': 'Asia', 'iran': 'Asia', 'iran (islamic republic of)': 'Asia',
    'iraq': 'Asia', 'israel': 'Asia', 'japan': 'Asia', 'jordan': 'Asia',
    'kazakhstan': 'Asia', 'korea (republic of)': 'Asia', 'korea, rep.': 'Asia',
    "korea (democratic people's rep. of)": 'Asia', 'korea, north': 'Asia', 'korea, south': 'Asia',
    'kuwait': 'Asia', 'kyrgyzstan': 'Asia', "lao people's democratic republic": 'Asia', 'laos': 'Asia',
    'lebanon': 'Asia', 'malaysia': 'Asia', 'maldives': 'Asia', 'mongolia': 'Asia',
    'myanmar': 'Asia', 'burma': 'Asia', 'nepal': 'Asia', 'oman': 'Asia', 'pakistan': 'Asia',
    'state of palestine': 'Asia', 'palestine': 'Asia', 'west bank and gaza': 'Asia',
    'philippines': 'Asia', 'qatar': 'Asia', 'saudi arabia': 'Asia', 'singapore': 'Asia',
    'sri lanka': 'Asia', 'syria': 'Asia', 'syrian arab republic': 'Asia',
    'tajikistan': 'Asia', 'thailand': 'Asia', 'timor-leste': 'Asia', 'east timor': 'Asia',
    'turkey': 'Asia', 'türkiye': 'Asia', 'turkmenistan': 'Asia',
    'united arab emirates': 'Asia', 'uzbekistan': 'Asia', 'viet nam': 'Asia',
    'vietnam': 'Asia', 'yemen': 'Asia',
    # Europe
    'albania': 'Europe', 'andorra': 'Europe', 'austria': 'Europe', 'belarus': 'Europe',
    'belgium': 'Europe', 'bosnia and herzegovina': 'Europe', 'bulgaria': 'Europe',
    'croatia': 'Europe', 'czechia': 'Europe', 'czech republic': 'Europe',
    'denmark': 'Europe', 'estonia': 'Europe', 'finland': 'Europe', 'france': 'Europe',
    'germany': 'Europe', 'greece': 'Europe', 'hungary': 'Europe', 'iceland': 'Europe',
    'ireland': 'Europe', 'italy': 'Europe', 'kosovo': 'Europe', 'latvia': 'Europe',
    'liechtenstein': 'Europe', 'lithuania': 'Europe', 'luxembourg': 'Europe',
    'malta': 'Europe', 'moldova': 'Europe', 'moldova (republic of)': 'Europe',
    'moldova, rep. of': 'Europe', 'monaco': 'Europe', 'montenegro': 'Europe',
    'netherlands': 'Europe', 'north macedonia': 'Europe', 'norway': 'Europe',
    'poland': 'Europe', 'portugal': 'Europe', 'romania': 'Europe',
    'russia': 'Europe', 'russian federation': 'Europe', 'san marino': 'Europe',
    'serbia': 'Europe', 'slovakia': 'Europe', 'slovak republic': 'Europe',
    'slovenia': 'Europe', 'spain': 'Europe', 'sweden': 'Europe',
    'switzerland': 'Europe', 'ukraine': 'Europe', 'united kingdom': 'Europe',
    # Americas
    'antigua and barbuda': 'Americas', 'argentina': 'Americas', 'bahamas': 'Americas',
    'barbados': 'Americas', 'belize': 'Americas', 'bolivia': 'Americas',
    'bolivia (plurinational state of)': 'Americas', 'brazil': 'Americas', 'canada': 'Americas',
    'chile': 'Americas', 'colombia': 'Americas', 'costa rica': 'Americas', 'cuba': 'Americas',
    'dominica': 'Americas', 'dominican republic': 'Americas', 'ecuador': 'Americas',
    'el salvador': 'Americas', 'grenada': 'Americas', 'guatemala': 'Americas',
    'guyana': 'Americas', 'haiti': 'Americas', 'honduras': 'Americas', 'jamaica': 'Americas',
    'mexico': 'Americas', 'nicaragua': 'Americas', 'panama': 'Americas', 'paraguay': 'Americas',
    'peru': 'Americas', 'saint kitts and nevis': 'Americas', 'st. kitts and nevis': 'Americas',
    'saint lucia': 'Americas', 'st. lucia': 'Americas',
    'saint vincent and the grenadines': 'Americas', 'suriname': 'Americas',
    'trinidad and tobago': 'Americas', 'united states': 'Americas',
    'united states of america': 'Americas', 'uruguay': 'Americas',
    'venezuela': 'Americas', 'venezuela (bolivarian republic of)': 'Americas',
    # Oceania
    'australia': 'Oceania', 'fiji': 'Oceania', 'kiribati': 'Oceania',
    'marshall islands': 'Oceania', 'micronesia': 'Oceania',
    'micronesia (federated states of)': 'Oceania', 'nauru': 'Oceania',
    'new zealand': 'Oceania', 'palau': 'Oceania', 'papua new guinea': 'Oceania',
    'samoa': 'Oceania', 'solomon islands': 'Oceania', 'tonga': 'Oceania',
    'tuvalu': 'Oceania', 'vanuatu': 'Oceania',
}

# Dataset metadata — original four + two new SSP datasets
DATASETS = {
    'GISS Temperature': {
        'file'       : 'GISS_temperature_trend_1980_2025.nc',
        'y_label'    : 'Temperature Trend (°C/decade)',
        'title'      : 'HDI vs. GISS Surface Temperature Trend (1980–2025)',
        'units'      : '°C/decade',
        'description': 'NASA GISS surface temperature trend derived from gridded NetCDF data.',
    },
    'GPM Precipitation': {
        'file'       : 'GPM_Precip_Trend_By_Country.csv',
        'trend_col'  : 'precip_trend_mm_per_year',
        'country_col': 'country_name',
        'y_label'    : 'Precipitation Trend (mm/year)',
        'title'      : 'HDI vs. GPM Precipitation Trend',
        'units'      : 'mm/year',
        'description': 'GPM IMERG monthly precipitation trend stratified by country.',
    },
    'GRACE Water Storage': {
        'file'       : 'GRACE_TWS_Trend_By_Country_Global.csv',
        'trend_col'  : 'trend_cm_per_year',
        'country_col': 'country_na',
        'y_label'    : 'Water Storage Trend (cm/year)',
        'title'      : 'HDI vs. Terrestrial Water Storage Trend (GRACE)',
        'units'      : 'cm/year',
        'description': 'GRACE terrestrial water storage trend stratified by country.',
    },
    'PM2.5 Air Quality': {
        'file'       : 'PM25_Monthly_Trend_By_Country.csv',
        'trend_col'  : 'pm25_trend_ug_m3_per_year',
        'country_col': 'country_name',
        'y_label'    : 'PM2.5 Trend (µg/m³/year)',
        'title'      : 'HDI vs. PM2.5 Monthly Trend',
        'units'      : 'µg/m³/year',
        'description': 'Global satellite PM2.5 monthly trend stratified by country.',
    },
    # ── New SSP datasets ──────────────────────────────────────────────────────
    'SSP Precipitation': {
        'files': {
            'SSP2-4.5': 'pr_slope_ssp245.tif',
            'SSP5-8.5': 'pr_slope_ssp585.tif',
        },
        'y_label'    : 'Precipitation Slope (mm/day/yr)',
        'title'      : 'HDI vs. CMIP6 Projected Precipitation Slope',
        'units'      : 'mm/day/yr',
        'description': (
            'CMIP6 projected precipitation trend slopes (SSP2-4.5 and SSP5-8.5) sampled at '
            'each country centroid from GeoTIFF rasters exported via Google Earth Engine.'
        ),
        'ssp': True,
    },
    'SSP Temperature': {
        'files': {
            'SSP2-4.5': 'tas_slope_ssp245.tif',
            'SSP5-8.5': 'tas_slope_ssp585.tif',
        },
        'y_label'    : 'Temperature Slope (°C/yr)',
        'title'      : 'HDI vs. CMIP6 Projected Temperature Slope (tas)',
        'units'      : '°C/yr',
        'description': (
            'CMIP6 projected near-surface air temperature trend slopes (SSP2-4.5 and SSP5-8.5) '
            'sampled at each country centroid from GeoTIFF rasters exported via Google Earth Engine.'
        ),
        'ssp': True,
    },
}

# Country name mappings per dataset → HDI names
COUNTRY_MAPPINGS = {
    'GISS Temperature': {},
    'GPM Precipitation': {
        'north korea'              : "korea (democratic people's rep. of)",
        'central african rep.'     : 'central african republic',
        'central african rep'      : 'central african republic',
        'south korea'              : 'korea (republic of)',
        'iran'                     : 'iran (islamic republic of)',
        'syria'                    : 'syrian arab republic',
        'vietnam'                  : 'viet nam',
        'laos'                     : "lao people's democratic republic",
        'congo-kinshasa'           : 'congo (democratic republic of the)',
        'congo (democratic republic of the)': 'congo (democratic republic of the)',
        'congo-brazzaville'        : 'congo',
        'ivory coast'              : "côte d'ivoire",
        "côte d'ivoire"            : "côte d'ivoire",
        "cote d'ivoire"            : "côte d'ivoire",
        'the gambia'               : 'gambia',
        'czech republic'           : 'czechia',
        'slovak republic'          : 'slovakia',
        'macedonia'                : 'north macedonia',
        'turkey'                   : 'türkiye',
        'russia'                   : 'russian federation',
        'kyrgyzstan'               : 'kyrgyzstan',
        'east timor'               : 'timor-leste',
        'burma'                    : 'myanmar',
        'cape verde'               : 'cabo verde',
        'united states'            : 'united states',
        'somaliland'               : 'somalia',
        'tanzania'                 : 'tanzania (united republic of)',
        'bolivia'                  : 'bolivia (plurinational state of)',
        'venezuela'                : 'venezuela (bolivarian republic of)',
        'moldova'                  : 'moldova (republic of)',
        'eswatini'                 : 'eswatini (kingdom of)',
        'palestine'                : 'state of palestine',
    },
    'GRACE Water Storage': {
        'dem rep of the congo': 'congo (democratic republic of the)',
        'rep of the congo'    : 'congo',
        'korea, south'        : 'korea (republic of)',
        'korea, north'        : "korea (democratic people's rep. of)",
        'burma'               : 'myanmar',
        'central african rep' : 'central african republic',
        "cote d'ivoire"       : "côte d'ivoire",
        'russia'              : 'russian federation',
        'iran'                : 'iran (islamic republic of)',
        'bolivia'             : 'bolivia (plurinational state of)',
        'venezuela'           : 'venezuela (bolivarian republic of)',
        'syria'               : 'syrian arab republic',
        'tanzania'            : 'tanzania (united republic of)',
        'moldova'             : 'moldova (republic of)',
        'laos'                : "lao people's democratic republic",
        'eswatini'            : 'eswatini (kingdom of)',
        'vietnam'             : 'viet nam',
    },
    'PM2.5 Air Quality': {
        'congo-kinshasa'       : 'congo (democratic republic of the)',
        'congo-brazzaville'    : 'congo',
        'south korea'          : 'korea (republic of)',
        'north korea'          : "korea (democratic people's rep. of)",
        'burma'                : 'myanmar',
        'central african rep.' : 'central african republic',
        "côte d'ivoire"        : "côte d'ivoire",
        'ivory coast'          : "côte d'ivoire",
        'russia'               : 'russian federation',
        'iran'                 : 'iran (islamic republic of)',
        'bolivia'              : 'bolivia (plurinational state of)',
        'venezuela'            : 'venezuela (bolivarian republic of)',
        'syria'                : 'syrian arab republic',
        'tanzania'             : 'tanzania (united republic of)',
        'moldova'              : 'moldova (republic of)',
        'laos'                 : "lao people's democratic republic",
        'eswatini'             : 'eswatini (kingdom of)',
        'vietnam'              : 'viet nam',
        'turkey'               : 'türkiye',
        'turkiye'              : 'türkiye',
        'czech republic'       : 'czechia',
        'east timor'           : 'timor-leste',
        'somaliland'           : 'somalia',
        'the gambia'           : 'gambia',
        'palestine'            : 'state of palestine',
        'cape verde'           : 'cabo verde',
    },
}

# Country centroids — used for GISS NetCDF extraction AND SSP raster sampling
COUNTRY_COORDS = {
    'afghanistan': (33.0, 65.0), 'albania': (41.0, 20.0), 'algeria': (28.0, 3.0),
    'angola': (-12.5, 18.5), 'argentina': (-34.0, -64.0), 'armenia': (40.0, 45.0),
    'australia': (-25.0, 135.0), 'austria': (47.5, 14.5), 'azerbaijan': (40.5, 47.5),
    'bahamas': (24.25, -76.0), 'bangladesh': (24.0, 90.0), 'belarus': (53.0, 28.0),
    'belgium': (50.5, 4.5), 'belize': (17.25, -88.75), 'benin': (9.5, 2.25),
    'bhutan': (27.5, 90.5), 'bolivia (plurinational state of)': (-17.0, -65.0),
    'bosnia and herzegovina': (44.0, 18.0), 'botswana': (-22.0, 24.0),
    'brazil': (-10.0, -55.0), 'brunei darussalam': (4.5, 114.67),
    'bulgaria': (43.0, 25.0), 'burkina faso': (13.0, -2.0), 'burundi': (-3.5, 30.0),
    'cabo verde': (16.0, -24.0), 'cambodia': (13.0, 105.0), 'cameroon': (6.0, 12.0),
    'canada': (60.0, -95.0), 'central african republic': (7.0, 21.0), 'chad': (15.0, 19.0),
    'chile': (-30.0, -71.0), 'china': (35.0, 105.0), 'colombia': (4.0, -72.0),
    'comoros': (-11.64, 43.33), 'congo': (-1.0, 15.0),
    'congo (democratic republic of the)': (-4.0, 23.0), 'costa rica': (10.0, -84.0),
    "côte d'ivoire": (8.0, -5.0), 'croatia': (45.0, 16.0), 'cuba': (21.5, -80.0),
    'cyprus': (35.0, 33.0), 'czechia': (49.75, 15.5), 'denmark': (56.0, 10.0),
    'djibouti': (11.5, 43.0), 'dominican republic': (19.0, -70.5),
    'ecuador': (-2.0, -77.5), 'egypt': (27.0, 30.0), 'el salvador': (13.83, -88.92),
    'equatorial guinea': (1.5, 10.0), 'eritrea': (15.0, 39.0), 'estonia': (59.0, 26.0),
    'eswatini (kingdom of)': (-26.5, 31.5), 'ethiopia': (8.0, 38.0),
    'fiji': (-17.71, 178.06), 'finland': (64.0, 26.0), 'france': (46.0, 2.0),
    'gabon': (-1.0, 11.75), 'gambia': (13.5, -15.5), 'georgia': (42.0, 43.5),
    'germany': (51.0, 9.0), 'ghana': (8.0, -2.0), 'greece': (39.0, 22.0),
    'guatemala': (15.5, -90.25), 'guinea': (11.0, -10.0), 'guinea-bissau': (12.0, -15.0),
    'guyana': (5.0, -59.0), 'haiti': (19.0, -72.42), 'honduras': (15.0, -86.5),
    'hungary': (47.0, 20.0), 'iceland': (65.0, -18.0), 'india': (20.0, 77.0),
    'indonesia': (-5.0, 120.0), 'iran (islamic republic of)': (32.0, 53.0),
    'iraq': (33.0, 44.0), 'ireland': (53.0, -8.0), 'israel': (31.5, 34.75),
    'italy': (42.83, 12.83), 'jamaica': (18.25, -77.5), 'japan': (36.0, 138.0),
    'jordan': (31.0, 36.0), 'kazakhstan': (48.0, 68.0), 'kenya': (1.0, 38.0),
    'korea (republic of)': (37.0, 127.5), "korea (democratic people's rep. of)": (40.0, 127.0),
    'kyrgyzstan': (41.0, 75.0), "lao people's democratic republic": (18.0, 105.0),
    'latvia': (57.0, 25.0), 'lebanon': (33.83, 35.83), 'lesotho': (-29.5, 28.5),
    'liberia': (6.5, -9.5), 'libya': (25.0, 17.0), 'lithuania': (56.0, 24.0),
    'luxembourg': (49.75, 6.17), 'madagascar': (-20.0, 47.0), 'malawi': (-13.5, 34.0),
    'malaysia': (2.5, 112.5), 'maldives': (3.2, 73.0), 'mali': (17.0, -4.0),
    'mauritania': (20.0, -12.0), 'mauritius': (-20.28, 57.58), 'mexico': (23.0, -102.0),
    'moldova (republic of)': (47.0, 29.0), 'mongolia': (46.0, 105.0),
    'montenegro': (42.5, 19.3), 'morocco': (32.0, -5.0), 'mozambique': (-18.25, 35.0),
    'myanmar': (22.0, 98.0), 'namibia': (-22.0, 17.0), 'nepal': (28.0, 84.0),
    'netherlands': (52.5, 5.75), 'new zealand': (-41.0, 174.0), 'nicaragua': (13.0, -85.0),
    'niger': (16.0, 8.0), 'nigeria': (10.0, 8.0), 'north macedonia': (41.83, 22.0),
    'norway': (62.0, 10.0), 'pakistan': (30.0, 70.0), 'panama': (9.0, -80.0),
    'papua new guinea': (-6.0, 147.0), 'paraguay': (-23.0, -58.0), 'peru': (-10.0, -76.0),
    'philippines': (13.0, 122.0), 'poland': (52.0, 20.0), 'portugal': (39.5, -8.0),
    'romania': (46.0, 25.0), 'russian federation': (60.0, 100.0), 'rwanda': (-2.0, 30.0),
    'saudi arabia': (25.0, 45.0), 'senegal': (14.0, -14.0), 'serbia': (44.0, 21.0),
    'sierra leone': (8.5, -11.5), 'singapore': (1.37, 103.8), 'slovakia': (48.67, 19.5),
    'slovenia': (46.0, 15.0), 'somalia': (10.0, 49.0), 'south africa': (-29.0, 24.0),
    'south sudan': (8.0, 30.0), 'spain': (40.0, -4.0), 'sri lanka': (7.0, 81.0),
    'state of palestine': (32.0, 35.25), 'sudan': (15.0, 30.0), 'sweden': (62.0, 15.0),
    'switzerland': (47.0, 8.0), 'syrian arab republic': (35.0, 38.0),
    'tajikistan': (39.0, 71.0), 'tanzania (united republic of)': (-6.0, 35.0),
    'thailand': (15.0, 100.0), 'timor-leste': (-8.83, 125.92), 'togo': (8.0, 1.17),
    'tunisia': (34.0, 9.0), 'türkiye': (39.0, 35.0), 'turkmenistan': (40.0, 60.0),
    'uganda': (1.0, 32.0), 'ukraine': (49.0, 32.0), 'united kingdom': (54.0, -2.0),
    'united states': (38.0, -97.0), 'uruguay': (-33.0, -56.0), 'uzbekistan': (41.0, 64.0),
    'venezuela (bolivarian republic of)': (8.0, -66.0), 'viet nam': (16.0, 106.0),
    'yemen': (15.5, 48.0), 'zambia': (-13.0, 27.5), 'zimbabwe': (-20.0, 30.0),
}

# ============================================================
# Data loading functions (cached per dataset)
# ============================================================
@st.cache_data
def load_hdi():
    hdi_df = pd.read_csv('HDR25_Statistical_Annex_HDI_Table.csv', skiprows=3)
    hdi_df.columns = hdi_df.columns.str.strip()
    country_col = 'Country' if 'Country' in hdi_df.columns else hdi_df.columns[1]
    value_col   = hdi_df.columns[2]
    hdi_df[country_col] = hdi_df[country_col].str.strip()
    hdi_df[value_col]   = pd.to_numeric(hdi_df[value_col], errors='coerce')
    hdi_df = hdi_df.dropna(subset=[country_col, value_col])
    return dict(zip(hdi_df[country_col].str.lower(), hdi_df[value_col]))

@st.cache_data
def load_csv_dataset(dataset_key):
    meta        = DATASETS[dataset_key]
    mapping     = COUNTRY_MAPPINGS[dataset_key]
    df          = pd.read_csv(meta['file'])
    country_col = meta['country_col']
    trend_col   = meta['trend_col']
    df[country_col] = df[country_col].str.strip()
    df = df.drop_duplicates(subset=country_col, keep='first')

    hdi_lookup = load_hdi()
    rows = []
    for _, row in df.iterrows():
        country_raw = row[country_col]
        trend       = row[trend_col]
        if pd.isna(country_raw) or pd.isna(trend):
            continue
        country_lower = country_raw.lower()
        mapped        = mapping.get(country_lower, country_lower)
        hdi_value     = hdi_lookup.get(mapped) or hdi_lookup.get(country_lower)
        if hdi_value is not None:
            continent = CONTINENT_MAPPING.get(mapped,
                        CONTINENT_MAPPING.get(country_lower, 'Unknown'))
            rows.append({'Country': country_raw, 'HDI': float(hdi_value),
                         'Trend': float(trend), 'Continent': continent})

    return _add_outlier_flag(pd.DataFrame(rows).dropna())

@st.cache_data
def load_giss_dataset():
    try:
        import netCDF4 as nc
    except ImportError:
        st.error("netCDF4 not installed. Run: pip install netCDF4")
        return pd.DataFrame()

    hdi_lookup = load_hdi()
    nc_file    = nc.Dataset('GISS_temperature_trend_1980_2025.nc', 'r')
    lat        = nc_file.variables['lat'][:]
    lon        = nc_file.variables['lon'][:]
    temp_trend = nc_file.variables['TEMPTREND'][:]
    missing    = nc_file.variables['TEMPTREND'].missing_value
    nc_file.close()

    def get_trend(lat_c, lon_c, window=3):
        li = np.abs(lat - lat_c).argmin()
        lj = np.abs(lon - lon_c).argmin()
        chunk = temp_trend[max(0, li-window):li+window+1,
                           max(0, lj-window):lj+window+1]
        valid = chunk[chunk != missing]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan

    rows = []
    for country_lower, hdi_value in hdi_lookup.items():
        coords = COUNTRY_COORDS.get(country_lower)
        if coords is None:
            continue
        trend = get_trend(*coords)
        if np.isnan(trend):
            continue
        continent = CONTINENT_MAPPING.get(country_lower, 'Unknown')
        rows.append({'Country': country_lower.title(), 'HDI': float(hdi_value),
                     'Trend': trend, 'Continent': continent})

    return _add_outlier_flag(pd.DataFrame(rows).dropna())


@st.cache_data
def load_ssp_dataset(dataset_key):
    """
    Sample CMIP6 GeoTIFF rasters (SSP2-4.5 and SSP5-8.5) at each country's
    centroid using COUNTRY_COORDS, merge with HDI, and return a combined
    DataFrame. 'Scenario' distinguishes the two SSPs; 'Trend' holds the slope.
    """
    if not HAS_RASTERIO:
        st.error("rasterio is required for SSP datasets. Run: pip install rasterio")
        return pd.DataFrame()

    meta       = DATASETS[dataset_key]
    hdi_lookup = load_hdi()

    def sample_raster(tif_path, lat_c, lon_c, window=2):
        """Average pixel values within a small window around the centroid."""
        try:
            with rasterio.open(tif_path) as src:
                row, col = src.index(lon_c, lat_c)
                h, w = src.height, src.width
                r0 = max(0, row - window); r1 = min(h, row + window + 1)
                c0 = max(0, col - window); c1 = min(w, col + window + 1)
                data = src.read(
                    1,
                    window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                ).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                data = np.where(np.abs(data) > 1e10, np.nan, data)
                valid = data[~np.isnan(data)]
                return float(np.mean(valid)) if len(valid) > 0 else np.nan
        except Exception:
            return np.nan

    rows = []
    for country_lower, hdi_value in hdi_lookup.items():
        coords = COUNTRY_COORDS.get(country_lower)
        if coords is None:
            continue
        lat_c, lon_c = coords
        continent = CONTINENT_MAPPING.get(country_lower, 'Unknown')

        for scenario_label, tif_file in meta['files'].items():
            trend = sample_raster(tif_file, lat_c, lon_c)
            if np.isnan(trend):
                continue
            rows.append({
                'Country'  : country_lower.title(),
                'HDI'      : float(hdi_value),
                'Trend'    : trend,
                'Continent': continent,
                'Scenario' : scenario_label,
            })

    df = pd.DataFrame(rows).dropna(subset=['HDI', 'Trend'])
    return _add_outlier_flag_ssp(df)


def _add_outlier_flag(df):
    if df.empty:
        return df
    Q1, Q3 = df['Trend'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['Outlier']   = ~df['Trend'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    df['Direction'] = np.where(df['Trend'] > 0, 'Increasing ↑', 'Decreasing ↓')
    return df


def _add_outlier_flag_ssp(df):
    """Outlier flag computed per scenario so each SSP uses its own distribution."""
    if df.empty:
        return df
    result = []
    for _, grp in df.groupby('Scenario'):
        Q1, Q3 = grp['Trend'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        grp = grp.copy()
        grp['Outlier']   = ~grp['Trend'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        grp['Direction'] = np.where(grp['Trend'] > 0, 'Increasing ↑', 'Decreasing ↓')
        result.append(grp)
    return pd.concat(result, ignore_index=True)


def load_dataset(dataset_key):
    if dataset_key == 'GISS Temperature':
        return load_giss_dataset()
    if dataset_key in ('SSP Precipitation', 'SSP Temperature'):
        return load_ssp_dataset(dataset_key)
    return load_csv_dataset(dataset_key)


# ============================================================
# App layout
# ============================================================
st.title("🌍 HDI vs. Climate & Environmental Trends")
st.markdown(
    "Explore how the **Human Development Index (HDI)** relates to long-term climate and "
    "environmental trends across countries. Select a dataset below, then use the sidebar "
    "filters to explore the data. **Hover** over any point for details."
)

# Dataset selector — prominent at the top
st.markdown("### Select Dataset")
selected_dataset = st.radio(
    label     = "",
    options   = list(DATASETS.keys()),
    horizontal= True,
)
meta   = DATASETS[selected_dataset]
is_ssp = meta.get('ssp', False)

st.caption(meta['description'])

# For SSP datasets, show a scenario toggle right below the description
if is_ssp:
    ssp_scenario = st.radio(
        "Scenario",
        options   = ['Both', 'SSP2-4.5', 'SSP5-8.5'],
        horizontal= True,
        key       = 'ssp_scenario',
    )

st.markdown("---")

# Load data
try:
    df_full = load_dataset(selected_dataset)
except FileNotFoundError as e:
    st.error(f"❌ File not found: **{e}**\n\nMake sure all required input files are in the same folder as this script.")
    st.stop()

if df_full.empty:
    st.warning("No data loaded. Check that your input files are present and formatted correctly.")
    st.stop()

# ============================================================
# Sidebar filters
# ============================================================
st.sidebar.header("🔧 Filters")

all_continents = sorted(df_full['Continent'].unique())
sel_continents = st.sidebar.multiselect("Continent", all_continents, default=all_continents)

hdi_min, hdi_max = float(df_full['HDI'].min()), float(df_full['HDI'].max())
sel_hdi = st.sidebar.slider("HDI range", hdi_min, hdi_max, (hdi_min, hdi_max), step=0.01)

t_min, t_max = float(df_full['Trend'].min()), float(df_full['Trend'].max())
sel_trend = st.sidebar.slider(
    f"Trend range ({meta['units']})", t_min, t_max, (t_min, t_max),
    format="%.4f"
)

show_outliers   = st.sidebar.checkbox("Show outliers",        value=True)
show_trendline  = st.sidebar.checkbox("Show OLS trendline",   value=True)
show_zeroline   = st.sidebar.checkbox("Show zero trend line", value=True)
show_hdi_thresh = st.sidebar.checkbox("Show HDI = 0.7 line",  value=True)
label_countries = st.sidebar.checkbox("Label all countries",  value=False)

st.sidebar.markdown("---")
marker_size = st.sidebar.slider("Marker size", 4, 20, 10)

# ============================================================
# Filter — handle SSP (multi-scenario) vs standard datasets
# ============================================================
if is_ssp and ssp_scenario != 'Both':
    df_full = df_full[df_full['Scenario'] == ssp_scenario].copy()

mask = (
    df_full['Continent'].isin(sel_continents) &
    df_full['HDI'].between(*sel_hdi) &
    df_full['Trend'].between(*sel_trend)
)
if not show_outliers:
    mask &= ~df_full['Outlier']
filtered = df_full[mask].copy()

# ============================================================
# Metrics row
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Countries shown", filtered['Country'].nunique())
c2.metric("Increasing ↑",    int((filtered['Trend'] > 0).sum()))
c3.metric("Decreasing ↓",    int((filtered['Trend'] < 0).sum()))
c4.metric("Mean trend",      f"{filtered['Trend'].mean():.5f} {meta['units']}")
c5.metric("Outliers",        int(filtered['Outlier'].sum()))

st.markdown("---")

# ============================================================
# Scatter plot
# ============================================================
fig = go.Figure()

# ── SSP "Both" view: colour by scenario (blue = SSP2-4.5, red = SSP5-8.5) ───
if is_ssp and ssp_scenario == 'Both':
    SSP_COLORS  = {'SSP2-4.5': '#38bdf8', 'SSP5-8.5': '#f87171'}
    SSP_SYMBOLS = {'SSP2-4.5': 'circle',  'SSP5-8.5': 'diamond'}

    for scenario in ['SSP2-4.5', 'SSP5-8.5']:
        sub_s  = filtered[filtered['Scenario'] == scenario]
        color  = SSP_COLORS[scenario]
        symbol = SSP_SYMBOLS[scenario]

        for is_outlier, opacity, border_color, border_w, size_mult, suffix in [
            (False, 0.72, 'white', 0.5, 1.0, ''),
            (True,  0.90, 'black', 1.8, 1.4, ' (outlier)'),
        ]:
            pts = sub_s[sub_s['Outlier'] == is_outlier]
            if pts.empty:
                continue
            fig.add_trace(go.Scatter(
                x    = pts['HDI'],
                y    = pts['Trend'],
                mode = 'markers+text' if label_countries else 'markers',
                name = f"{scenario}{suffix}",
                marker = dict(
                    color   = color,
                    size    = marker_size * size_mult,
                    symbol  = 'x' if is_outlier else symbol,
                    opacity = opacity,
                    line    = dict(color=border_color, width=border_w),
                ),
                text         = pts['Country'] if label_countries else None,
                textposition = 'top right',
                textfont     = dict(size=9),
                customdata   = pts[['Country', 'Continent', 'Direction', 'Outlier', 'Scenario']].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Continent: %{customdata[1]}<br>"
                    f"HDI: %{{x:.3f}}<br>"
                    f"{meta['y_label']}: %{{y:.6f}}<br>"
                    "Direction: %{customdata[2]}<br>"
                    "Scenario: %{customdata[4]}<br>"
                    "Outlier: %{customdata[3]}<extra></extra>"
                ),
                legendgroup = scenario,
                showlegend  = not is_outlier,
            ))

    # Per-scenario OLS lines
    if show_trendline:
        for scenario, color in SSP_COLORS.items():
            sub_s = filtered[(filtered['Scenario'] == scenario) & ~filtered['Outlier']]
            if len(sub_s) > 2:
                m, b   = np.polyfit(sub_s['HDI'], sub_s['Trend'], 1)
                x_line = np.linspace(filtered['HDI'].min(), filtered['HDI'].max(), 200)
                r      = sub_s['HDI'].corr(sub_s['Trend'])
                fig.add_trace(go.Scatter(
                    x    = x_line, y = m * x_line + b, mode = 'lines',
                    name = f"OLS {scenario} (r={r:.3f})",
                    line = dict(color=color, width=2, dash='dash'),
                    hovertemplate=f"OLS {scenario}: y = {m:.5f}x + {b:.5f}<extra></extra>",
                ))

# ── Standard datasets and single-scenario SSP: colour by continent ───────────
else:
    for continent in sorted(filtered['Continent'].unique()):
        sub    = filtered[filtered['Continent'] == continent]
        color  = CONTINENT_COLORS.get(continent, '#95a5a6')
        symbol = CONTINENT_MARKERS.get(continent, 'circle')

        for is_outlier, opacity, border_color, border_w, size_mult, suffix in [
            (False, 0.70, 'white', 0.5, 1.0, ''),
            (True,  0.90, 'black', 1.8, 1.4, ' (outlier)'),
        ]:
            pts = sub[sub['Outlier'] == is_outlier]
            if pts.empty:
                continue

            # Include Scenario in hover for single-scenario SSP views
            if is_ssp and 'Scenario' in pts.columns:
                customdata  = pts[['Country', 'Continent', 'Direction', 'Outlier', 'Scenario']].values
                hover_extra = "Scenario: %{customdata[4]}<br>"
            else:
                customdata  = pts[['Country', 'Continent', 'Direction', 'Outlier']].values
                hover_extra = ""

            fig.add_trace(go.Scatter(
                x    = pts['HDI'],
                y    = pts['Trend'],
                mode = 'markers+text' if label_countries else 'markers',
                name = f"{continent}{suffix}",
                marker = dict(
                    color   = color,
                    size    = marker_size * size_mult,
                    symbol  = 'diamond' if is_outlier else symbol,
                    opacity = opacity,
                    line    = dict(color=border_color, width=border_w),
                ),
                text         = pts['Country'] if label_countries else None,
                textposition = 'top right',
                textfont     = dict(size=9),
                customdata   = customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Continent: %{customdata[1]}<br>"
                    f"HDI: %{{x:.3f}}<br>"
                    f"{meta['y_label']}: %{{y:.6f}}<br>"
                    "Direction: %{customdata[2]}<br>"
                    + hover_extra +
                    "Outlier: %{customdata[3]}<extra></extra>"
                ),
                legendgroup = continent,
                showlegend  = not is_outlier,
            ))

    # Single OLS trendline
    if show_trendline and len(filtered) > 2:
        fit = filtered[~filtered['Outlier']]
        if len(fit) > 2:
            m, b   = np.polyfit(fit['HDI'], fit['Trend'], 1)
            x_line = np.linspace(filtered['HDI'].min(), filtered['HDI'].max(), 200)
            r      = fit['HDI'].corr(fit['Trend'])
            fig.add_trace(go.Scatter(
                x    = x_line, y = m * x_line + b, mode = 'lines',
                name = f"OLS fit (r={r:.3f})",
                line = dict(color='#2c3e50', width=2),
                hovertemplate=f"OLS: y = {m:.4f}x + {b:.4f}<extra></extra>",
            ))

# Reference lines (shared by all datasets)
if show_zeroline:
    fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.4,
                  annotation_text='No trend', annotation_position='bottom right')
if show_hdi_thresh:
    fig.add_vline(x=0.7, line_dash='dot', line_color='grey', opacity=0.5,
                  annotation_text='HDI = 0.7', annotation_position='top left')

plot_title = meta['title']
if is_ssp and ssp_scenario != 'Both':
    plot_title += f" — {ssp_scenario}"

fig.update_layout(
    title      = dict(text=plot_title, font=dict(size=18)),
    xaxis      = dict(title='Human Development Index (HDI)', range=[0.33, 1.02],
                      showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    yaxis      = dict(title=meta['y_label'], showgrid=True,
                      gridcolor='rgba(200,200,200,0.3)'),
    legend     = dict(title='Legend', x=1.01, y=1, bgcolor='rgba(255,255,255,0.85)',
                      bordercolor='lightgrey', borderwidth=1),
    hovermode  = 'closest',
    plot_bgcolor='white',
    height     = 650,
    margin     = dict(l=60, r=180, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🏆 Top & Bottom Countries", "📋 Data Table"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        # For SSP "Both" view, show breakdown by scenario instead of continent
        if is_ssp and ssp_scenario == 'Both':
            st.subheader("By Scenario")
            ss = (filtered.groupby('Scenario')['Trend']
                  .agg(n='count', mean='mean', std='std', median='median')
                  .reset_index())
            ss.columns = ['Scenario', 'N', f'Mean ({meta["units"]})', 'Std', 'Median']
            st.dataframe(ss.round(6), use_container_width=True, hide_index=True)
        else:
            st.subheader("By Continent")
            cs = (filtered.groupby('Continent')['Trend']
                  .agg(n='count', mean='mean', std='std', median='median')
                  .sort_values('mean', ascending=False).reset_index())
            cs.columns = ['Continent', 'N', f'Mean ({meta["units"]})', 'Std', 'Median']
            st.dataframe(cs.round(4), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("By HDI Group")
        bins   = [0, 0.55, 0.70, 0.80, 1.0]
        labels = ['Low (<0.55)', 'Medium (0.55–0.70)', 'High (0.70–0.80)', 'Very High (>0.80)']
        filtered['HDI_Group'] = pd.cut(filtered['HDI'], bins=bins, labels=labels)
        hs = (filtered.groupby('HDI_Group', observed=True)['Trend']
              .agg(n='count', mean='mean', std='std').reset_index())
        hs.columns = ['HDI Group', 'N', f'Mean ({meta["units"]})', 'Std']
        st.dataframe(hs.round(4), use_container_width=True, hide_index=True)

    st.subheader("Correlation")
    if is_ssp and ssp_scenario == 'Both':
        # Report r separately for each scenario
        for scenario in ['SSP2-4.5', 'SSP5-8.5']:
            sub_s    = filtered[filtered['Scenario'] == scenario]
            sub_norm = sub_s[~sub_s['Outlier']]
            r_all    = sub_s['HDI'].corr(sub_s['Trend'])
            r_normal = sub_norm['HDI'].corr(sub_norm['Trend'])
            st.markdown(
                f"**{scenario}** — r (all): `{r_all:.3f}` | "
                f"r (excl. outliers): `{r_normal:.3f}` | n: `{len(sub_s)}`"
            )
    else:
        r_all    = filtered['HDI'].corr(filtered['Trend'])
        r_normal = filtered[~filtered['Outlier']]['HDI'].corr(
                   filtered[~filtered['Outlier']]['Trend'])
        st.markdown(
            f"- **r (all countries):** `{r_all:.3f}`  \n"
            f"- **r (excl. outliers):** `{r_normal:.3f}`  \n"
            f"- **n:** `{len(filtered)}`"
        )

with tab2:
    n_show    = st.slider("Number of countries", 5, 20, 10)
    cols_show = ['Country', 'Continent', 'HDI', 'Trend', 'Outlier']

    if is_ssp and ssp_scenario == 'Both':
        # Top/bottom per scenario
        for scenario in ['SSP2-4.5', 'SSP5-8.5']:
            st.markdown(f"**{scenario}**")
            sub_s = filtered[filtered['Scenario'] == scenario]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("🔴 Highest trend (↑)")
                st.dataframe(
                    sub_s.nlargest(n_show, 'Trend')[cols_show]
                    .rename(columns={'Trend': meta['y_label']}).round(6).reset_index(drop=True),
                    use_container_width=True, hide_index=True
                )
            with c2:
                st.markdown("🔵 Lowest trend (↓)")
                st.dataframe(
                    sub_s.nsmallest(n_show, 'Trend')[cols_show]
                    .rename(columns={'Trend': meta['y_label']}).round(6).reset_index(drop=True),
                    use_container_width=True, hide_index=True
                )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"🔴 Highest trend (↑)")
            st.dataframe(
                filtered.nlargest(n_show, 'Trend')[cols_show]
                .rename(columns={'Trend': meta['y_label']}).round(4).reset_index(drop=True),
                use_container_width=True, hide_index=True
            )
        with c2:
            st.subheader(f"🔵 Lowest trend (↓)")
            st.dataframe(
                filtered.nsmallest(n_show, 'Trend')[cols_show]
                .rename(columns={'Trend': meta['y_label']}).round(4).reset_index(drop=True),
                use_container_width=True, hide_index=True
            )

with tab3:
    st.subheader(f"All filtered countries ({filtered['Country'].nunique()})")
    table_cols = ['Country', 'Continent', 'HDI', 'Trend', 'Direction', 'Outlier']
    if is_ssp and 'Scenario' in filtered.columns:
        table_cols.insert(2, 'Scenario')
    display_df = (filtered[table_cols]
                  .rename(columns={'Trend': meta['y_label']})
                  .sort_values(meta['y_label'], ascending=False)
                  .round(4).reset_index(drop=True))
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data      = display_df.to_csv(index=False).encode('utf-8'),
        file_name = f"hdi_vs_{selected_dataset.split()[1].lower()}_trend_filtered.csv",
        mime      = 'text/csv',
    )

st.markdown("---")
st.caption(
    "Data sources: PM2.5 — Global Satellite PM2.5 (GEE) | "
    "Precipitation — GPM IMERG | Water Storage — GRACE | "
    "Temperature — NASA GISS | "
    "CMIP6 Projections — SSP2-4.5 & SSP5-8.5 via GEE | "
    "HDI — UNDP Human Development Report 2025"
)
