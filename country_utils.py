"""
country_utils.py
────────────────
Shared constants, data-loading helpers, and index definitions
extracted from 8_Country_Profile.py.

Import this module in any page that needs country environmental data:

    from country_utils import (
        INDICES, COUNTRY_COORDS, ALIASES, CONTINENT_MAPPING,
        get_all_indices, load_hdi, fuzzy_resolve,
    )
"""

import numpy as np
import pandas as pd
from difflib import get_close_matches
import streamlit as st

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ─────────────────────────────────────────────────────────────────────────────
# Continent mapping
# ─────────────────────────────────────────────────────────────────────────────
CONTINENT_MAPPING = {
    'algeria':'Africa','angola':'Africa','benin':'Africa','botswana':'Africa',
    'burkina faso':'Africa','burundi':'Africa','cabo verde':'Africa','cape verde':'Africa',
    'cameroon':'Africa','central african republic':'Africa','central african rep.':'Africa',
    'central african rep':'Africa','chad':'Africa','comoros':'Africa','congo':'Africa',
    'congo (democratic republic of the)':'Africa',"côte d'ivoire":'Africa',"cote d'ivoire":'Africa',
    'djibouti':'Africa','egypt':'Africa','equatorial guinea':'Africa','eritrea':'Africa',
    'eswatini':'Africa','eswatini (kingdom of)':'Africa','ethiopia':'Africa','gabon':'Africa',
    'gambia':'Africa','ghana':'Africa','guinea':'Africa','guinea-bissau':'Africa',
    'kenya':'Africa','lesotho':'Africa','liberia':'Africa','libya':'Africa',
    'madagascar':'Africa','malawi':'Africa','mali':'Africa','mauritania':'Africa',
    'mauritius':'Africa','morocco':'Africa','mozambique':'Africa','namibia':'Africa',
    'niger':'Africa','nigeria':'Africa','rwanda':'Africa','senegal':'Africa',
    'seychelles':'Africa','sierra leone':'Africa','somalia':'Africa','south africa':'Africa',
    'south sudan':'Africa','sudan':'Africa','tanzania':'Africa',
    'tanzania (united republic of)':'Africa','togo':'Africa','tunisia':'Africa',
    'uganda':'Africa','zambia':'Africa','zimbabwe':'Africa',
    'afghanistan':'Asia','armenia':'Asia','azerbaijan':'Asia','bahrain':'Asia',
    'bangladesh':'Asia','bhutan':'Asia','brunei darussalam':'Asia','cambodia':'Asia',
    'china':'Asia','cyprus':'Asia','georgia':'Asia','india':'Asia','indonesia':'Asia',
    'iran':'Asia','iran (islamic republic of)':'Asia','iraq':'Asia','israel':'Asia',
    'japan':'Asia','jordan':'Asia','kazakhstan':'Asia','korea (republic of)':'Asia',
    "korea (democratic people's rep. of)":'Asia','kuwait':'Asia','kyrgyzstan':'Asia',
    "lao people's democratic republic":'Asia','laos':'Asia','lebanon':'Asia',
    'malaysia':'Asia','maldives':'Asia','mongolia':'Asia','myanmar':'Asia','nepal':'Asia',
    'oman':'Asia','pakistan':'Asia','state of palestine':'Asia','palestine':'Asia',
    'philippines':'Asia','qatar':'Asia','saudi arabia':'Asia','singapore':'Asia',
    'sri lanka':'Asia','syrian arab republic':'Asia','tajikistan':'Asia','thailand':'Asia',
    'timor-leste':'Asia','turkey':'Asia','türkiye':'Asia','turkmenistan':'Asia',
    'united arab emirates':'Asia','uzbekistan':'Asia','viet nam':'Asia','vietnam':'Asia',
    'yemen':'Asia',
    'albania':'Europe','austria':'Europe','belarus':'Europe','belgium':'Europe',
    'bosnia and herzegovina':'Europe','bulgaria':'Europe','croatia':'Europe',
    'czechia':'Europe','czech republic':'Europe','denmark':'Europe','estonia':'Europe',
    'finland':'Europe','france':'Europe','germany':'Europe','greece':'Europe',
    'hungary':'Europe','iceland':'Europe','ireland':'Europe','italy':'Europe',
    'latvia':'Europe','liechtenstein':'Europe','lithuania':'Europe','luxembourg':'Europe',
    'malta':'Europe','moldova':'Europe','moldova (republic of)':'Europe','monaco':'Europe',
    'montenegro':'Europe','netherlands':'Europe','north macedonia':'Europe','norway':'Europe',
    'poland':'Europe','portugal':'Europe','romania':'Europe','russia':'Europe',
    'russian federation':'Europe','serbia':'Europe','slovakia':'Europe','slovenia':'Europe',
    'spain':'Europe','sweden':'Europe','switzerland':'Europe','ukraine':'Europe',
    'united kingdom':'Europe',
    'antigua and barbuda':'Americas','argentina':'Americas','bahamas':'Americas',
    'barbados':'Americas','belize':'Americas','bolivia':'Americas',
    'bolivia (plurinational state of)':'Americas','brazil':'Americas','canada':'Americas',
    'chile':'Americas','colombia':'Americas','costa rica':'Americas','cuba':'Americas',
    'dominican republic':'Americas','ecuador':'Americas','el salvador':'Americas',
    'guatemala':'Americas','guyana':'Americas','haiti':'Americas','honduras':'Americas',
    'jamaica':'Americas','mexico':'Americas','nicaragua':'Americas','panama':'Americas',
    'paraguay':'Americas','peru':'Americas','suriname':'Americas',
    'trinidad and tobago':'Americas','united states':'Americas',
    'united states of america':'Americas','uruguay':'Americas',
    'venezuela':'Americas','venezuela (bolivarian republic of)':'Americas',
    'australia':'Oceania','fiji':'Oceania','kiribati':'Oceania','new zealand':'Oceania',
    'papua new guinea':'Oceania','samoa':'Oceania','solomon islands':'Oceania',
    'tonga':'Oceania','vanuatu':'Oceania',
}

# ─────────────────────────────────────────────────────────────────────────────
# Country centroids
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_COORDS = {
    'afghanistan':(33.0,65.0),'albania':(41.0,20.0),'algeria':(28.0,3.0),
    'angola':(-12.5,18.5),'argentina':(-34.0,-64.0),'armenia':(40.0,45.0),
    'australia':(-25.0,135.0),'austria':(47.5,14.5),'azerbaijan':(40.5,47.5),
    'bahamas':(24.25,-76.0),'bangladesh':(24.0,90.0),'belarus':(53.0,28.0),
    'belgium':(50.5,4.5),'belize':(17.25,-88.75),'benin':(9.5,2.25),
    'bhutan':(27.5,90.5),'bolivia (plurinational state of)':(-17.0,-65.0),
    'bosnia and herzegovina':(44.0,18.0),'botswana':(-22.0,24.0),
    'brazil':(-10.0,-55.0),'bulgaria':(43.0,25.0),'burkina faso':(13.0,-2.0),
    'burundi':(-3.5,30.0),'cabo verde':(16.0,-24.0),'cambodia':(13.0,105.0),
    'cameroon':(6.0,12.0),'canada':(60.0,-95.0),'central african republic':(7.0,21.0),
    'chad':(15.0,19.0),'chile':(-30.0,-71.0),'china':(35.0,105.0),
    'colombia':(4.0,-72.0),'congo':(-1.0,15.0),
    'congo (democratic republic of the)':(-4.0,23.0),'costa rica':(10.0,-84.0),
    "côte d'ivoire":(8.0,-5.0),'croatia':(45.0,16.0),'cuba':(21.5,-80.0),
    'cyprus':(35.0,33.0),'czechia':(49.75,15.5),'denmark':(56.0,10.0),
    'djibouti':(11.5,43.0),'dominican republic':(19.0,-70.5),
    'ecuador':(-2.0,-77.5),'egypt':(27.0,30.0),'el salvador':(13.83,-88.92),
    'eritrea':(15.0,39.0),'estonia':(59.0,26.0),'eswatini (kingdom of)':(-26.5,31.5),
    'ethiopia':(8.0,38.0),'fiji':(-17.71,178.06),'finland':(64.0,26.0),
    'france':(46.0,2.0),'gabon':(-1.0,11.75),'gambia':(13.5,-15.5),
    'georgia':(42.0,43.5),'germany':(51.0,9.0),'ghana':(8.0,-2.0),
    'greece':(39.0,22.0),'guatemala':(15.5,-90.25),'guinea':(11.0,-10.0),
    'guinea-bissau':(12.0,-15.0),'guyana':(5.0,-59.0),'haiti':(19.0,-72.42),
    'honduras':(15.0,-86.5),'hungary':(47.0,20.0),'iceland':(65.0,-18.0),
    'india':(20.0,77.0),'indonesia':(-5.0,120.0),'iran (islamic republic of)':(32.0,53.0),
    'iraq':(33.0,44.0),'ireland':(53.0,-8.0),'israel':(31.5,34.75),
    'italy':(42.83,12.83),'jamaica':(18.25,-77.5),'japan':(36.0,138.0),
    'jordan':(31.0,36.0),'kazakhstan':(48.0,68.0),'kenya':(1.0,38.0),
    'korea (republic of)':(37.0,127.5),"korea (democratic people's rep. of)":(40.0,127.0),
    'kyrgyzstan':(41.0,75.0),"lao people's democratic republic":(18.0,105.0),
    'latvia':(57.0,25.0),'lebanon':(33.83,35.83),'lesotho':(-29.5,28.5),
    'liberia':(6.5,-9.5),'libya':(25.0,17.0),'lithuania':(56.0,24.0),
    'luxembourg':(49.75,6.17),'madagascar':(-20.0,47.0),'malawi':(-13.5,34.0),
    'malaysia':(2.5,112.5),'maldives':(3.2,73.0),'mali':(17.0,-4.0),
    'mauritania':(20.0,-12.0),'mauritius':(-20.28,57.58),'mexico':(23.0,-102.0),
    'moldova (republic of)':(47.0,29.0),'mongolia':(46.0,105.0),
    'montenegro':(42.5,19.3),'morocco':(32.0,-5.0),'mozambique':(-18.25,35.0),
    'myanmar':(22.0,98.0),'namibia':(-22.0,17.0),'nepal':(28.0,84.0),
    'netherlands':(52.5,5.75),'new zealand':(-41.0,174.0),'nicaragua':(13.0,-85.0),
    'niger':(16.0,8.0),'nigeria':(10.0,8.0),'north macedonia':(41.83,22.0),
    'norway':(62.0,10.0),'pakistan':(30.0,70.0),'panama':(9.0,-80.0),
    'papua new guinea':(-6.0,147.0),'paraguay':(-23.0,-58.0),'peru':(-10.0,-76.0),
    'philippines':(13.0,122.0),'poland':(52.0,20.0),'portugal':(39.5,-8.0),
    'romania':(46.0,25.0),'russian federation':(60.0,100.0),'rwanda':(-2.0,30.0),
    'saudi arabia':(25.0,45.0),'senegal':(14.0,-15.0),'serbia':(44.0,21.0),
    'sierra leone':(8.5,-11.5),'singapore':(1.37,103.8),'slovakia':(48.67,19.5),
    'slovenia':(46.0,15.0),'somalia':(10.0,49.0),'south africa':(-29.0,24.0),
    'south sudan':(8.0,30.0),'spain':(40.0,-4.0),'sri lanka':(7.0,81.0),
    'state of palestine':(32.0,35.25),'sudan':(15.0,30.0),'sweden':(62.0,15.0),
    'switzerland':(47.0,8.0),'syrian arab republic':(35.0,38.0),
    'tajikistan':(39.0,71.0),'tanzania (united republic of)':(-6.0,35.0),
    'thailand':(15.0,100.0),'timor-leste':(-8.83,125.92),'togo':(8.0,1.17),
    'tunisia':(34.0,9.0),'türkiye':(39.0,35.0),'turkmenistan':(40.0,60.0),
    'uganda':(1.0,32.0),'ukraine':(49.0,32.0),'united kingdom':(54.0,-2.0),
    'united states':(38.0,-97.0),'uruguay':(-33.0,-56.0),'uzbekistan':(41.0,64.0),
    'venezuela (bolivarian republic of)':(8.0,-66.0),'viet nam':(16.0,106.0),
    'yemen':(15.5,48.0),'zambia':(-13.0,27.5),'zimbabwe':(-20.0,30.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# Aliases  (common names → canonical lower-case key in COUNTRY_COORDS)
# ─────────────────────────────────────────────────────────────────────────────
ALIASES = {
    'north korea'       : "korea (democratic people's rep. of)",
    'south korea'       : 'korea (republic of)',
    'iran'              : 'iran (islamic republic of)',
    'syria'             : 'syrian arab republic',
    'vietnam'           : 'viet nam',
    'laos'              : "lao people's democratic republic",
    'lao pdr'           : "lao people's democratic republic",
    'congo-kinshasa'    : 'congo (democratic republic of the)',
    'dr congo'          : 'congo (democratic republic of the)',
    'drc'               : 'congo (democratic republic of the)',
    'dem rep of the congo': 'congo (democratic republic of the)',
    'congo-brazzaville' : 'congo',
    'rep of the congo'  : 'congo',
    'ivory coast'       : "côte d'ivoire",
    "cote d'ivoire"     : "côte d'ivoire",
    'the gambia'        : 'gambia',
    'czech republic'    : 'czechia',
    'czech rep'         : 'czechia',
    'slovak republic'   : 'slovakia',
    'macedonia'         : 'north macedonia',
    'turkey'            : 'türkiye',
    'russia'            : 'russian federation',
    'east timor'        : 'timor-leste',
    'burma'             : 'myanmar',
    'cape verde'        : 'cabo verde',
    'somaliland'        : 'somalia',
    'tanzania'          : 'tanzania (united republic of)',
    'bolivia'           : 'bolivia (plurinational state of)',
    'venezuela'         : 'venezuela (bolivarian republic of)',
    'moldova'           : 'moldova (republic of)',
    'eswatini'          : 'eswatini (kingdom of)',
    'palestine'         : 'state of palestine',
    'korea, south'      : 'korea (republic of)',
    'korea, north'      : "korea (democratic people's rep. of)",
    'central african rep': 'central african republic',
    'central african rep.': 'central african republic',
    'usa'               : 'united states',
    'us'                : 'united states',
    'america'           : 'united states',
    'uk'                : 'united kingdom',
    'great britain'     : 'united kingdom',
    'england'           : 'united kingdom',
    'uae'               : 'united arab emirates',
    'bosnia'            : 'bosnia and herzegovina',
    'trinidad'          : 'trinidad and tobago',
    'kyrgyz republic'   : 'kyrgyzstan',
}

# ─────────────────────────────────────────────────────────────────────────────
# Index definitions
# ─────────────────────────────────────────────────────────────────────────────
INDICES = {
    'gpm_precip': {
        'label'      : 'Precipitation Trend',
        'units'      : 'mm/yr',
        'icon'       : '🌧️',
        'source'     : 'NASA GPM IMERG',
        'file'       : 'GPM_Precip_Trend_By_Country.csv',
        'kind'       : 'csv',
        'country_col': 'country_name',
        'trend_col'  : 'precip_trend_mm_per_year',
        'pos_good'   : True,
    },
    'grace_water': {
        'label'      : 'Water Storage Trend',
        'units'      : 'cm/yr',
        'icon'       : '💧',
        'source'     : 'NASA GRACE / GRACE-FO',
        'file'       : 'GRACE_TWS_Trend_By_Country_Global.csv',
        'kind'       : 'csv',
        'country_col': 'country_na',
        'trend_col'  : 'trend_cm_per_year',
        'pos_good'   : True,
    },
    'pm25': {
        'label'      : 'PM2.5 Trend',
        'units'      : 'µg/m³/yr',
        'icon'       : '💨',
        'source'     : 'Global Satellite PM2.5 (GEE)',
        'file'       : 'PM25_Monthly_Trend_By_Country.csv',
        'kind'       : 'csv',
        'country_col': 'country_name',
        'trend_col'  : 'pm25_trend_ug_m3_per_year',
        'pos_good'   : False,
    },
    'giss_temp': {
        'label'  : 'Surface Temperature Trend',
        'units'  : '°C/decade',
        'icon'   : '🌡️',
        'source' : 'NASA GISTEMP',
        'file'   : 'GISS_temperature_trend_1980_2025.nc',
        'kind'   : 'netcdf',
        'pos_good': False,
    },
    'ssp245_temp': {
        'label'  : 'Future Temp (SSP2-4.5)',
        'units'  : '°C/yr',
        'icon'   : '🔮',
        'source' : 'CMIP6 SSP2-4.5',
        'file'   : 'tas_slope_ssp245.tif',
        'kind'   : 'tif',
        'pos_good': False,
    },
    'ssp585_temp': {
        'label'  : 'Future Temp (SSP5-8.5)',
        'units'  : '°C/yr',
        'icon'   : '🔮',
        'source' : 'CMIP6 SSP5-8.5',
        'file'   : 'tas_slope_ssp585.tif',
        'kind'   : 'tif',
        'pos_good': False,
    },
    'ssp245_precip': {
        'label'  : 'Future Precip (SSP2-4.5)',
        'units'  : 'mm/day/yr',
        'icon'   : '🌦️',
        'source' : 'CMIP6 SSP2-4.5',
        'file'   : 'pr_slope_ssp245.tif',
        'kind'   : 'tif',
        'pos_good': True,
    },
    'ssp585_precip': {
        'label'  : 'Future Precip (SSP5-8.5)',
        'units'  : 'mm/day/yr',
        'icon'   : '🌦️',
        'source' : 'CMIP6 SSP5-8.5',
        'file'   : 'pr_slope_ssp585.tif',
        'kind'   : 'tif',
        'pos_good': True,
    },
}

OBS_KEYS = ['gpm_precip', 'grace_water', 'pm25', 'giss_temp']
FUT_KEYS = ['ssp245_temp', 'ssp585_temp', 'ssp245_precip', 'ssp585_precip']

# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy country resolver
# ─────────────────────────────────────────────────────────────────────────────
def _build_corpus():
    corpus = {}
    for canonical in COUNTRY_COORDS:
        corpus[canonical] = canonical
    for alias, canonical in ALIASES.items():
        corpus[alias.lower()] = canonical
    return corpus

_CORPUS = _build_corpus()
_CORPUS_KEYS = list(_CORPUS.keys())


def fuzzy_resolve(query: str):
    """
    Resolve free-text → (canonical_lower, display_name, alternatives, error).
    Returns (None, None, [], msg) on failure.
    """
    if not query or not query.strip():
        return None, None, [], None

    q = query.strip().lower()

    if q in _CORPUS:
        c = _CORPUS[q]
        return c, c.title(), [], None

    close = get_close_matches(q, _CORPUS_KEYS, n=6, cutoff=0.6)
    if close:
        best = _CORPUS[close[0]]
        alts = list(dict.fromkeys(_CORPUS[k].title() for k in close[1:]))[:4]
        return best, best.title(), alts, None

    hits = [k for k in _CORPUS_KEYS if k.startswith(q)] or \
           [k for k in _CORPUS_KEYS if q in k]
    if hits:
        best = _CORPUS[hits[0]]
        alts = list(dict.fromkeys(_CORPUS[k].title() for k in hits[1:5]))
        return best, best.title(), alts, None

    return None, None, [], f"No country found for **'{query}'**."


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders  (identical logic to 8_Country_Profile.py)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_hdi():
    import os
    path = 'HDR25_Statistical_Annex_HDI_Table.csv'
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, skiprows=3)
    df.columns = df.columns.str.strip()
    ccol = 'Country' if 'Country' in df.columns else df.columns[1]
    vcol = df.columns[2]
    df[ccol] = df[ccol].str.strip()
    df[vcol] = pd.to_numeric(df[vcol], errors='coerce')
    df = df.dropna(subset=[ccol, vcol])
    return dict(zip(df[ccol].str.lower(), df[vcol]))


@st.cache_data
def load_csv_index(filepath, country_col, trend_col):
    import os
    if not os.path.exists(filepath):
        return None, None
    df = pd.read_csv(filepath)
    df[country_col] = df[country_col].str.strip().str.lower()
    df = df.drop_duplicates(subset=country_col)
    df[trend_col] = pd.to_numeric(df[trend_col], errors='coerce')
    df = df.dropna(subset=[trend_col])
    return dict(zip(df[country_col], df[trend_col])), float(df[trend_col].mean())


@st.cache_data
def load_giss_index():
    import os
    if not os.path.exists('GISS_temperature_trend_1980_2025.nc'):
        return None, None
    try:
        import netCDF4 as nc
    except ImportError:
        return None, None
    ds   = nc.Dataset('GISS_temperature_trend_1980_2025.nc', 'r')
    lat  = ds.variables['lat'][:]
    lon  = ds.variables['lon'][:]
    data = ds.variables['TEMPTREND'][:]
    miss = ds.variables['TEMPTREND'].missing_value
    ds.close()

    def _sample(lat_c, lon_c, w=3):
        li = np.abs(lat - lat_c).argmin()
        lj = np.abs(lon - lon_c).argmin()
        chunk = data[max(0, li-w):li+w+1, max(0, lj-w):lj+w+1]
        valid = chunk[chunk != miss]
        return float(np.mean(valid)) if len(valid) else np.nan

    lookup, vals = {}, []
    for c, coords in COUNTRY_COORDS.items():
        v = _sample(*coords)
        if not np.isnan(v):
            lookup[c] = v
            vals.append(v)
    return lookup, (float(np.mean(vals)) if vals else np.nan)


@st.cache_data
def load_tif_index(filepath):
    import os
    if not os.path.exists(filepath) or not HAS_RASTERIO:
        return None, None
    try:
        with rasterio.open(filepath) as src:
            nodata = src.nodata
            lookup, vals = {}, []
            for c, (lat_c, lon_c) in COUNTRY_COORDS.items():
                try:
                    row, col = src.index(lon_c, lat_c)
                    h, w = src.height, src.width
                    r0,r1 = max(0,row-2), min(h,row+3)
                    c0,c1 = max(0,col-2), min(w,col+3)
                    patch = src.read(1, window=rasterio.windows.Window(
                        c0, r0, c1-c0, r1-r0)).astype(float)
                    if nodata is not None:
                        patch = np.where(patch == nodata, np.nan, patch)
                    patch = np.where(np.abs(patch) > 1e10, np.nan, patch)
                    valid = patch[~np.isnan(patch)]
                    if len(valid):
                        v = float(np.mean(valid))
                        lookup[c] = v
                        vals.append(v)
                except Exception:
                    continue
        return lookup, (float(np.mean(vals)) if vals else np.nan)
    except Exception:
        return None, None


def _name_variants(canonical: str) -> list:
    """
    Return every lower-case name variant worth trying against a CSV lookup
    for the given canonical key.

    Covers:
    - the canonical key itself          e.g. 'united states'
    - all ALIASES that point to it      e.g. 'united states of america', 'usa', 'us'
    - reverse-ALIASES (long → short)    e.g. ALIASES values that equal canonical
    - known long-form extras embedded
      in CONTINENT_MAPPING              e.g. 'united states of america'
    """
    seen = set()
    variants = []

    def _add(name):
        n = (name or '').strip().lower()
        if n and n not in seen:
            seen.add(n)
            variants.append(n)

    _add(canonical)

    # Every alias that resolves TO this canonical
    for alias, canon in ALIASES.items():
        if canon == canonical:
            _add(alias)

    # Every key in CONTINENT_MAPPING that is close to canonical
    # (catches 'united states of america' for 'united states', etc.)
    for k in CONTINENT_MAPPING:
        if k.startswith(canonical) or canonical.startswith(k):
            _add(k)

    return variants


def get_all_indices(canonical: str) -> dict:
    """Return {index_key: (value, global_mean)} for every index in INDICES."""
    results = {}
    variants = _name_variants(canonical)

    for key, meta in INDICES.items():
        if meta['kind'] == 'csv':
            lookup, gmean = load_csv_index(
                meta['file'], meta['country_col'], meta['trend_col'])
            lkp = lookup or {}
            # Try every name variant in order; take the first hit
            val = None
            for v in variants:
                val = lkp.get(v)
                if val is not None:
                    break
            results[key] = (val, gmean)

        elif meta['kind'] == 'netcdf':
            lookup, gmean = load_giss_index()
            results[key] = ((lookup or {}).get(canonical), gmean)

        elif meta['kind'] == 'tif':
            lookup, gmean = load_tif_index(meta['file'])
            results[key] = ((lookup or {}).get(canonical), gmean)

    return results


def hdi_tier(val):
    if val is None:
        return '—', '#8b949e', 'rgba(139,148,158,0.15)'
    if val >= 0.800:
        return 'Very High', '#3498db', 'rgba(52,152,219,0.15)'
    if val >= 0.700:
        return 'High', '#2ecc71', 'rgba(46,204,113,0.15)'
    if val >= 0.550:
        return 'Medium', '#f39c12', 'rgba(243,156,18,0.15)'
    return 'Low', '#e74c3c', 'rgba(231,76,60,0.15)'
