from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st


@st.cache_resource
def load_districts():
    """Carga el GeoJSON de distritos una sola vez."""
    geojson_path = Path(__file__).resolve().parents[2] / "data" / "sf_police_districts.geojson"
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def detect_district(lat: float, lon: float) -> str | None:
    """Detecta el distrito policial a partir de lat/lon."""
    gdf = load_districts()
    point = Point(lon, lat)
    match = gdf[gdf.geometry.contains(point)]
    if not match.empty:
        return match.iloc[0]["district"]
    return None
