import json
import folium
import streamlit as st
from streamlit_folium import st_folium

from components.district_detector import detect_district, load_districts


def render_clickable_map(
    lat: float,
    lon: float,
    distrito: str | None = None,
    zoom: int = 13,
    height: int = 360,
    radius_m: int = 250,
) -> None:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True)

    # Cargar GeoJSON y quedarnos solo con columnas serializables
    districts_gdf = load_districts()[["district", "geometry"]].copy()
    districts_gdf["district"] = districts_gdf["district"].astype(str)

    folium.GeoJson(
        json.loads(districts_gdf.to_json()),
        name="Distritos policiales",
        style_function=lambda feature: {
            "fillColor": "#3186cc",
            "color": "#e41212",
            "weight": 1,
            "fillOpacity": 0.15,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["district"],
            aliases=["Distrito:"],
            sticky=True
        )
    ).add_to(m)

    tooltip = f"Distrito: {distrito}" if distrito else "Ubicación seleccionada"
    folium.Marker([lat, lon], tooltip=tooltip).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        fill=True,
        fill_opacity=0.08,
    ).add_to(m)

    st.caption("📍 Haz click en el mapa para seleccionar Latitud/Longitud.")

    map_data = st_folium(m, height=height, width=None)

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        new_lat = float(clicked["lat"])
        new_lon = float(clicked["lng"])

        lat_changed = st.session_state.get("map_lat") != new_lat
        lon_changed = st.session_state.get("map_lon") != new_lon

        if lat_changed or lon_changed:
            st.session_state["map_lat"] = new_lat
            st.session_state["map_lon"] = new_lon

            detected_district = detect_district(new_lat, new_lon)
            st.session_state["map_distrito"] = detected_district

            st.session_state["map_clicked"] = True
            st.rerun()