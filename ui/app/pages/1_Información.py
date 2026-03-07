import streamlit as st

st.set_page_config(page_title="Información", layout="wide")

st.title("Información del Proyecto")

st.write("""
Este dashboard es un prototipo del modelo de clasificación de crimen en San Francisco.

Variables de entrada:
- Fecha
- Hora
- Latitud
- Longitud
- Distrito

Actualmente funciona en modo MOCK (simulado).
En la siguiente fase se conectará a una API real desplegada en AWS.
""")
