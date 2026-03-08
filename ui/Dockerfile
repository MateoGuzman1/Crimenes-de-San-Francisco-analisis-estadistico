FROM python:3.10

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' app-user

# Definir directorio de trabajo
WORKDIR /opt/crimessf-ui

# Instalar dependencias del sistema (requeridas por fiona/geopandas)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
ADD . /opt/crimessf-ui/
RUN pip install --upgrade pip
RUN pip install -r /opt/crimessf-ui/requirements.txt

# Cambiar propiedad al usuario
RUN chown -R app-user:app-user ./

USER app-user

# Puerto a exponer para el dashboard
EXPOSE 8501

# Ejecutar dashboard
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
