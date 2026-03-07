FROM python:3.10

# Crear usuario
RUN adduser --disabled-password --gecos '' api-user

# Directorio de trabajo
WORKDIR /opt/crimessf-api

# Copiar el proyecto completo
COPY crimesSF-api/ /opt/crimessf-api/

# Actualizar pip
RUN pip install --upgrade pip

# Instalar dependencias
RUN pip install -r /opt/crimessf-api/requirements.txt

# Permisos
RUN chmod +x /opt/crimessf-api/run.sh
RUN chown -R api-user:api-user /opt/crimessf-api

USER api-user

# Puerto
EXPOSE 8001

# Ejecutar script de arranque
CMD ["bash", "run.sh"]