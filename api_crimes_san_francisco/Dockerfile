FROM python:3.10

RUN adduser --disabled-password --gecos '' api-user

# Directorio de trabajo
WORKDIR /opt/crimessf-api

# Copiar SOLO la carpeta del API
COPY crimesSF-api /opt/crimessf-api/

RUN pip install --upgrade pip

# Instalar dependencias desde la carpeta correcta
RUN pip install -r /opt/crimessf-api/requirements.txt

RUN chmod +x /opt/crimessf-api/run.sh
RUN chown -R api-user:api-user /opt/crimessf-api

USER api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]