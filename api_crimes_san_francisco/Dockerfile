FROM python:3.10

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/crimessf-api

# Copiar solo el contenido del API
COPY . /opt/crimessf-api/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x run.sh
RUN chown -R api-user:api-user /opt/crimessf-api

USER api-user

EXPOSE 8001

CMD ["bash", "run.sh"]