# Manual de Uso del Tablero  
## Clasificador de Crímenes en San Francisco

## 1. Introducción

El tablero interactivo **Clasificador de Crímenes en San Francisco** permite predecir el tipo de crimen más probable en una ubicación y momento específicos dentro de la ciudad.

La aplicación utiliza un modelo de **Machine Learning basado en XGBoost** entrenado con el dataset histórico de criminalidad de San Francisco. El tablero se comunica con una **API de inferencia desplegada en producción**, la cual procesa las entradas del usuario y retorna la predicción junto con la probabilidad asociada.

El objetivo del tablero es facilitar la exploración de **patrones espacio-temporales delictivos** mediante una interfaz visual e interactiva.

---

## 2. Acceso al tablero

Para utilizar el tablero:

1. Acceda a la URL de la aplicación.
2. Espere a que cargue la interfaz principal.
3. Se mostrará una pantalla dividida en dos secciones principales:
   - **Panel de Entrada de Datos**
   - **Panel de Resultados**

---

## 3. Interfaz del tablero

La interfaz está dividida en dos áreas principales.

### 3.1 Panel de Entrada de Datos

En este panel el usuario puede ingresar los parámetros necesarios para generar una predicción.

Los campos disponibles son:

<img width="1493" height="809" alt="image" src="https://github.com/user-attachments/assets/efa56a8c-fa9b-401b-95ec-0a5c5cd4b515" />


#### Fecha
Permite seleccionar la fecha del evento.

El sistema utiliza esta información para calcular variables temporales como el **día de la semana**, que influyen en la predicción del modelo.

#### Hora
Permite seleccionar la hora del evento.

La hora es una variable relevante para identificar patrones de criminalidad asociados a distintos momentos del día.

#### Mapa interactivo
El tablero incluye un **mapa interactivo de San Francisco** que permite seleccionar una ubicación específica.

Para seleccionar una ubicación:

1. Haga **clic sobre cualquier punto del mapa**.
2. El sistema actualizará automáticamente:
   - **Latitud**
   - **Longitud**
3. El sistema identificará automáticamente el **distrito policial correspondiente** utilizando información geoespacial (GeoJSON).

El distrito detectado se mostrará en el formulario.

#### Coordenadas geográficas
Las coordenadas seleccionadas en el mapa se reflejan en los campos:

- **Latitud**
- **Longitud**

Estos valores representan la ubicación del evento dentro de la ciudad.

#### Distrito policial
El distrito policial se detecta automáticamente a partir de las coordenadas seleccionadas en el mapa.

El distrito es una variable categórica importante para el modelo, ya que los patrones de criminalidad varían significativamente entre diferentes zonas de la ciudad.

#### Botón "Predecir"
Una vez ingresados los parámetros, el usuario puede generar la predicción presionando el botón **Predecir**.

Al presionar el botón:

1. La aplicación envía los datos a la **API de inferencia**.
2. La API ejecuta el modelo de Machine Learning.
3. El resultado se devuelve al tablero.

---

## 4. Panel de Resultados

En el panel derecho de la interfaz se muestran los resultados generados por el modelo.

<img width="745" height="414" alt="image" src="https://github.com/user-attachments/assets/e4ef2c2b-cd77-4080-bf84-71d8e5be5f4c" />


Los elementos mostrados incluyen:

#### Tipo de crimen predicho
Indica la **categoría de crimen más probable** según el modelo.

Ejemplo:

`LARCENY/THEFT`

#### Probabilidad de la predicción
Se muestra la **probabilidad estimada** asociada a la predicción.

Ejemplo:

`Probabilidad: 24.8%`

Este valor representa el nivel de confianza del modelo en la clasificación realizada.

#### Información del modelo
El tablero puede mostrar información adicional como:

- tiempo de inferencia
- origen de la predicción
- modelo utilizado

Esto permite validar que la predicción proviene del **servicio de inferencia en producción**.

---

## 5. Flujo de uso del tablero

El flujo típico de uso es el siguiente:

1. Seleccionar una **fecha**.
2. Seleccionar la **hora del evento**.
3. Hacer **clic en el mapa** para elegir la ubicación.
4. Verificar que el **distrito policial detectado sea correcto**.
5. Presionar el botón **Predecir**.
6. Analizar el resultado generado por el modelo.

---

## 6. Interpretación de resultados

El tablero permite analizar cómo distintas variables influyen en la predicción del modelo.

Al modificar:

- ubicación
- hora
- distrito
- día de la semana

es posible observar cómo cambian las probabilidades de los distintos tipos de crimen.

Esto permite explorar **patrones de criminalidad urbana** desde una perspectiva analítica.

---

## 7. Casos de uso

El tablero puede utilizarse para distintos fines:

### Análisis exploratorio de datos urbanos
Permite analizar cómo diferentes zonas de la ciudad presentan distintos patrones delictivos.

### Simulación de escenarios
El usuario puede modificar los parámetros de entrada para simular posibles situaciones.

### Apoyo académico
El sistema puede utilizarse como herramienta de apoyo para el estudio de:

- Machine Learning aplicado
- análisis de criminalidad urbana
- visualización de datos geoespaciales

---

## 8. Arquitectura del sistema

El flujo de funcionamiento del sistema es el siguiente:

```text
Usuario
   ↓
Tablero interactivo (Streamlit)
   ↓
Solicitud HTTP
   ↓
API de inferencia
   ↓
Modelo XGBoost
   ↓
Respuesta JSON
   ↓
Visualización del resultado
