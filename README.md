# Reconocimiento de Emociones en Perros con CNN

Este proyecto implementa un detector de emociones caninas usando una red neuronal convolucional entrenada con imagenes del dataset **Dog Emotion** de Kaggle. El resultado final se usa en una aplicacion web que permite probar el modelo con imagenes del dataset, archivos locales o la camara del navegador.

La aplicacion no solo predice la emocion: primero usa un detector de objetos preentrenado, **COCO-SSD**, para verificar que la imagen contenga un perro. Si no detecta un perro con suficiente confianza, no ejecuta la prediccion de emocion.

## Emociones que reconoce

El modelo clasifica cada imagen en una de estas 4 clases:

- `angry`: enojado
- `happy`: feliz
- `relaxed`: relajado
- `sad`: triste

## Estructura del proyecto

```text
DogEmotionRecognition-CNN-main/
|-- DogEmotion.ipynb              # Notebook donde se descarga, procesa y entrena el modelo
|-- emotionDog.h5                 # Modelo entrenado guardado en formato Keras/HDF5
|-- model.json                    # Arquitectura del modelo convertida para TensorFlow.js
|-- group1-shard*.bin             # Pesos del modelo para TensorFlow.js
|-- index.html                    # Interfaz web principal
|-- style.css                     # Estilos visuales de la aplicacion
|-- app.js                        # Logica de carga, validacion y prediccion
`-- dog-emotion/
    `-- Dog Emotion/
        |-- labels.csv            # Lista de imagenes y etiquetas
        |-- angry/                # Imagenes de perros enojados
        |-- happy/                # Imagenes de perros felices
        |-- relaxed/              # Imagenes de perros relajados
        `-- sad/                  # Imagenes de perros tristes
```

## Dataset

El conjunto de datos usado es **Dog Emotion**, disponible en Kaggle:

https://www.kaggle.com/datasets/danielshanbalico/dog-emotion

El dataset contiene imagenes de perros organizadas por emocion. En el proyecto tambien se usa `labels.csv`, que relaciona cada archivo con su etiqueta real. La aplicacion web lee ese CSV para escoger imagenes aleatorias del dataset y mostrar la etiqueta original de Kaggle junto a la prediccion del modelo.

## Explicacion del notebook `DogEmotion.ipynb`

El notebook contiene el flujo de entrenamiento del modelo.

### 1. Instalacion e importacion de librerias

Se usan librerias de Python para descargar datos, procesar imagenes, entrenar la red neuronal y visualizar resultados:

- `opendatasets`: descarga el dataset desde Kaggle.
- `os`: recorre carpetas y rutas del sistema.
- `matplotlib.pyplot`: muestra imagenes y graficas.
- `cv2`: lee, convierte y redimensiona imagenes.
- `numpy`: maneja arreglos numericos.
- `tqdm`: muestra barras de progreso.
- `tensorflow` y `keras`: construyen y entrenan la CNN.
- `train_test_split`: separa datos de entrenamiento y prueba.
- `ImageDataGenerator`: aplica aumento de datos.

### 2. Descarga del dataset

El notebook define el enlace:

```python
dataset_link = "https://www.kaggle.com/datasets/danielshanbalico/dog-emotion"
dataset_dir = "./dog-emotion/Dog Emotion"
```

Con `opendatasets` descarga las imagenes y las deja dentro de la carpeta `dog-emotion/Dog Emotion`.

### 3. Carga y preprocesamiento de imagenes

Las imagenes se preparan para que todas tengan el mismo formato de entrada:

- Se leen desde las carpetas de emociones.
- Se convierten a escala de grises.
- Se redimensionan a `150 x 150` pixeles.
- Se normalizan dividiendo los valores de pixeles entre `255`.
- Se guardan en arreglos `X` para imagenes y `y` para etiquetas.

La entrada final del modelo tiene forma:

```text
(150, 150, 1)
```

El `1` indica que la imagen tiene un solo canal porque esta en escala de grises.

### 4. Division de datos

El notebook separa los datos en entrenamiento y prueba:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Esto reserva el 20% de las imagenes para evaluar el modelo y usa el 80% restante para entrenarlo.

### 5. Arquitectura de la CNN

El modelo se define con `tf.keras.Sequential`. La arquitectura principal es:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
```

Funcion de cada parte:

- `Conv2D`: extrae patrones visuales como bordes, texturas y formas.
- `MaxPooling2D`: reduce el tamano de los mapas de caracteristicas y conserva la informacion mas importante.
- `Flatten`: convierte las caracteristicas 2D en un vector.
- `Dense(128)`: aprende combinaciones de caracteristicas para clasificar.
- `Dense(4, softmax)`: devuelve una probabilidad para cada emocion.

### 6. Compilacion y entrenamiento

El modelo se compila con el optimizador `adam` y se entrena durante 80 epocas:

```python
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=80,
    validation_data=(X_test, y_test)
)
```

`ImageDataGenerator` ayuda a crear variaciones de las imagenes de entrenamiento. Esto mejora la capacidad de generalizacion porque el modelo no ve siempre exactamente la misma imagen.

### 7. Guardado y conversion del modelo

Al final, el modelo se guarda como:

```python
model.save('emotionDog.h5')
```

Luego se intenta convertir el modelo a TensorFlow.js con:

```bash
tensorflowjs_converter --input_format keras emotionDog.h5 carpeta_salida
```

En este repositorio ya estan los archivos resultantes que usa el navegador:

- `model.json`
- `group1-shard1of5.bin`
- `group1-shard2of5.bin`
- `group1-shard3of5.bin`
- `group1-shard4of5.bin`
- `group1-shard5of5.bin`

## Explicacion de la aplicacion web

La app web esta formada por `index.html`, `style.css` y `app.js`.

### `index.html`

Este archivo define la estructura visual de la pagina.

Partes importantes:

- Carga Bootstrap y la fuente `Nunito`.
- Carga `style.css` para los estilos propios.
- Muestra un encabezado con imagenes de ejemplo del dataset.
- Crea tres modos de entrada:
  - **Kaggle**: toma una imagen aleatoria del dataset.
  - **Archivo**: permite subir una foto local.
  - **Camara**: usa la camara del navegador.
- Muestra el resultado, la confianza, la etiqueta real de Kaggle y el estado del filtro de perro.
- Carga TensorFlow.js:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js"></script>
```

- Carga COCO-SSD:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js"></script>
```

- Carga la logica principal:

```html
<script src="app.js"></script>
```

### `style.css`

Este archivo define el diseno de la aplicacion:

- Variables de color en `:root`.
- Fondo, tipografia y layout general.
- Panel superior de presentacion.
- Panel de controles.
- Botones de modo.
- Seccion de resultado.
- Vista previa con `canvas` o `video`.
- Diseno responsivo para pantallas pequenas.

Las clases principales son:

- `.app-shell`: contenedor general.
- `.hero-panel`: encabezado visual.
- `.workspace-grid`: divide controles y vista previa.
- `.control-panel`: panel de seleccion de fuente.
- `.preview-panel`: panel donde se muestra imagen o camara.
- `.media-stage`: area donde vive el `canvas` o `video`.
- `.hidden`: oculta elementos cuando se cambia de modo.

### `app.js`

Este archivo contiene toda la logica de la aplicacion.

#### Referencias al DOM

Al inicio se obtienen elementos de la pagina:

```javascript
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
```

Tambien se guardan referencias a botones, etiquetas de estado, paneles y controles.

#### Configuracion del dataset

```javascript
const DATASET_CSV = "dog-emotion/Dog%20Emotion/labels.csv";
const DATASET_BASE = "dog-emotion/Dog%20Emotion";
```

Estas constantes indican donde esta el CSV y donde estan las carpetas de imagenes.

#### Lista de emociones

```javascript
const emociones = [
    { key: "angry", name: "Enojado", color: "#dc2626" },
    { key: "happy", name: "Feliz", color: "#15803d" },
    { key: "relaxed", name: "Relajado", color: "#0891b2" },
    { key: "sad", name: "Triste", color: "#64748b" }
];
```

El orden de esta lista debe coincidir con el orden de salida del modelo. La prediccion devuelve 4 valores y el indice con mayor probabilidad se traduce usando este arreglo.

#### Cambio de modo

La funcion `setMode(mode)` cambia entre:

- `random`: imagen aleatoria del dataset.
- `upload`: imagen subida por el usuario.
- `camera`: camara en vivo.

Tambien detiene la camara cuando corresponde, limpia el `canvas` y actualiza textos de estado.

#### Carga del modelo CNN

```javascript
modelo = await tf.loadLayersModel("./model.json");
```

Esta linea carga el modelo convertido a TensorFlow.js. `model.json` referencia automaticamente los archivos `.bin` donde estan los pesos.

#### Carga del detector de perros

```javascript
detectorPerros = await cocoSsd.load();
```

COCO-SSD es un modelo preentrenado que detecta objetos comunes. En este proyecto se usa para buscar detecciones con `class === "dog"`.

#### Carga del dataset

La funcion `cargarDataset()` lee `labels.csv`, lo separa por lineas y crea un arreglo con objetos como:

```javascript
{
    filename: "imagen.jpg",
    label: "happy"
}
```

Con ese arreglo la app puede escoger imagenes aleatorias y saber la etiqueta real.

#### Imagen aleatoria

`generarImagenAleatoria()`:

1. Elige una imagen segun el filtro de emocion.
2. Muestra la etiqueta real de Kaggle.
3. Dibuja la imagen en el `canvas`.
4. Llama a `analizarSiEsPerro(canvas)`.

#### Imagen subida por el usuario

Cuando cambia el input de archivo:

1. Se lee el archivo con `FileReader`.
2. Se crea un objeto `Image`.
3. Se dibuja en el `canvas`.
4. Se valida si contiene un perro.
5. Si pasa el filtro, se predice la emocion.

#### Camara

`activarCamara()` usa:

```javascript
navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
})
```

Esto solicita permiso al navegador y activa la camara. Luego `predecirLoop()` analiza periodicamente el video.

Para evitar exceso de procesamiento, la validacion de perro en camara se hace aproximadamente cada 900 ms, y el ciclo de prediccion usa `requestAnimationFrame` con una pequena pausa.

#### Validacion de perro

La funcion `validarPerro(elemento)` ejecuta:

```javascript
const detecciones = await detectorPerros.detect(elemento);
```

Luego filtra las detecciones:

```javascript
item.class === "dog"
```

Si no encuentra un perro con confianza minima de `0.45`, la app muestra:

```text
No parece ser un perro
```

y no ejecuta la CNN de emociones.

#### Prediccion de emocion

`ejecutarPrediccion(elemento)` prepara la imagen igual que en el entrenamiento:

```javascript
const tensor = tf.browser.fromPixels(elemento)
    .resizeNearestNeighbor([150, 150])
    .mean(2)
    .expandDims(-1)
    .expandDims(0)
    .toFloat()
    .div(255.0);
```

Paso a paso:

- `fromPixels`: convierte el `canvas` o `video` en tensor.
- `resizeNearestNeighbor([150, 150])`: redimensiona la imagen.
- `mean(2)`: convierte de RGB a escala de grises.
- `expandDims(-1)`: agrega el canal de color, quedando `(150, 150, 1)`.
- `expandDims(0)`: agrega la dimension de lote, quedando `(1, 150, 150, 1)`.
- `toFloat()`: convierte los valores a numeros flotantes.
- `div(255.0)`: normaliza pixeles entre 0 y 1.

Despues ejecuta:

```javascript
const prediccion = modelo.predict(tensor);
```

El resultado es un arreglo con 4 probabilidades. La app toma la mayor, busca su indice y muestra la emocion correspondiente.

`tf.tidy()` se usa para liberar memoria de tensores temporales y evitar que el navegador consuma memoria innecesaria con cada prediccion.

## Flujo completo de prediccion

```text
Imagen del dataset, archivo o camara
        v
Mostrar en canvas o video
        v
COCO-SSD verifica si hay perro
        v
Si no hay perro: se detiene
        v
Si hay perro: preparar tensor 150x150 en escala de grises
        v
CNN predice angry, happy, relaxed o sad
        v
Mostrar emocion y confianza
```

## Como ejecutar la aplicacion

Abre una terminal dentro de la carpeta donde estan `index.html`, `app.js`, `style.css` y `model.json`.

Ejecuta:

```bash
python -m http.server 8080
```

Luego abre en el navegador:

```text
http://localhost:8080
```

Es importante usar un servidor local porque el navegador puede bloquear la lectura de `labels.csv`, `model.json` o los archivos `.bin` si se abre `index.html` directamente desde el explorador de archivos.

## Requisitos principales

Para entrenar o modificar el modelo desde el notebook:

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- opendatasets
- tensorflowjs, si se quiere convertir el modelo para navegador

Para ejecutar la app web:

- Navegador moderno
- Conexion a internet para cargar TensorFlow.js, COCO-SSD, Bootstrap y Google Fonts desde CDN
- Servidor local, por ejemplo `python -m http.server 8080`
- Permiso de camara si se usa el modo **Camara**

## Archivos del modelo web

La app no carga `emotionDog.h5` directamente. Ese archivo es para Python/Keras. En el navegador se cargan:

- `model.json`
- `group1-shard1of5.bin`
- `group1-shard2of5.bin`
- `group1-shard3of5.bin`
- `group1-shard4of5.bin`
- `group1-shard5of5.bin`

Por eso, si se vuelve a entrenar el modelo, hay que convertirlo otra vez a TensorFlow.js y reemplazar estos archivos.

## Limitaciones

- El modelo solo conoce las 4 emociones del dataset.
- La prediccion depende mucho de la calidad, angulo e iluminacion de la imagen.
- El filtro de perro usa COCO-SSD, no una regla perfecta. Puede fallar si el perro esta muy pequeno, cortado o en una posicion rara.
- La emocion real de un perro no siempre puede determinarse con exactitud solo por una foto; este proyecto es una clasificacion visual basada en patrones del dataset.
- La aplicacion necesita internet para cargar librerias externas desde CDN.

## Objetivo final

El objetivo es demostrar un flujo completo de vision por computadora:

1. Preparar un dataset de imagenes.
2. Entrenar una CNN en Python.
3. Guardar y convertir el modelo.
4. Ejecutar inferencia en el navegador con TensorFlow.js.
5. Combinar el clasificador con un filtro visual de perro para evitar predicciones sobre imagenes incorrectas.
