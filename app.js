const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const resultadoDiv = document.getElementById("resultado");
const sourceLabel = document.getElementById("source-label");
const realLabel = document.getElementById("real-label");
const confidenceLabel = document.getElementById("confidence-label");
const dogLabel = document.getElementById("dog-label");
const modelStatus = document.getElementById("model-status");
const detectorStatus = document.getElementById("detector-status");
const previewTitle = document.getElementById("preview-title");

const btnRandom = document.getElementById("btn-random");
const btnUpload = document.getElementById("btn-upload");
const btnCamera = document.getElementById("btn-camera");
const btnNewRandom = document.getElementById("btn-new-random");
const btnStartCamera = document.getElementById("btn-start-camera");
const btnStopCamera = document.getElementById("btn-stop-camera");
const inputImagen = document.getElementById("input-imagen");
const emotionFilter = document.getElementById("emotion-filter");

const panels = {
    random: document.getElementById("panel-random"),
    upload: document.getElementById("panel-upload"),
    camera: document.getElementById("panel-camera")
};

const modeButtons = {
    random: btnRandom,
    upload: btnUpload,
    camera: btnCamera
};

const DATASET_CSV = "dog-emotion/Dog%20Emotion/labels.csv";
const DATASET_BASE = "dog-emotion/Dog%20Emotion";

let modelo = null;
let detectorPerros = null;
let dataset = [];
let modoCamara = false;
let streamActual = null;
let camaraPerroDetectado = false;
let ultimaRevisionCamara = 0;

const emociones = [
    { key: "angry", name: "Enojado", color: "#dc2626" },
    { key: "happy", name: "Feliz", color: "#15803d" },
    { key: "relaxed", name: "Relajado", color: "#0891b2" },
    { key: "sad", name: "Triste", color: "#64748b" }
];

const emotionByKey = Object.fromEntries(emociones.map((emotion) => [emotion.key, emotion]));

function setStatus(text, type = "") {
    modelStatus.textContent = text;
    modelStatus.className = `status-pill ${type}`.trim();
}

function setDetectorStatus(text, type = "") {
    detectorStatus.textContent = text;
    detectorStatus.className = `status-pill ${type}`.trim();
}

function setResult(text, color = "#ffffff") {
    resultadoDiv.textContent = text;
    resultadoDiv.style.color = color;
}

function setDogLabel(text, color = "#bfdbfe") {
    dogLabel.textContent = text;
    dogLabel.style.color = color;
}

function setMode(mode) {
    detenerCamara();
    modoCamara = false;

    Object.entries(panels).forEach(([key, panel]) => {
        panel.classList.toggle("hidden", key !== mode);
    });

    Object.entries(modeButtons).forEach(([key, button]) => {
        button.classList.toggle("active", key === mode);
    });

    video.classList.add("hidden");
    canvas.classList.remove("hidden");

    if (mode === "random") {
        previewTitle.textContent = "Imagen del dataset";
        sourceLabel.textContent = "Fuente: Kaggle Dog Emotion";
    }

    if (mode === "upload") {
        previewTitle.textContent = "Imagen seleccionada";
        sourceLabel.textContent = "Fuente: archivo local";
        setDogLabel("Filtro perro: esperando imagen");
        realLabel.textContent = "Etiqueta Kaggle: --";
        confidenceLabel.textContent = "Confianza: --";
        setResult("Esperando imagen");
        clearCanvas();
    }

    if (mode === "camera") {
        previewTitle.textContent = "Camara en vivo";
        sourceLabel.textContent = "Fuente: camara";
        setDogLabel("Filtro perro: esperando camara");
        realLabel.textContent = "Etiqueta Kaggle: --";
        confidenceLabel.textContent = "Confianza: --";
        setResult("Camara lista para iniciar");
        clearCanvas();
    }
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawImageContain(img) {
    clearCanvas();
    const ratio = Math.min(canvas.width / img.width, canvas.height / img.height);
    const newWidth = img.width * ratio;
    const newHeight = img.height * ratio;
    const x = (canvas.width - newWidth) / 2;
    const y = (canvas.height - newHeight) / 2;
    ctx.drawImage(img, x, y, newWidth, newHeight);
}

async function cargarModelo() {
    try {
        setStatus("CNN cargando");
        setResult("Cargando modelo...");
        modelo = await tf.loadLayersModel("./model.json");
        setStatus("CNN lista", "ready");
        setResult("Modelo listo", "#86efac");
    } catch (error) {
        console.error(error);
        setStatus("Error CNN", "error");
        setResult("Error cargando modelo", "#fecaca");
    }
}

async function cargarDetectorPerros() {
    try {
        setDetectorStatus("Filtro cargando");
        if (!window.cocoSsd) {
            throw new Error("coco-ssd no esta disponible");
        }
        detectorPerros = await cocoSsd.load();
        setDetectorStatus("Filtro listo", "ready");
        setDogLabel("Filtro perro: listo");
    } catch (error) {
        console.error(error);
        setDetectorStatus("Filtro sin cargar", "error");
        setDogLabel("Filtro perro: no disponible", "#fecaca");
    }
}

async function cargarDataset() {
    try {
        const response = await fetch(DATASET_CSV);
        if (!response.ok) {
            throw new Error(`No se pudo leer labels.csv: ${response.status}`);
        }

        const csv = await response.text();
        dataset = csv
            .trim()
            .split(/\r?\n/)
            .slice(1)
            .map((line) => {
                const columns = line.split(",");
                return {
                    filename: columns[1],
                    label: columns[2]
                };
            })
            .filter((item) => item.filename && emotionByKey[item.label]);

        sourceLabel.textContent = `Fuente: Kaggle Dog Emotion (${dataset.length} imagenes)`;

        if (dataset.length > 0) {
            await generarImagenAleatoria();
        }
    } catch (error) {
        console.error(error);
        sourceLabel.textContent = "Fuente: dataset no disponible";
        setResult("Abre la app con servidor local", "#fcd34d");
    }
}

function elegirImagenAleatoria() {
    const selectedEmotion = emotionFilter.value;
    const candidates = selectedEmotion === "all"
        ? dataset
        : dataset.filter((item) => item.label === selectedEmotion);

    if (candidates.length === 0) {
        return null;
    }

    return candidates[Math.floor(Math.random() * candidates.length)];
}

async function generarImagenAleatoria() {
    if (dataset.length === 0) {
        setResult("Dataset no cargado", "#fcd34d");
        return;
    }

    const item = elegirImagenAleatoria();
    if (!item) {
        setResult("No hay imagenes para esa emocion", "#fcd34d");
        return;
    }

    detenerCamara();
    video.classList.add("hidden");
    canvas.classList.remove("hidden");

    const realEmotion = emotionByKey[item.label];
    setDogLabel("Filtro perro: revisando...");
    realLabel.textContent = `Etiqueta Kaggle: ${realEmotion.name}`;
    confidenceLabel.textContent = "Confianza: calculando...";
    sourceLabel.textContent = `Fuente: ${item.filename}`;
    setResult("Procesando imagen...", "#fcd34d");

    const img = new Image();
    img.onload = async () => {
        drawImageContain(img);
        await analizarSiEsPerro(canvas);
    };
    img.onerror = () => {
        setResult("No se pudo abrir la imagen", "#fecaca");
    };
    img.src = `${DATASET_BASE}/${item.label}/${encodeURIComponent(item.filename)}`;
}

async function activarCamara() {
    try {
        streamActual = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });

        modoCamara = true;
        video.srcObject = streamActual;
        video.classList.remove("hidden");
        canvas.classList.add("hidden");
        setDogLabel("Filtro perro: revisando...");
        setResult("Analizando camara...", "#bfdbfe");

        video.onloadeddata = () => {
            predecirLoop();
        };
    } catch (error) {
        console.error(error);
        setResult("No se pudo abrir la camara", "#fecaca");
    }
}

function detenerCamara() {
    if (streamActual) {
        streamActual.getTracks().forEach((track) => track.stop());
        video.srcObject = null;
        streamActual = null;
    }
    modoCamara = false;
}

inputImagen.addEventListener("change", (event) => {
    const archivo = event.target.files[0];
    if (!archivo) return;

    detenerCamara();
    setResult("Procesando imagen...", "#fcd34d");
    setDogLabel("Filtro perro: revisando...");
    realLabel.textContent = "Etiqueta Kaggle: --";
    confidenceLabel.textContent = "Confianza: calculando...";
    sourceLabel.textContent = `Fuente: ${archivo.name}`;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = async () => {
            drawImageContain(img);
            await analizarSiEsPerro(canvas);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(archivo);
});

async function predecirLoop() {
    if (!modoCamara) return;

    if (video.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA) {
        const ahora = Date.now();
        if (ahora - ultimaRevisionCamara > 900) {
            camaraPerroDetectado = await validarPerro(video);
            ultimaRevisionCamara = ahora;
        }

        if (camaraPerroDetectado) {
            await ejecutarPrediccion(video);
        }
    }

    setTimeout(() => requestAnimationFrame(predecirLoop), 180);
}

async function analizarSiEsPerro(elemento) {
    const hayPerro = await validarPerro(elemento);
    if (!hayPerro) {
        return null;
    }
    return ejecutarPrediccion(elemento);
}

async function validarPerro(elemento) {
    if (!detectorPerros) {
        setDogLabel("Filtro perro: no disponible", "#fecaca");
        setResult("No puedo validar si es perro", "#fecaca");
        confidenceLabel.textContent = "Confianza: --";
        return false;
    }

    try {
        const detecciones = await detectorPerros.detect(elemento);
        const mejorPerro = detecciones
            .filter((item) => item.class === "dog")
            .sort((a, b) => b.score - a.score)[0];

        if (!mejorPerro || mejorPerro.score < 0.45) {
            setDogLabel("Filtro perro: no detectado", "#fecaca");
            setResult("No parece ser un perro", "#fecaca");
            confidenceLabel.textContent = "Confianza: --";
            return false;
        }

        setDogLabel(`Filtro perro: detectado ${(mejorPerro.score * 100).toFixed(1)}%`, "#bbf7d0");
        return true;
    } catch (error) {
        console.error(error);
        setDogLabel("Filtro perro: error", "#fecaca");
        setResult("Error validando perro", "#fecaca");
        confidenceLabel.textContent = "Confianza: --";
        return false;
    }
}

async function ejecutarPrediccion(elemento) {
    if (!modelo) {
        setResult("Modelo aun cargando", "#fcd34d");
        return null;
    }

    try {
        const scores = tf.tidy(() => {
            const tensor = tf.browser.fromPixels(elemento)
                .resizeNearestNeighbor([150, 150])
                .mean(2)
                .expandDims(-1)
                .expandDims(0)
                .toFloat()
                .div(255.0);

            const prediccion = modelo.predict(tensor);
            return Array.from(prediccion.dataSync());
        });

        const maxScore = Math.max(...scores);
        const indice = scores.indexOf(maxScore);
        const emotion = emociones[indice] || { name: "Desconocido", color: "#ffffff" };

        setResult(emotion.name, emotion.color);
        confidenceLabel.textContent = `Confianza: ${(maxScore * 100).toFixed(1)}%`;

        return { emotion, confidence: maxScore };
    } catch (error) {
        console.error(error);
        setResult("Error en prediccion", "#fecaca");
        return null;
    }
}

btnRandom.addEventListener("click", () => setMode("random"));
btnUpload.addEventListener("click", () => setMode("upload"));
btnCamera.addEventListener("click", () => setMode("camera"));
btnNewRandom.addEventListener("click", generarImagenAleatoria);
emotionFilter.addEventListener("change", generarImagenAleatoria);
btnStartCamera.addEventListener("click", activarCamara);
btnStopCamera.addEventListener("click", () => {
    detenerCamara();
    video.classList.add("hidden");
    canvas.classList.remove("hidden");
    setDogLabel("Filtro perro: camara detenida");
    setResult("Camara detenida", "#ffffff");
});

async function iniciarApp() {
    setMode("random");
    await Promise.all([cargarModelo(), cargarDetectorPerros()]);
    await cargarDataset();
}

iniciarApp();
