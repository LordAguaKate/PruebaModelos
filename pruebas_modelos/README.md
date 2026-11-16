# Pruebas de los modelos

Esta carpeta tiene todo lo necesario para repetir las pruebas de los modelos de clasificación de
residuos: los scripts, una copia de los modelos y los resultados que ya se obtuvieron.

```text
pruebas_modelos/
├── raspberry/     scripts que se ejecutan en la Raspberry Pi
├── local/         scripts que se ejecutan en la computadora
├── reportes/      scripts que generan gráficas y PDF a partir de los resultados
├── modelos/       copia de los modelos probados
└── resultados/    reportes, datos crudos y gráficas de las pruebas ya hechas
```

Todos los comandos de esta guía se ejecutan **desde la raíz del repositorio** (`PruebaModelos/`).

---

## 1. Lo que se necesita

### 1.1 Conjunto de imágenes de prueba (obligatorio)

El conjunto de prueba **no viene en el repositorio** por su tamaño. Es el test set de
`waste_classification` (4534 imágenes), con **una subcarpeta por clase**:

```text
test_set/
├── aluminio/
├── carton/
├── otros/
├── plastico/
└── vidrio/
```

Dónde ponerlo depende del script:

| Script | Dónde busca las imágenes |
|---|---|
| `raspberry/benchmark_visual.py` | `pruebas_modelos/raspberry/test_set/`, o la carpeta que se pase con `--images-dir` |
| `raspberry/Benchmark_rpi.py` | La carpeta que se pase con `--images-dir` (obligatorio) |
| `local/*.py` | `data/raw/waste_classification/test/` en la raíz del repositorio (ruta fija) |
| `reportes/graficar_benchmark_rpi.py` | La carpeta que se pase con `--test-dir` (solo para los mosaicos de imágenes) |

### 1.2 Modelos

Ya están en `modelos/` y los scripts los toman de ahí; no hay que mover nada.

| Modelo | Clave | Archivos |
|---|---|---|
| MobileNetV3-Large | `mobilenet` | `MobileNetV3/modelo_residuos_rpi_V3_05.tflite` y `.keras` |
| EfficientNetB0 | `efficientnet` | `EfficientNetB0/modelo_residuos_EfficientNetB0_v1.tflite` y `.keras` |
| ShuffleNetV2 (ImageNet) | `shufflenet` | `ShuffleNetV2/modelo_residuos_shufflenetV2_imagenet.tflite` y `.keras` |
| SqueezeNet v1.1 (ImageNet) | `squeezenet` | `SqueezeNet/modelo_residuos_SqueezeNet_imagenet.tflite` y `.keras` |
| YOLO11s-cls | `yolo` | `YOLO/modelo_residuos_YOLO_v1.tflite` y `.pt` |
| MobileNetV2 (cámara) | — | `MobileNetV2/modelo_residuos_rpi.tflite` |

La clave es la que se usa con `--modelos` y `--recomendado`.

### 1.3 Dependencias de Python

Se recomienda un entorno virtual (`python3 -m venv .venv && source .venv/bin/activate`).

| Dónde | Para qué | Instalación |
|---|---|---|
| Raspberry Pi | Benchmarks (`raspberry/`) | `pip install ai-edge-litert numpy pillow psutil rich` |
| Raspberry Pi | Prueba con cámara | además `pip install opencv-python` |
| Computadora | Evaluación de `.keras` y `.tflite` (`local/`) | `pip install tensorflow numpy scikit-learn tabulate` |
| Computadora | Gráficas (`reportes/graficar_benchmark_rpi.py`) | `pip install matplotlib numpy pillow` y el comando `rsvg-convert` (paquete `librsvg2-bin` en Debian/Ubuntu, `librsvg` en Arch) |
| Computadora | PDF (`reportes/generar_pdf_reporte.py`) | `pip install weasyprint markdown` |

Notas:

- Los scripts de la Raspberry buscan el intérprete en este orden: `tflite_runtime`, `ai_edge_litert`
  y `tensorflow.lite`. Basta con tener uno; por eso también funcionan en la computadora si tiene
  TensorFlow (`pip install tensorflow psutil rich pillow`).
- `rsvg-convert` solo hace falta para convertir las capturas de pantalla SVG a PNG; sin él, las
  gráficas se generan igual y las capturas se omiten.

---

## 2. Pruebas en la Raspberry Pi

### 2.1 Preparar la Raspberry

Copiar a la Raspberry la carpeta con el script, los modelos `.tflite` y el conjunto de prueba. Por
ejemplo, desde la computadora (`raspberry` es el nombre o la IP de la Raspberry):

```bash
ssh raspberry 'mkdir -p ~/ecosort_benchmark/modelos'
scp pruebas_modelos/raspberry/benchmark_visual.py raspberry:ecosort_benchmark/
scp pruebas_modelos/modelos/*/*.tflite raspberry:ecosort_benchmark/modelos/
scp -r ruta/al/test_set raspberry:ecosort_benchmark/test_set
```

En la Raspberry queda así, y el script encuentra `modelos/` y `test_set/` solo:

```text
~/ecosort_benchmark/
├── benchmark_visual.py
├── modelos/      los .tflite
└── test_set/     una subcarpeta por clase
```

### 2.2 Benchmark de los 5 modelos (`benchmark_visual.py`)

Prueba los cinco modelos `.tflite` sobre las mismas imágenes y muestra el proceso en la terminal. Por
cada modelo mide:

- latencia y precisión, con una inferencia cronometrada por imagen;
- estabilidad, con inferencia continua mientras mide CPU, RAM y temperatura.

Al final muestra una tabla comparativa y las matrices de confusión.

```bash
cd ~/ecosort_benchmark

# Corrida oficial: todas las imágenes, 5 min de estabilidad por modelo (~40 min)
python3 benchmark_visual.py --salida-dir resultados_5modelos_1hilo

# Corrida con 4 hilos (~22 min)
python3 benchmark_visual.py --num-threads 4 --duracion-estabilidad 120 \
        --salida-dir resultados_5modelos_4hilos

# Demostración corta (por ejemplo, para grabar un video)
python3 benchmark_visual.py --max-imagenes 200 --duracion-estabilidad 30 --enfriamiento 5
```

Opciones principales:

| Opción | Default | Qué hace |
|---|---|---|
| `--models-dir` | `modelos/` junto al script | Carpeta con los `.tflite` (los busca también en subcarpetas) |
| `--images-dir` | `test_set/` junto al script | Conjunto de prueba |
| `--modelos` | los 5 | Modelos a probar, por clave: `mobilenet efficientnet shufflenet squeezenet yolo` |
| `--max-imagenes` | `0` (todas) | Límite de imágenes |
| `--n-warmup` | `10` | Inferencias de calentamiento |
| `--duracion-estabilidad` | `300` | Segundos de inferencia continua por modelo |
| `--enfriamiento` | `60` | Segundos de pausa antes de cada modelo |
| `--num-threads` | el del runtime | Hilos del intérprete |
| `--salida-dir` | `resultados_<fecha>` | Carpeta de salida |
| `--sin-capturas` | — | No guardar las capturas SVG |

Salida: `<salida-dir>/resultados.json` (todas las cifras) y `<salida-dir>/capturas/*.svg` (capturas de
cada pantalla).

También se puede correr desde la raíz del repositorio en la computadora, con TensorFlow instalado,
pasando el conjunto de prueba:

```bash
python3 pruebas_modelos/raspberry/benchmark_visual.py --images-dir ruta/al/test_set \
        --max-imagenes 200 --duracion-estabilidad 30 --enfriamiento 5
```

### 2.3 Benchmark de un solo modelo (`Benchmark_rpi.py`)

Versión anterior, más simple: latencia, CPU/RAM y estabilidad de un `.tflite`.

```bash
python3 pruebas_modelos/raspberry/Benchmark_rpi.py \
        --model pruebas_modelos/modelos/MobileNetV3/modelo_residuos_rpi_V3_05.tflite \
        --images-dir ruta/al/test_set --max-imagenes 200 --duracion-estabilidad 300
```

`--model` recibe la ruta al archivo. Salida: `benchmark_resultados.json` en la carpeta actual.

### 2.4 Prueba con cámara (`test_camara_realtime.py`)

Clasifica en tiempo real lo que ve la cámara 0 (USB o la que OpenCV abra como `VideoCapture(0)`), con
el modelo `modelos/MobileNetV2/modelo_residuos_rpi.tflite`. Necesita pantalla. Se cierra con `q`.

```bash
python3 pruebas_modelos/raspberry/test_camara_realtime.py
```

---

## 3. Pruebas en la computadora

Los tres scripts leen el conjunto de prueba de `data/raw/waste_classification/test/`, así que primero
hay que copiarlo ahí:

```bash
mkdir -p data/raw/waste_classification
cp -r ruta/al/test_set data/raw/waste_classification/test
```

### 3.1 Comparar los modelos `.keras` (`comparar_modelos_inferencia.py`)

Carga cada `.keras`, predice todo el conjunto de prueba y muestra una tabla comparativa:
accuracy, precision, recall, F1, tamaño y latencia. Incluye MobileNetV3, EfficientNetB0 y ShuffleNetV2.
Los modelos que no estén en `modelos/` se omiten.

```bash
python3 pruebas_modelos/local/comparar_modelos_inferencia.py
```

### 3.2 Precisión del modelo cuantizado (`Evaluate_tflite.py`)

Evalúa `MobileNetV3/modelo_residuos_rpi_V3_05.tflite` sobre el conjunto de prueba. Lo compara con
los valores del `.keras` para ver cuánto cambia la precisión al cuantizar.

```bash
python3 pruebas_modelos/local/Evaluate_tflite.py
```

### 3.3 Tabla desde los notebooks (`comparar_modelos.py`)

No ejecuta modelos. Arma la tabla con los resultados guardados en los notebooks de evaluación,
que están en `notebooks/` del repositorio **EntrenamientoIA**. Por eso solo funciona dentro de ese
repositorio y no en este.

---

## 4. Generar gráficas y PDF

### 4.1 Gráficas (`graficar_benchmark_rpi.py`)

Se corre en la computadora con el `resultados.json` que generó la Raspberry. Ejemplo con los resultados
del benchmark de 5 modelos que ya están en `resultados/`:

```bash
python3 pruebas_modelos/reportes/graficar_benchmark_rpi.py \
    --resultados        pruebas_modelos/resultados/reports/benchmark_rpi5_5modelos/resultados_rpi5_5modelos_1hilo.json \
    --resultados-hilos  pruebas_modelos/resultados/reports/benchmark_rpi5_5modelos/resultados_rpi5_5modelos_4hilos.json \
    --capturas          pruebas_modelos/resultados/reports/benchmark_rpi5_5modelos/capturas_svg \
    --seleccion         pruebas_modelos/resultados/reports/benchmark_rpi5_5modelos/seleccion_mosaicos.json \
    --test-dir          ruta/al/test_set \
    --salida            pruebas_modelos/resultados/reports/figures/raspberry5_5modelos
```

Para resultados nuevos, primero se copian de la Raspberry:

```bash
scp raspberry:ecosort_benchmark/resultados_5modelos_1hilo/resultados.json nuevo_1hilo.json
scp 'raspberry:ecosort_benchmark/resultados_5modelos_1hilo/capturas/*.svg' capturas_svg/
```

### 4.2 PDF del reporte (`generar_pdf_reporte.py`)

Convierte un reporte `.md` de `resultados/docs/` a PDF, con portada, índice y figuras numeradas:

```bash
python3 pruebas_modelos/reportes/generar_pdf_reporte.py \
    --md     pruebas_modelos/resultados/docs/reporte_benchmark_5_modelos_rpi5_mobilenet.md \
    --salida pruebas_modelos/resultados/docs/reporte_benchmark_5_modelos_rpi5_mobilenet.pdf \
    --datos  pruebas_modelos/resultados/reports/benchmark_rpi5_5modelos/resultados_rpi5_5modelos_1hilo.json \
    --subtitulo "Comparación de MobileNetV3-Large, EfficientNetB0, ShuffleNetV2, SqueezeNet v1.1 y YOLO11s-cls en la Raspberry Pi 5" \
    --recomendado mobilenet
```

`--recomendado` es la clave del modelo que lleva el sello de recomendado (`yolo`, `mobilenet`, …).
Con una clave que no exista, como `ninguno`, no se pone sello.

---

## 5. Resultados ya obtenidos (`resultados/`)

Es una copia de los resultados del repositorio EntrenamientoIA. Mantiene la misma estructura, así que
las imágenes de los reportes `.md` se ven.

| Carpeta | Contenido |
|---|---|
| `resultados/docs/` | Reportes en `.md` y `.pdf`: 5 modelos en la Pi 5 (versiones con MobileNet, con YOLO y sin recomendación), 3 modelos en la laptop y la Pi 5, y Pi 4 |
| `resultados/reports/benchmark_rpi5_5modelos/` | Datos crudos (JSON), logs y capturas del benchmark de 5 modelos en la Pi 5 |
| `resultados/reports/benchmark_rpi/` | Datos y logs del benchmark de 3 modelos en la Pi 5 |
| `resultados/reports/benchmark_laptop/` | Datos del benchmark en la laptop |
| `resultados/reports/figures/` | Gráficas de los benchmarks |

Las figuras del reporte de la Pi 4 no están incluidas porque no existen en el repositorio original.
