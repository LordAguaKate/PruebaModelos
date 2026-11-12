import logging
from pathlib import Path

# --- RUTAS PRINCIPALES ---
# BASE_DIR es la ruta absoluta a la carpeta "PruebaModelos"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
LOGS_DIR = BASE_DIR / "logs"

# Crear la carpeta de logs automáticamente si no existe
LOGS_DIR.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DE LOGGING ---
LOG_FILE = LOGS_DIR / "ecosort_app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'), # Guarda en archivo
        logging.StreamHandler()                          # Muestra en terminal
    ]
)
# Instancia principal del logger para usar en todo el proyecto
logger = logging.getLogger("RENOVA")

# --- MODELOS DISPONIBLES ---
# Diccionario para poblar el menú desplegable de CustomTkinter y mapear a la ruta exacta
AVAILABLE_MODELS = {
    "MobileNetV3 (Recomendado)": MODELS_DIR / "MobileNetV3" / "modelo_residuos_rpi_V3_05.tflite",
    "EfficientNetB0": MODELS_DIR / "EfficientNetB0" / "modelo_residuos_EfficientNetB0_v1.tflite",
    "MobileNetV2": MODELS_DIR / "MobileNetV2" / "modelo_residuos_rpi_03.tflite",
    "ShuffleNetV2": MODELS_DIR / "ShuffleNetV2" / "modelo_residuos_shufflenetV2_imagenet.tflite",
    "SqueezeNet": MODELS_DIR / "SqueezeNet" / "modelo_residuos_SqueezeNet_imagenet.tflite",
    "YOLO v1": MODELS_DIR / "YOLO" / "modelo_residuos_YOLO_v1.tflite"
}

# --- PARÁMETROS DEL MODELO Y VISIÓN ---
CLASSES = ["aluminio", "carton", "otros", "plastico", "vidrio"]
CONFIDENCE_THRESHOLD = 0.85
INPUT_SIZE = (224, 224) # Tamaño de entrada estándar. (Ajustaremos si YOLO o EfficientNet piden otro)

# --- CONFIGURACIÓN DE LA INTERFAZ (CustomTkinter) ---
UI_THEME = "Dark"      # Opciones: "System", "Dark", "Light"
UI_COLOR = "green"     # Opciones: "blue", "green", "dark-blue"
WINDOW_TITLE = "Edge AI - EcoSort"
WINDOW_SIZE = "1024x600" # Resolución ideal para pantallas de Raspberry Pi