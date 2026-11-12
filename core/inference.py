import time
import numpy as np
import cv2
import config
from config import logger

try:
    # Importamos el motor ligero de inferencia para Edge AI
    import ai_edge_litert.interpreter as tflite
except ImportError:
    logger.error("No se encontró ai-edge-litert. Asegúrate de instalarlo en tu entorno virtual.")
    raise

class InferenceEngine:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = config.INPUT_SIZE
        self.is_loaded = False

    def load_model(self, model_path):
        """
        Carga un modelo .tflite, reserva memoria para los tensores 
        y extrae los detalles de entrada y salida.
        """
        try:
            logger.info(f"Intentando cargar modelo desde: {model_path}")
            
            # Inicializar el intérprete
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
            
            # Reservar memoria en la RAM para los tensores
            self.interpreter.allocate_tensors()

            # Obtener detalles de la estructura del modelo
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Extraer el tamaño de entrada dinámicamente del modelo
            shape = self.input_details[0]['shape']
            
            # Detectar si el modelo es Channel-First (ej. YOLO) o Channel-Last (ej. MobileNet)
            if len(shape) == 4 and shape[1] == 3:
                # Formato NCHW [Batch, Canales, Alto, Ancho] -> Típico en YOLO
                self.input_shape = (shape[2], shape[3])
                self.is_channel_first = True
            else:
                # Formato NHWC [Batch, Alto, Ancho, Canales] -> Típico en Keras/MobileNet
                self.input_shape = (shape[1], shape[2]) 
                self.is_channel_first = False
            
            self.is_loaded = True
            logger.info(f"Modelo cargado. Input shape: {self.input_shape} | Channel-First: {self.is_channel_first}")
            return True

        except Exception as e:
            self.is_loaded = False
            logger.error(f"Error al cargar el modelo {model_path}: {e}")
            return False

    def preprocess_image(self, frame):
        """
        Acondiciona el fotograma RGB de la cámara para que coincida 
        exactamente con lo que espera el modelo.
        """
        # 1. El frame ya viene en RGB puro, solo redimensionamos al tamaño del modelo
        img_resized = cv2.resize(frame, self.input_shape)
        
        # 2. Convertir a un arreglo numpy tipo float32
        input_data = np.float32(img_resized)
        
        # NOTA CRÍTICA: NO dividimos entre 255.0 aquí porque el modelo 
        # MobileNetV3 de Keras ya integra su propia capa de Rescaling.
        
        # Adaptación automática para modelos como YOLO (Channel-First)
        if getattr(self, 'is_channel_first', False):
            # Cambiar de (Alto, Ancho, Canales) a (Canales, Alto, Ancho)
            input_data = np.transpose(input_data, (2, 0, 1))
            
        # 3. Añadir la dimensión del batch
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data

    def predict(self, frame):
        """
        Ejecuta la inferencia sobre un fotograma y devuelve la clase, 
        la confianza y el tiempo de latencia.
        """
        if not self.is_loaded:
            logger.warning("Intento de predicción sin modelo cargado.")
            return "Ninguno", 0.0, 0.0

        # Preprocesar
        input_data = self.preprocess_image(frame)

        # Inyectar los datos al tensor de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Medir tiempo y ejecutar inferencia
        start_time = time.perf_counter()
        self.interpreter.invoke()
        end_time = time.perf_counter()

        # Extraer los resultados del tensor de salida
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # output_data es un array 2D, ej: [[0.01, 0.95, 0.02, 0.01, 0.01]]
        probabilities = output_data[0] 
        
        # Encontrar el índice con la probabilidad más alta
        max_idx = np.argmax(probabilities)
        confidence = float(probabilities[max_idx])
        
        # Mapear el índice al nombre de la clase (aluminio, carton, etc.)
        predicted_class = config.CLASSES[max_idx]
        
        # Calcular latencia en milisegundos
        inference_time_ms = (end_time - start_time) * 1000

        # Aplicar el umbral de confianza (si es menor, lo clasificamos como 'Desconocido' o esperamos)
        if confidence < config.CONFIDENCE_THRESHOLD:
            predicted_class = "buscando..."
            
        return predicted_class, confidence, inference_time_ms