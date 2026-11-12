import cv2
import numpy as np
from config import logger

try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False
    logger.error("No se encontró Picamera2. El sistema base no lo tiene instalado.")

class Camera:
    def __init__(self, camera_index=0):
        """
        Inicializa el objeto de la cámara usando Picamera2 (Nativo de RPi 5 / Bookworm).
        """
        self.picam2 = None
        self.is_running = False

    def start(self):
        """Abre el flujo de video de la cámara."""
        if not self.is_running:
            logger.info("Iniciando cámara nativa (Picamera2) para Raspberry Pi 5...")
            
            if not PICAM2_AVAILABLE:
                logger.error("Error: Picamera2 no está disponible.")
                return False
            
            try:
                self.picam2 = Picamera2()
                
                # Configurar explícitamente el formato a RGB888 para evitar el "Efecto Pitufo"
                vid_config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
                self.picam2.configure(vid_config)
                
                self.picam2.start()
                self.is_running = True
                logger.info("Cámara iniciada correctamente con Picamera2.")
                return True
                
            except Exception as e:
                logger.error(f"Error de hardware al abrir la cámara: {e}")
                return False
        return True

    def get_frame(self):
        """
        Captura y devuelve un solo fotograma directo a la memoria (RAM).
        Retorna un NumPy array en formato RGB Nativo (3 canales).
        """
        if self.is_running and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                
                # Si el sistema operativo insiste en enviar 4 canales (RGBX)
                # simplemente recortamos el cuarto canal de relleno sin alterar los colores.
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                return np.ascontiguousarray(frame)
            except Exception as e:
                logger.warning(f"No se pudo leer el fotograma. Detalle: {e}")
        return None

    def stop(self):
        """Detiene la cámara y libera los recursos del hardware."""
        if self.is_running:
            logger.info("Deteniendo la cámara y liberando recursos...")
            self.is_running = False
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
            logger.info("Cámara detenida de forma segura.")