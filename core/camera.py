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
        self.swap_rb = False

    def start(self):
        """Abre el flujo de video de la cámara."""
        if not self.is_running:
            logger.info("Iniciando cámara nativa (Picamera2) para Raspberry Pi 5...")
            
            if not PICAM2_AVAILABLE:
                logger.error("Error: Picamera2 no está disponible.")
                return False
            
            try:
                self.picam2 = Picamera2()
                
                # OJO: los nombres de formato de libcamera están "al revés" (little-endian):
                # "RGB888" entrega los píxeles como [B, G, R] (imagen azulada / "Efecto Pitufo"),
                # mientras que "BGR888" entrega [R, G, B], que es lo que esperan PIL y el modelo.
                vid_config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
                self.picam2.configure(vid_config)

                # Verificar el formato que realmente aplicó libcamera (puede ajustarlo).
                # "RGB888" y "XRGB8888" entregan [B, G, R(, X)] y hay que invertir los canales.
                actual_format = self.picam2.camera_configuration()["main"]["format"]
                self.swap_rb = actual_format in ("RGB888", "XRGB8888")
                logger.info(f"Formato de cámara aplicado: {actual_format} | Invertir R/B: {self.swap_rb}")

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
                
                # Si llegan 4 canales, recortamos el cuarto canal de relleno (X).
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                # Si el formato entrega [B, G, R], lo pasamos a [R, G, B].
                if self.swap_rb:
                    frame = frame[:, :, ::-1]

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