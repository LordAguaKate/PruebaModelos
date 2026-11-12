import cv2
import time
from PIL import Image
import config
from config import logger

# Importar los módulos que ya construimos
from ui.main_window import MainWindow
from core.camera import Camera
from core.inference import InferenceEngine

class EcoSortApp:
    def __init__(self):
        logger.info("Iniciando aplicación EcoSort...")
        
        self.ui = MainWindow()
        self.camera = Camera(camera_index=0)
        self.engine = InferenceEngine()

        self.ui.on_model_change_callback = self.change_model
        self.ui.on_toggle_camera_callback = self.toggle_camera
        self.ui.protocol("WM_DELETE_WINDOW", self.on_close)

        # Variables para Métricas HUD
        self.current_model_name = ""
        self.prev_frame_time = time.time()
        self.fps_smoothed = 0.0

        default_model_name = list(config.AVAILABLE_MODELS.keys())[0]
        self.change_model(default_model_name)

    def change_model(self, model_name):
        model_path = config.AVAILABLE_MODELS.get(model_name)
        if model_path:
            logger.info(f"Cambiando al modelo: {model_name}")
            self.current_model_name = model_name.split()[0] # Toma solo el primer nombre (ej. "MobileNetV3")
            success = self.engine.load_model(model_path)
            if success:
                self.ui.status_label.configure(text=f"Modelo cargado: {self.current_model_name}", text_color="#00FF00")
            else:
                self.ui.status_label.configure(text=f"Error cargando modelo", text_color="red")

    def toggle_camera(self):
        if not self.camera.is_running:
            success = self.camera.start()
            if success:
                self.ui.set_camera_button_state(is_running=True)
                self.ui.status_label.configure(text="Cámara encendida", text_color="#00FF00")
                self.prev_frame_time = time.time() # Reiniciar contador FPS
                self.process_video_stream()
        else:
            self.camera.stop()
            self.ui.set_camera_button_state(is_running=False)
            self.ui.status_label.configure(text="Cámara detenida", text_color="gray")
            self.ui.video_label.configure(image="", text="Cámara Detenida")

    def draw_hud(self, frame, fps, latency):
        """Dibuja métricas de rendimiento (HUD) sobre el fotograma con OpenCV."""
        # Configuración de fuente
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Sombra y Texto para FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (16, 31), font, 0.7, (0,0,0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (15, 30), font, 0.7, (0,255,0), 2)
        
        # Sombra y Texto para Nombre del Modelo
        cv2.putText(frame, f"Red: {self.current_model_name}", (16, 61), font, 0.6, (0,0,0), 2)
        cv2.putText(frame, f"Red: {self.current_model_name}", (15, 60), font, 0.6, (255,255,0), 2)
        
        # Sombra y Texto para Resolución
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Res: {w}x{h}", (16, 91), font, 0.6, (0,0,0), 2)
        cv2.putText(frame, f"Res: {w}x{h}", (15, 90), font, 0.6, (0,255,255), 2)

        return frame

    def process_video_stream(self):
        if self.camera.is_running:
            frame = self.camera.get_frame()
            
            if frame is not None:
                # Calcular FPS reales
                current_time = time.time()
                fps = 1.0 / (current_time - self.prev_frame_time) if (current_time - self.prev_frame_time) > 0 else 0
                self.prev_frame_time = current_time
                
                # Suavizar los FPS (Filtro de media móvil para que el número no salte tan brusco)
                self.fps_smoothed = 0.9 * self.fps_smoothed + 0.1 * fps

                # IA: Realizar inferencia
                predicted_class, confidence, latency = self.engine.predict(frame)
                
                # Dibujar las métricas (HUD) sobre el cuadro RGB usando OpenCV
                frame_with_hud = self.draw_hud(frame, self.fps_smoothed, latency)
                
                # Visión: Pasamos el frame dibujado a PIL para la UI
                pil_img = Image.fromarray(frame_with_hud)
                
                # Actualizar la Interfaz Gráfica
                self.ui.update_video_frame(pil_img)
                self.ui.update_results(predicted_class, confidence, latency)
            
            # Ejecutar el siguiente ciclo
            self.ui.after(10, self.process_video_stream)

    def on_close(self):
        logger.info("Cerrando la aplicación...")
        if self.camera.is_running:
            self.camera.stop()
        self.ui.destroy()

    def run(self):
        self.ui.mainloop()

if __name__ == "__main__":
    app = EcoSortApp()
    app.run()