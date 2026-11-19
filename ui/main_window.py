import customtkinter as ctk
from PIL import Image
import config

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Configuración de la ventana principal
        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_SIZE)
        ctk.set_appearance_mode(config.UI_THEME)
        ctk.set_default_color_theme(config.UI_COLOR)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- PANEL LATERAL (SIDEBAR) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="ECOSORT AI", font=ctk.CTkFont(size=26, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 2))

        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Módulo Edge AI - EcoSort", font=ctk.CTkFont(size=13))
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Arquitectura de Red Neuronal:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.model_label.grid(row=2, column=0, padx=20, pady=(5, 0), sticky="w")
        
        model_names = list(config.AVAILABLE_MODELS.keys())
        self.model_selector = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=model_names,
            command=self._on_model_changed,
            font=ctk.CTkFont(size=13)
        )
        self.model_selector.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.true_class_label = ctk.CTkLabel(self.sidebar_frame, text="Clase Real (Objeto a evaluar):", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.true_class_label.grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")

        self.true_class_selector = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=config.CLASSES,
            command=self._on_true_class_changed,
            font=ctk.CTkFont(size=13)
        )
        self.true_class_selector.grid(row=5, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.eval_realtime_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Evaluar en Tiempo Real",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.eval_realtime_switch.select()
        self.eval_realtime_switch.grid(row=6, column=0, padx=20, pady=(2, 12), sticky="w")
        
        self.btn_toggle_camera = ctk.CTkButton(
            self.sidebar_frame, 
            text="INICIAR CÁMARA", 
            command=self._on_toggle_camera,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=38
        )
        self.btn_toggle_camera.grid(row=7, column=0, padx=20, pady=6, sticky="ew")

        self.btn_capture_image = ctk.CTkButton(
            self.sidebar_frame,
            text="CAPTURAR IMAGEN",
            command=self._on_capture_image,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=38,
            fg_color="#1f538d",
            hover_color="#14375e",
            state="disabled"
        )
        self.btn_capture_image.grid(row=8, column=0, padx=20, pady=6, sticky="ew")

        self.btn_reset_metrics = ctk.CTkButton(
            self.sidebar_frame,
            text="REINICIAR MÉTRICAS",
            command=self._on_reset_metrics,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=32,
            fg_color="#4a4a4a",
            hover_color="#333333"
        )
        self.btn_reset_metrics.grid(row=9, column=0, padx=20, pady=6, sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Estado: Sistema Listo", text_color="gray")
        self.status_label.grid(row=11, column=0, padx=20, pady=15, sticky="s")

        # --- ÁREA PRINCIPAL (VIDEO Y RESULTADOS) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="Esperando conexión CSI...", font=ctk.CTkFont(size=20), fg_color="#1a1a1a", corner_radius=15)
        self.video_label.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # Panel de Resultados Mejorado
        self.result_frame = ctk.CTkFrame(self.main_frame, height=120, fg_color="#2b2b2b", corner_radius=15)
        self.result_frame.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(1, weight=1)
        
        self.prediction_label = ctk.CTkLabel(self.result_frame, text="CLASE: ---", font=ctk.CTkFont(size=32, weight="bold"))
        self.prediction_label.grid(row=0, column=0, padx=25, pady=(12, 4), sticky="w")

        self.confidence_label = ctk.CTkLabel(self.result_frame, text="Confianza: 0.0%", font=ctk.CTkFont(size=22, weight="bold"), text_color="#a3a3a3")
        self.confidence_label.grid(row=0, column=1, padx=25, pady=(12, 4), sticky="e")

        self.metrics_label = ctk.CTkLabel(
            self.result_frame,
            text="precision: 0.00  |  recall: 0.00  |  f1-score: 0.00  |  support: 0",
            font=ctk.CTkFont(family="monospace", size=14, weight="bold"),
            text_color="#00e5ff"
        )
        self.metrics_label.grid(row=1, column=0, columnspan=2, padx=25, pady=(0, 12), sticky="ew")

        # --- CALLBACKS ---
        self.on_model_change_callback = None
        self.on_toggle_camera_callback = None
        self.on_capture_image_callback = None
        self.on_true_class_change_callback = None
        self.on_reset_metrics_callback = None

        # Diccionario de colores para los materiales
        self.color_map = {
            "aluminio": "#A9A9A9", # Gris Plata
            "carton": "#D2B48C",   # Café Claro / Naranja
            "plastico": "#1E90FF", # Azul Brillante
            "vidrio": "#20B2AA",   # Verde Agua / Cyan
            "otros": "#FFFFFF",    # Blanco
            "buscando...": "#555555" # Gris oscuro
        }

    def _on_model_changed(self, selected_model_name):
        if self.on_model_change_callback:
            self.on_model_change_callback(selected_model_name)

    def _on_toggle_camera(self):
        if self.on_toggle_camera_callback:
            self.on_toggle_camera_callback()

    def _on_capture_image(self):
        if self.on_capture_image_callback:
            self.on_capture_image_callback()

    def _on_true_class_changed(self, selected_true_class):
        if self.on_true_class_change_callback:
            self.on_true_class_change_callback(selected_true_class)

    def _on_reset_metrics(self):
        if self.on_reset_metrics_callback:
            self.on_reset_metrics_callback()

    def is_realtime_eval_enabled(self):
        return bool(self.eval_realtime_switch.get())

    def update_metrics_bar(self, true_class, precision, recall, f1, support, total_support):
        texto = (
            f"[{true_class.upper()}]  precision: {precision:.2f}  |  "
            f"recall: {recall:.2f}  |  f1-score: {f1:.2f}  |  "
            f"support: {support} (total: {total_support})"
        )
        self.metrics_label.configure(text=texto)

    def update_video_frame(self, pil_image):
        ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
        self.video_label.configure(image=ctk_img, text="")

    def update_results(self, predicted_class, confidence, inference_time_ms):
        # Seleccionar color según el material detectado
        color = self.color_map.get(predicted_class.lower(), "#FFFFFF")
        
        texto_clase = f"CLASE: {predicted_class.upper()}"
        if predicted_class == "buscando...":
            texto_clase = "ANALIZANDO..."
            
        self.prediction_label.configure(text=texto_clase, text_color=color)
        self.confidence_label.configure(text=f"Confianza: {confidence * 100:.1f}%")
        self.status_label.configure(text=f"Latencia IA: {inference_time_ms:.1f} ms", text_color="#00FF00")
        
    def set_camera_button_state(self, is_running):
        if is_running:
            self.btn_toggle_camera.configure(text="DETENER CÁMARA", fg_color="#8B0000", hover_color="#5a0000")
            self.btn_capture_image.configure(state="normal")
        else:
            self.btn_toggle_camera.configure(text="INICIAR CÁMARA", fg_color=config.UI_COLOR, hover_color="#1E6E43")
            self.btn_capture_image.configure(state="disabled")