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
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="ECOSORT AI", font=ctk.CTkFont(size=28, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))

        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Módulo Edge AI - EcoSort", font=ctk.CTkFont(size=14))
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Arquitectura de Red Neuronal:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.model_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        
        model_names = list(config.AVAILABLE_MODELS.keys())
        self.model_selector = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=model_names,
            command=self._on_model_changed,
            font=ctk.CTkFont(size=13)
        )
        self.model_selector.grid(row=3, column=0, padx=20, pady=(10, 30))
        
        self.btn_toggle_camera = ctk.CTkButton(
            self.sidebar_frame, 
            text="INICIAR CÁMARA", 
            command=self._on_toggle_camera,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=40
        )
        self.btn_toggle_camera.grid(row=4, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Estado: Sistema Listo", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=20, sticky="s")

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
        
        self.prediction_label = ctk.CTkLabel(self.result_frame, text="CLASE: ---", font=ctk.CTkFont(size=36, weight="bold"))
        self.prediction_label.grid(row=0, column=0, padx=30, pady=20, sticky="w")

        self.confidence_label = ctk.CTkLabel(self.result_frame, text="Confianza: 0.0%", font=ctk.CTkFont(size=24, weight="bold"), text_color="#a3a3a3")
        self.confidence_label.grid(row=0, column=1, padx=30, pady=20, sticky="e")

        # --- CALLBACKS ---
        self.on_model_change_callback = None
        self.on_toggle_camera_callback = None

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
        else:
            self.btn_toggle_camera.configure(text="INICIAR CÁMARA", fg_color=config.UI_COLOR, hover_color="#1E6E43")