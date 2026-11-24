import cv2
import json
import time
import numpy as np
from datetime import datetime
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
        self.ui.on_capture_image_callback = self.capture_image
        self.ui.on_true_class_change_callback = self.change_true_class
        self.ui.on_reset_metrics_callback = self.reset_metrics
        self.ui.protocol("WM_DELETE_WINDOW", self.on_close)

        # Variables para Métricas HUD y Captura
        self.current_model_name = ""
        self.current_true_class = config.CLASSES[0]
        self.prev_frame_time = time.time()
        self.fps_smoothed = 0.0
        self.current_pil_image = None
        self.last_raw_frame = None
        self.last_predicted_class = "ninguno"
        self.last_raw_predicted_class = None
        self.last_confidence = 0.0
        self.last_latency = 0.0

        # Matrices de confusión por modelo (filas = clase real, columnas = clase predicha)
        self.confusion_matrices = {}

        default_model_name = list(config.AVAILABLE_MODELS.keys())[0]
        self.change_model(default_model_name)

    def _get_current_cm(self):
        n = len(config.CLASSES)
        if self.current_model_name not in self.confusion_matrices:
            self.confusion_matrices[self.current_model_name] = np.zeros((n, n), dtype=int)
        return self.confusion_matrices[self.current_model_name]

    def change_true_class(self, true_class):
        self.current_true_class = true_class
        logger.info(f"Clase real seleccionada para evaluación: {true_class}")
        self._refresh_ui_metrics_bar()

    def reset_metrics(self):
        n = len(config.CLASSES)
        self.confusion_matrices[self.current_model_name] = np.zeros((n, n), dtype=int)
        logger.info(f"Métricas reiniciadas para el modelo: {self.current_model_name}")
        self._refresh_ui_metrics_bar()
        self.ui.status_label.configure(text=f"Métricas reiniciadas ({self.current_model_name})", text_color="#00e5ff")

    def record_prediction(self, true_class, pred_class):
        """Registra un par (clase_real, clase_predicha) en la matriz de confusión del modelo activo."""
        if true_class in config.CLASSES and pred_class in config.CLASSES:
            cm = self._get_current_cm()
            i = config.CLASSES.index(true_class)
            j = config.CLASSES.index(pred_class)
            cm[i, j] += 1

    def compute_classification_metrics(self):
        """Calcula precision, recall, f1-score y support por clase, macro avg y weighted avg."""
        cm = self._get_current_cm()
        n = len(config.CLASSES)
        per_class = {}
        precs, recs, f1s, supports = [], [], [], []

        for i, cls_name in enumerate(config.CLASSES):
            tp = int(cm[i, i])
            support = int(cm[i, :].sum())
            predicted_total = int(cm[:, i].sum())

            precision = tp / predicted_total if predicted_total > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class[cls_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
            precs.append(precision)
            recs.append(recall)
            f1s.append(f1)
            supports.append(support)

        total_support = int(sum(supports))
        total_correct = int(np.trace(cm))
        accuracy = total_correct / total_support if total_support > 0 else 0.0

        # Promedio macro sobre las clases que ya tienen soporte o predicciones (o sobre todas si está vacío)
        active_indices = [i for i in range(n) if supports[i] > 0 or int(cm[:, i].sum()) > 0]
        if active_indices:
            macro_prec = float(np.mean([precs[i] for i in active_indices]))
            macro_rec = float(np.mean([recs[i] for i in active_indices]))
            macro_f1 = float(np.mean([f1s[i] for i in active_indices]))
        else:
            macro_prec = macro_rec = macro_f1 = 0.0

        if total_support > 0:
            weighted_prec = float(sum(precs[i] * supports[i] for i in range(n)) / total_support)
            weighted_rec = float(sum(recs[i] * supports[i] for i in range(n)) / total_support)
            weighted_f1 = float(sum(f1s[i] * supports[i] for i in range(n)) / total_support)
        else:
            weighted_prec = weighted_rec = weighted_f1 = 0.0

        return {
            "per_class": per_class,
            "accuracy": accuracy,
            "total_support": total_support,
            "macro_avg": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1, "support": total_support},
            "weighted_avg": {"precision": weighted_prec, "recall": weighted_rec, "f1": weighted_f1, "support": total_support},
        }

    def format_classification_report(self, metrics):
        """Devuelve una tabla en texto plano estilo scikit-learn con precision, recall, f1-score y support."""
        lines = [
            f"Reporte de Clasificación — Modelo: {self.current_model_name} (Clase Real activa: {self.current_true_class})",
            f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
            "",
        ]
        for cls_name in config.CLASSES:
            m = metrics["per_class"][cls_name]
            marker = "*" if cls_name == self.current_true_class else " "
            lines.append(
                f"{marker}{cls_name:>11} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f} {m['support']:>10d}"
            )
        lines.append("")
        total_sup = metrics["total_support"]
        lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {metrics['accuracy']:>10.2f} {total_sup:>10d}")
        ma = metrics["macro_avg"]
        lines.append(
            f"{'macro avg':>12} {ma['precision']:>10.2f} {ma['recall']:>10.2f} {ma['f1']:>10.2f} {total_sup:>10d}"
        )
        wa = metrics["weighted_avg"]
        lines.append(
            f"{'weighted avg':>12} {wa['precision']:>10.2f} {wa['recall']:>10.2f} {wa['f1']:>10.2f} {total_sup:>10d}"
        )
        return "\n".join(lines)

    def _refresh_ui_metrics_bar(self, metrics=None):
        if metrics is None:
            metrics = self.compute_classification_metrics()
        cls_m = metrics["per_class"][self.current_true_class]
        self.ui.update_metrics_bar(
            true_class=self.current_true_class,
            precision=cls_m["precision"],
            recall=cls_m["recall"],
            f1=cls_m["f1"],
            support=cls_m["support"],
            total_support=metrics["total_support"],
        )

    def change_model(self, model_name):
        model_path = config.AVAILABLE_MODELS.get(model_name)
        if model_path:
            logger.info(f"Cambiando al modelo: {model_name}")
            self.current_model_name = model_name.split()[0]  # Toma solo el primer nombre (ej. "MobileNetV3")
            success = self.engine.load_model(model_path)
            if success:
                self.ui.status_label.configure(text=f"Modelo cargado: {self.current_model_name}", text_color="#00FF00")
                self._refresh_ui_metrics_bar()
            else:
                self.ui.status_label.configure(text="Error cargando modelo", text_color="red")

    def toggle_camera(self):
        if not self.camera.is_running:
            success = self.camera.start()
            if success:
                self.ui.set_camera_button_state(is_running=True)
                self.ui.status_label.configure(text="Cámara encendida", text_color="#00FF00")
                self.prev_frame_time = time.time()  # Reiniciar contador FPS
                self.process_video_stream()
        else:
            self.camera.stop()
            self.current_pil_image = None
            self.last_raw_frame = None
            self.ui.set_camera_button_state(is_running=False)
            self.ui.status_label.configure(text="Cámara detenida", text_color="gray")
            self.ui.video_label.configure(image="", text="Cámara Detenida")

    @staticmethod
    def _json_default(obj):
        """Convierte escalares o arreglos de NumPy a tipos nativos de Python para serialización JSON segura."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def build_capture_annotation(self, image_filename, capture_dt, metrics):
        """Construye el diccionario JSON con toda la información del HUD y evaluación de la imagen capturada."""
        w, h = self.current_pil_image.size if self.current_pil_image is not None else (640, 480)
        raw_pred = self.last_raw_predicted_class or self.last_predicted_class

        por_clase_json = {}
        for cls_name in config.CLASSES:
            m = metrics["per_class"][cls_name]
            por_clase_json[cls_name] = {
                "precision": round(float(m["precision"]), 4),
                "recall": round(float(m["recall"]), 4),
                "f1-score": round(float(m["f1"]), 4),
                "support": int(m["support"]),
            }

        ma = metrics["macro_avg"]
        wa = metrics["weighted_avg"]

        return {
            "timestamp": capture_dt.isoformat(),
            "imagen": {
                "archivo": image_filename,
                "ruta_relativa": f"figure/{image_filename}",
                "resolucion": {
                    "ancho": int(w),
                    "alto": int(h),
                    "formato": f"{int(w)}x{int(h)}",
                },
            },
            "hud_rendimiento": {
                "fps": round(float(self.fps_smoothed), 2),
                "fps_hud": int(self.fps_smoothed),
                "latencia_ia_ms": round(float(self.last_latency), 3),
                "red_neuronal": self.current_model_name,
            },
            "modelo": {
                "nombre": self.current_model_name,
                "input_shape": [int(x) for x in self.engine.input_shape],
                "channel_first": bool(getattr(self.engine, "is_channel_first", False)),
                "umbral_confianza": float(config.CONFIDENCE_THRESHOLD),
            },
            "prediccion_captura": {
                "clase_real": self.current_true_class,
                "clase_predicha": raw_pred,
                "estado_ui": self.last_predicted_class,
                "acierto": bool(raw_pred == self.current_true_class),
                "confianza": round(float(self.last_confidence), 4),
                "confianza_porcentaje": round(float(self.last_confidence) * 100.0, 2),
                "evaluacion_tiempo_real_activa": bool(self.ui.is_realtime_eval_enabled()),
            },
            "metricas_evaluacion": {
                "clase_real_activa": {
                    "clase": self.current_true_class,
                    **por_clase_json[self.current_true_class],
                },
                "por_clase": por_clase_json,
                "accuracy": round(float(metrics["accuracy"]), 4),
                "macro_avg": {
                    "precision": round(float(ma["precision"]), 4),
                    "recall": round(float(ma["recall"]), 4),
                    "f1-score": round(float(ma["f1"]), 4),
                    "support": int(ma["support"]),
                },
                "weighted_avg": {
                    "precision": round(float(wa["precision"]), 4),
                    "recall": round(float(wa["recall"]), 4),
                    "f1-score": round(float(wa["f1"]), 4),
                    "support": int(wa["support"]),
                },
                "total_support": int(metrics["total_support"]),
                "matriz_confusion": {
                    "clases": list(config.CLASSES),
                    "matriz": [[int(val) for val in row] for row in self._get_current_cm().tolist()],
                },
            },
        }

    def capture_image(self):
        """Guarda el fotograma actual en figure/ y su anotación completa en figure/annotations/."""
        if not self.camera.is_running or self.current_pil_image is None:
            logger.warning("Intento de captura sin fotograma activo.")
            self.ui.status_label.configure(text="Inicia la cámara para capturar", text_color="orange")
            return

        try:
            config.FIGURE_DIR.mkdir(exist_ok=True)
            config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

            # Si la evaluación continua estaba desactivada, registramos esta captura puntual en las métricas
            if not self.ui.is_realtime_eval_enabled() and self.last_raw_predicted_class is not None:
                self.record_prediction(self.current_true_class, self.last_raw_predicted_class)
                metrics = self.compute_classification_metrics()
                self._refresh_ui_metrics_bar(metrics)
                if self.last_raw_frame is not None:
                    updated_frame = self.draw_hud(
                        self.last_raw_frame.copy(),
                        self.fps_smoothed,
                        self.last_latency,
                        self.last_predicted_class,
                        self.last_confidence,
                        metrics,
                    )
                    self.current_pil_image = Image.fromarray(updated_frame)
                    self.ui.update_video_frame(self.current_pil_image)
            else:
                metrics = self.compute_classification_metrics()

            capture_dt = datetime.now()
            timestamp = capture_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            clean_pred = self.last_predicted_class.replace(".", "").strip() or "sin_clase"
            base_name = f"captura_{timestamp}_{self.current_model_name}_real-{self.current_true_class}_pred-{clean_pred}"

            image_filename = f"{base_name}.png"
            image_filepath = config.FIGURE_DIR / image_filename
            self.current_pil_image.save(image_filepath)

            json_filename = f"{base_name}.json"
            json_filepath = config.ANNOTATIONS_DIR / json_filename
            annotation_data = self.build_capture_annotation(image_filename, capture_dt, metrics)
            json_text = json.dumps(annotation_data, indent=2, ensure_ascii=False, default=self._json_default)
            json_filepath.write_text(json_text + "\n", encoding="utf-8")

            report_str = self.format_classification_report(metrics)
            logger.info(
                f"Captura guardada:\n  Imagen: {image_filepath}\n  JSON  : {json_filepath}\n{report_str}"
            )
            self.ui.status_label.configure(text=f"IMG+JSON: {base_name[:22]}...", text_color="#00BFFF")
        except Exception as e:
            logger.error(f"Error al guardar la captura: {e}")
            self.ui.status_label.configure(text="Error al guardar captura", text_color="red")

    def draw_hud(self, frame, fps, latency, predicted_class, confidence, metrics):
        """Dibuja métricas de rendimiento (HUD) y tabla precision/recall/f1-score/support sobre el fotograma."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = frame.shape[:2]

        # --- 1. HUD Superior Izquierdo (FPS, Modelo, Resolución, Latencia, Real vs Pred) ---
        cv2.putText(frame, f"FPS: {int(fps)} | Lat: {latency:.1f}ms", (16, 29), font, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)} | Lat: {latency:.1f}ms", (15, 28), font, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Red: {self.current_model_name}", (16, 56), font, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, f"Red: {self.current_model_name}", (15, 55), font, 0.55, (255, 255, 0), 2)

        cv2.putText(frame, f"Res: {w}x{h}", (16, 81), font, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, f"Res: {w}x{h}", (15, 80), font, 0.55, (0, 255, 255), 2)

        raw_pred = self.engine.last_raw_class or predicted_class
        match_color = (0, 255, 128) if raw_pred == self.current_true_class else (255, 100, 100)
        eval_text = f"Real: {self.current_true_class} | Pred: {raw_pred} ({confidence * 100:.1f}%)"
        cv2.putText(frame, eval_text, (16, 106), font, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, eval_text, (15, 105), font, 0.5, match_color, 2)

        # --- 2. Tabla de Evaluación (precision, recall, f1-score, support) en Esquina Inferior Derecha ---
        box_w, box_h = 335, 168
        x0 = max(10, w - box_w - 10)
        y0 = max(10, h - box_h - 10)
        x1, y1 = x0 + box_w, y0 + box_h

        # Fondo semitransparente oscuro
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 100, 100), 1)

        # Encabezado de la tabla
        header = f"{'CLASE':<10} {'PREC':>5} {'REC':>5} {'F1':>5} {'SUP':>5}"
        cv2.putText(frame, header, (x0 + 10, y0 + 18), font, 0.4, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + 8, y0 + 23), (x1 - 8, y0 + 23), (90, 90, 90), 1)

        # Filas por cada clase
        y_cursor = y0 + 38
        for cls_name in config.CLASSES:
            m = metrics["per_class"][cls_name]
            is_active = cls_name == self.current_true_class
            prefix = ">" if is_active else " "
            row_str = f"{prefix}{cls_name:<9} {m['precision']:>5.2f} {m['recall']:>5.2f} {m['f1']:>5.2f} {m['support']:>5d}"
            row_color = (0, 255, 128) if is_active else (225, 225, 225)
            cv2.putText(frame, row_str, (x0 + 10, y_cursor), font, 0.4, row_color, 1, cv2.LINE_AA)
            y_cursor += 16

        cv2.line(frame, (x0 + 8, y_cursor - 10), (x1 - 8, y_cursor - 10), (90, 90, 90), 1)

        # Filas de resumen: accuracy, macro avg, weighted avg
        total_sup = metrics["total_support"]
        acc_str = f" {'accuracy':<9} {'':>5} {'':>5} {metrics['accuracy']:>5.2f} {total_sup:>5d}"
        cv2.putText(frame, acc_str, (x0 + 10, y_cursor + 3), font, 0.4, (255, 230, 120), 1, cv2.LINE_AA)

        ma = metrics["macro_avg"]
        ma_str = f" {'macro avg':<9} {ma['precision']:>5.2f} {ma['recall']:>5.2f} {ma['f1']:>5.2f} {total_sup:>5d}"
        cv2.putText(frame, ma_str, (x0 + 10, y_cursor + 19), font, 0.4, (180, 220, 255), 1, cv2.LINE_AA)

        wa = metrics["weighted_avg"]
        wa_str = f" {'weighted':<9} {wa['precision']:>5.2f} {wa['recall']:>5.2f} {wa['f1']:>5.2f} {total_sup:>5d}"
        cv2.putText(frame, wa_str, (x0 + 10, y_cursor + 35), font, 0.4, (180, 220, 255), 1, cv2.LINE_AA)

        return frame

    def process_video_stream(self):
        if self.camera.is_running:
            frame = self.camera.get_frame()

            if frame is not None:
                self.last_raw_frame = frame.copy()

                # Calcular FPS reales
                current_time = time.time()
                fps = 1.0 / (current_time - self.prev_frame_time) if (current_time - self.prev_frame_time) > 0 else 0
                self.prev_frame_time = current_time

                # Suavizar los FPS (Filtro de media móvil para que el número no salte tan brusco)
                self.fps_smoothed = 0.9 * self.fps_smoothed + 0.1 * fps

                # IA: Realizar inferencia
                predicted_class, confidence, latency = self.engine.predict(frame)
                raw_pred_class = self.engine.last_raw_class

                self.last_predicted_class = predicted_class
                self.last_raw_predicted_class = raw_pred_class
                self.last_confidence = confidence
                self.last_latency = latency

                # Actualizar matriz de confusión en tiempo real si el switch está activo
                if self.ui.is_realtime_eval_enabled() and raw_pred_class is not None:
                    self.record_prediction(self.current_true_class, raw_pred_class)

                metrics = self.compute_classification_metrics()

                # Dibujar las métricas (HUD + tabla precision/recall/f1/support) sobre el cuadro RGB
                frame_with_hud = self.draw_hud(
                    frame.copy(),
                    self.fps_smoothed,
                    latency,
                    predicted_class,
                    confidence,
                    metrics,
                )

                # Visión: Pasamos el frame dibujado a PIL para la UI
                pil_img = Image.fromarray(frame_with_hud)
                self.current_pil_image = pil_img

                # Actualizar la Interfaz Gráfica
                self.ui.update_video_frame(pil_img)
                self.ui.update_results(predicted_class, confidence, latency)
                self._refresh_ui_metrics_bar(metrics)

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
