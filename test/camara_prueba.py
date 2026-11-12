import cv2

def scan_cameras():
    print("Iniciando escaneo de puertos de video (0 al 10)...")
    working_cameras = []
    
    for i in range(11):
        # Intentamos abrir el índice actual
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Si abre, intentamos leer un fotograma
            ret, frame = cap.read()
            if ret:
                print(f"✅ ¡Éxito! Cámara real encontrada en el índice: {i}")
                print(f"   Resolución detectada: {frame.shape[1]}x{frame.shape[0]}")
                working_cameras.append(i)
            else:
                print(f"⚠️ Índice {i} abrió, pero NO envía fotogramas (probablemente un codificador virtual).")
            cap.release()
        else:
            pass # No hay nada en este índice

    if not working_cameras:
        print("\n❌ OpenCV no pudo extraer fotogramas de ningún índice.")
        print("   Solución probable: Ejecutar el proyecto usando 'libcamerify python3 main.py'")
    else:
        print(f"\n🎉 ¡Todo listo! Tu cámara está en el/los índice(s): {working_cameras}")

if __name__ == "__main__":
    scan_cameras()