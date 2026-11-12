#!/bin/bash

echo "1/2: Asegurando soporte de motor gráfico en el sistema operativo..."
sudo apt-get update
sudo apt-get install -y python3-tk python3-pil.imagetk

echo "2/2: Forzando al entorno virtual a usar una versión de Pillow completa e independiente..."

Usamos --ignore-installed para decirle a pip que ignore la versión de Linux

y descargue la versión oficial completa de Python directamente en tu .venv

pip install --ignore-installed Pillow

echo ""
echo "¡Reparación completa! 🎉"
echo "Por favor, ejecuta nuevamente: python main.py"