# 🎨 PaintingsAI
**Clasificación multitarea de obras de arte usando redes neuronales convolucionales.**

Este proyecto trata de predecir el artista y el estilo de una obra artística usando una red neuronal basada en ResNet. 

**Importante:** Este proyecto está realizado con un IDE - Entorno de Desarrollo Integrado compatible con Python (Visual Studio Code)

## 📁 Estructura del proyecto
```
PaintingsAI/
│
├── README.md                    <- Resumen del proyecto y guía rápida)
├── report.md                    <- Documento técnico explicativo (versión extendida del README)
├── final.ipynb                  <- Notebook con el experimento principal
├── src/                         <- Código fuente organizado de la solución
│   ├── dataset.py               <- Clase ArtDataset y transformaciones
│   ├── model.py                 <- Definición de la ResNet34 (MultiTaskResNet_m) y MultiTaskHeads
│   ├── train.py                 <- Entrenamiento y validación
│   └── utils.py                 <- Funciones auxiliares 
├── requirements.txt             <- Lista de dependencias
└── data/                  
    ├── README.md                <- Resumen de los contenidos de la carpeta
    ├── dataset_completo.csv     <- Dataset sobre el que se trabaja
    ├── filtered_dataset.csv     <- Dataset usado en la ResNet34 (MultiTaskResNet_m)
    └── images_download.py       <- Instrucciones para descargar las imágenes

```

## 🚀 Cómo ejecutar
1. Requisitos previos:
- Tener [Python 3.8+](https://www.python.org/downloads/) instalado.
- (Recomendado) Tener `pip` y `virtualenv` configurados.

---

2. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/PaintingsAI.git
cd PaintingsAI
```

---

3. Instalación previa: (se puede usar o la 3.1 o 3.2)

   3.1 Instalar las librerías desde ["requirements.txt"](requirements.txt):
   
       3.1.1 Crear y activar un entorno virtual (opcional pero recomendable):
       ```bash
       python -m venv venv
       source venv/bin/activate # En Linux/macOS
       -\venv\Scripts\activate # En Windows
       ```

       3.1.2 Instalar todas las dependencias necesarias:
       ```bash
       pip install -r requirements.txt
       ```

   3.2 Instalación manual de dependencias:
   Si no tienes el archivo ["requirements.txt"](requirements.txt), puedes instalar una por una.
   ```bash
   pip install torch torchvision pandas matplotlib pillow
   ```

   3.3 Verificación de la instalación:
   ```bash
   python -c "import torch; import torchvision; import pandas; import matplotlib; from PIL import Image; print('Todo correcto')"
   ```

---

4. Descarga las imágenes en [`data/1001_images`](data/README.md) y asegúrate de que el CSV [`dataset_completo.csv`](data/dataset_completo) esté accesible si decides no usar la url.

---

5. Ejecuta el entrenamiento:
```bash
python src/train.py --epochs 20 --batch-size 32 --model resnet34
```

## 🛠️ Técnicas empleadas
* Transfer Learning con ResNet34
* Arquitectura multitarea
* Configuración flexible
* Regularización 
* Pérdida ponderada 
* Balanceo de clases

## 📊 Resultados
La arquitectura MultiTaskResNet_m (ResNet34 backbone) con balanceo de clases y regularización alcanza más del 60% de precisión en la predicción de artistas y estilos.
