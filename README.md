# 🎨 PaintingsAI
**Clasificación multitarea de obras de arte usando redes neuronales convolucionales.**

Este proyecto aborda el reto de predecir el artista y el estilo de una pintura utilizando una red basada en ResNet. 

## 🛠️ Técnicas empleadas
* Regularización
* Balanceo de clases
* Aumento de datos para mejorar el rendimiento del modelo

**Librerías utilizadas:** Python, Pytorch (torch.nn, torch.optim, torch.utils.data, torchvision,...), pandas, os, PIL, matplotlib.pyplot.

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
2. Clona el repositorio:
```bash
git clone https://github.com/tu_usuario/PaintingsAI.git
cd PaintingsAI
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Descarga las imágenes en `data/1001_images` y asegúrate de que el CSV (`dataset_combined.csv`) esté accesible.

4. Ejecuta el entrenamiento:
```bash
python src/train.py --epochs 20 --batch-size 32 --model resnet34
```

## 📊 Resultados
La arquitectura MultiTaskResNet_m (ResNet34 backbone) con balanceo de clases y regularización alcanza más del 60% de precisión en la predicción de artistas y estilos.
