# Informe Técnico - PaintingsAI

## 1. Introducción
El reconocimiento de obras de arte es una tarea compleja donde intervienen múltiples factores (autores y estilos). Este proyecto trata la clasificación multitarea: identificando tanto el artista como el estilo de una obra artística digitalizada.

## 2. Solución Propuesta
Se entrena una red convolucional multitarea personalizada, **MultiTaskResNet_m**, con backbone ResNet34 y dos cabezas especializadas. El pipeline incluye:

- **Modelo**: MultiTaskResNet_m (basado en ResNet34 preentrenado)
- **Cabezas multitarea**: Dos salidas (`artist_head`, `style_head`)
- **Regularización**: Dropout, BatchNorm
- **Optimización**: Adam + `ReduceLROnPlateau`
- **Aumento de datos**: Rotaciones, flip, color jitter
- **Balanceo de clases**: `WeightedRandomSampler` + agrupación de clases minoritarias como "Otros"
- **Dataset**: 70% train, 15% val, 15% test

## 3. Resultados

| Modelo               | Accuracy Artista | Accuracy Estilo | Observaciones                       |
|----------------------|------------------|------------------|-------------------------------------|
| MultiTaskResNet (ResNet18 Base)        | 2.99%            | 16.92%            | Primera versión básica multitarea                  |
| MultiTaskResNet_1 (ResNet18 Dropout)     | 4.48%            | 25.37%            | Con BatchNorm, dropout=0.4, scheduler, activación              |
| MultiTaskResNet_34 (ResNet34 Dropout)  | 3.48%            | 17.91%            | Más capas, arquitectura más profunda, dropout=0.4 |
| MultiTaskResNet_m (ResNet34 mejorada)   | 16.92%            | 22.39%            | Mejoras: Balanceo, dropout=0.2, heads separadas  |

## 4. Estado del Arte
- Uso de **transfer learning** en modelos como ResNet y EfficientNet [1]
- Técnicas multitarea para clasificación múltiple [2]
- Métodos de balanceo de clases y data augmentation [3]

### Referencias
[1] He et al., 2015. Deep Residual Learning for Image Recognition.  
[2] Ruder, S., 2017. An Overview of Multi-Task Learning in Deep Neural Networks.  
[3] Zhang et al., 2017. Mixup: Beyond Empirical Risk Minimization.

## Apéndice: Técnicas empleadas en `MultiTaskResNet_m`
La red `MultiTaskResNet_m` representa la versión final y más óptima del experimento. 
A continuación se detallan las técnicas y configuraciones clave utilizadas:

### ✅ 1. Transfer Learning
Se utiliza ResNet34 preentrenada como extractor de características (`features = resnet34(weights=ResNet34_Weights.DEFAULT)`), eliminando su capa final para añadir cabezas personalizadas.

### 🧩 2. Multitarea
Se definen dos salidas distintas:
- `artist_head` para clasificación del artista
- `style_head` para clasificación del estilo
Ambas comparten una capa intermedia (`shared_head`) con capas densas y activaciones.

### 🎛️ 3. Modularidad
El modelo es configurable mediante argumentos para:
- Activación (`ReLU` o `Tanh`)
- Inclusión de `BatchNorm`
- Inclusión y nivel de `Dropout`

### 🧃 4. Regularización
- `Dropout` con probabilidad ajustable (`dropout_prob`)
- `BatchNorm1d` para estabilizar la distribución de activaciones

### ⚖️ 5. Pérdida ponderada
Función de pérdida propia para dar más importancia a la predicción de autor:
```python
0.7 * CrossEntropy(artista) + 0.3 * CrossEntropy(estilo)
```

### 📉 6. Scheduler de aprendizaje
Uso de `ReduceLROnPlateau` para reducir el learning rate automáticamente si no hay mejora.

### 🧠 7. Link Function
`link_FN` permite personalizar el procesamiento posterior (ej. aplicar softmax), aunque por defecto es identidad (`lambda x, dim: x`)

### 📌 8. Detalles técnicos
Las librerías utilizadas en este proyecto incluyen:

| Librería       | Descripción                                     |
|----------------|-------------------------------------------------|
| `torch`        | Framework principal para redes neuronales       |
| `torchvision`  | Modelos preentrenados y transformaciones        |
| `pandas`       | Lectura y manipulación de CSV                   |
| `matplotlib`   | Visualización de resultados                     |
| `Pillow`       | Lectura y procesamiento de imágenes             |
| `os` / `random`| Incluidas en Python estándar                    |






