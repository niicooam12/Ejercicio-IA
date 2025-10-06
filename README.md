# Redes-Neuronales-Trabajo

  ## Descripcion
.Este proyecto desarrolla una aplicación para el reconocimiento de dígitos manuscritos mediante una Red Neuronal Convolucional (CNN) entrenada con el conjunto de datos MNIST. La aplicación permite cargar o arrastrar imágenes de dígitos dibujados (por ejemplo, en Paint), predecir el número utilizando un modelo de TensorFlow/Keras y subir dichas imágenes a Firebase Storage, obteniendo una URL pública. La interfaz gráfica está construida con Flask. 
  ## Caracteristicas
.Interfaz gráfica interactiva con soporte para arrastrar y soltar, además de carga manual de imágenes.

.Carga automática de las imágenes en Firebase Storage con generación de URL pública.

.Depuración integrada a través de mensajes en consola para simplificar el desarrollo.

.Umbral de confianza configurable (actualmente establecido en 0.2) para validar las predicciones.

  ## Requisitos
.Python 3.8 o superior
.Librerías requeridas (instalables vía pip):
.tensorflow (para el modelo CNN)
.keras (integrado con TensorFlow)
.firebase-admin (para interacción con Firebase Storage)
.numpy (para cálculos matriciales)
.pillow (para procesamiento de imágenes)
.matplotlib (para visualización durante el entrenamiento)
.pyrebase4 (para autenticación Firebase, opcional)
.tkinterdnd2 (para drag-and-drop en Tkinter)

