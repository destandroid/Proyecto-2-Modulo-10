# CNN multitarea para predicción demográfica (UTKFace)

Proyecto de **clasificación multitarea** que estima **edad** (10 rangos), **género** (2) y **raza** (5) a partir de rostros. Incluye EDA, opción de **preprocesamiento facial** (MTCNN + normalización + letterbox) y entrenamiento con **validación cruzada 5-fold**, manteniendo **10% de test** fijo. Implementado en **Google Colab**.

## Datos
- **UTKFace**: etiquetas en el nombre `age_gender_race_datetime.jpg`.  
  https://susanqq.github.io/UTKFace/

## Modelo
- CNN liviana (3× Conv+Pool → GlobalAvgPool → Dropout).
- Tres salidas **softmax**: `age(10)`, `gender(2)`, `race(5)`.
- Opción rápida **sin** preprocesamiento (solo resize+normalización) para entrenar más veloz.

## Entrenamiento y evaluación
- Particiones: **10% test**; en el 90% restante **5-fold CV** (≈ 72% train / 18% val / 10% test).
- Optimizador **Adam**, pérdidas CCE por cabeza, **EarlyStopping** + **ReduceLROnPlateau**.
- Modelo final exportado como **`modelo.h5`**.
