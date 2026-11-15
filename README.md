# Optimización en dos niveles para la segmentación automática de tumores cerebrales en resonancia magnética

### Descenso de gradiente en una red U-Net y metaoptimización de hiperparámetros mediante un algoritmo genético

**Javier Ruano Hernández · Alejandro Delgado Valera**
Universidad de Las Palmas de Gran Canaria · Asignatura *Optimización y Heurística* · Curso 2026–2027

**Idioma:** **Español** · [English](README.en.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](docs/development.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](docs/development.md)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#demo)

![Imagen de entrada, predicción del modelo F y anotación del experto](docs/hero_overlay.png)

*Izquierda: imagen de entrada · Centro: predicción del modelo F · Derecha: la anotación del experto (este caso de glioma: Dice 0.87).*

## Sobre el proyecto

Segmentar tumores cerebrales en una resonancia magnética es, para una red neuronal, un problema difícil por dos razones: el tumor ocupa de media solo el **1,7 % de los píxeles** de cada corte (un fuerte desequilibrio de clases) y el criterio de un experto no es exacto al milímetro. Este proyecto aborda el reto planteándolo como un **problema de optimización en dos niveles**:

- **Nivel interno — entrenar bien la red.** El entrenamiento de una U-Net se formula como la minimización de una función de pérdida no convexa sobre entre 1,8 y 7,2 millones de parámetros, resuelta con descenso de gradiente estocástico (Adam). Desde la óptica de la optimización analizamos la arquitectura, las funciones de pérdida sustitutas (Dice, Focal y Focal Tversky) y la programación de la tasa de aprendizaje.
- **Nivel externo — ajustar los hiperparámetros.** Los cuatro hiperparámetros que condicionan ese entrenamiento (tasa de aprendizaje, decaimiento de pesos, número de filtros y profundidad) los ajusta un **algoritmo genético propio** de codificación real, con selección por torneo, cruce BLX-α, mutación gaussiana con reflexión y elitismo, y una función de aptitud de baja fidelidad: el Dice de validación tras solo tres épocas de entrenamiento.

Con 46 evaluaciones, el algoritmo genético mejora la aptitud de la configuración manual de 0.441 a 0.509 y, reentrenada a longitud completa, la configuración evolucionada eleva el Dice medio en el conjunto de prueba de **0.691 a 0.776**. El análisis posterior de la solución heurística revela un *sesgo de horizonte corto* en la aptitud —favorece redes pequeñas de convergencia rápida—, que corregimos con mayor capacidad, *cosine annealing* de la tasa de aprendizaje y una calibración de la pérdida guiada por diagnóstico: el modelo entrenado desde cero final alcanza **0.8175**, y con transfer learning desde una ResNet34 preentrenada, **0.8250** —un **0.134** por encima del punto de partida—. El algoritmo genético, por sí solo, explica aproximadamente **dos tercios de la mejora total**.

Todo se valida con un split *por paciente* sobre un dataset público de 3.064 cortes, y el proyecto se completa con una **demo interactiva bilingüe** que cualquiera puede ejecutar en el navegador.

## Demo

> **Enlace a la demo:** [streamlit.app](https://unet-tumor-segmentation-jruano-adelgado.streamlit.app/)

## Cómo funciona

1. **Datos con corte por paciente.** 3.064 cortes T1 con contraste de 233 pacientes (dataset Figshare), repartidos en 2.004 para entrenamiento, 468 de validación y 592 de prueba. El reparto es por paciente, no por imagen: ningún paciente aparece a la vez en entrenamiento y prueba, y el conjunto de prueba **no se usa nunca** para elegir modelos ni hiperparámetros.

2. **Una U-Net que clasifica cada píxel.** Arquitectura codificador–descodificador habitual en segmentación médica: cada píxel de la imagen se clasifica como fondo, meningioma, glioma o hipófisis.

3. **Pérdidas diseñadas para el desequilibrio de clases.** Como el tumor ocupa una fracción mínima de la imagen, una pérdida por píxel trivial aprendería a predecir «fondo» siempre. Combinamos el coeficiente Dice (solape con la anotación del experto) con las pérdidas Focal y Focal Tversky, que penalizan más los falsos negativos, y calibramos sus pesos con un análisis de errores (precisión/sensibilidad por clase y matrices de confusión).

4. **Aumento de datos y programación del aprendizaje.** Diseño factorial 2×2 de aumento de datos (espacial / de intensidad) y comparación de tasas de aprendizaje constantes frente a *cosine annealing*, con selección de modelo mediante parada temprana sobre el conjunto de validación.

5. **Optimización externa con algoritmo genético.** Un AG en codificación real ajusta tasa de aprendizaje, decaimiento de pesos, filtros y profundidad. La aptitud se mide a baja fidelidad (Dice de validación a tres épocas) con memoización, lo que permite **46 evaluaciones** en total; se compara contra una **búsqueda aleatoria con idéntico presupuesto** como referencia justa. Además, el resultado se analiza críticamente: el AG converge 3,2 veces más rápido que la búsqueda aleatoria, pero su aptitud tiene sesgo de horizonte corto, defecto que identificamos y corregimos en la fase de refinamiento.

6. **Transfer learning como complemento.** El mejor modelo sin preentrenar se contrasta con una U-Net cuyo codificador ResNet34 viene preentrenado en ImageNet (tasa reducida para el codificador): mejora la confusión entre tipos de tumor, sin mejora estadísticamente significativa en la delimitación (intervalos de confiancia por bootstrap de pacientes).

7. **Evaluación rigurosa.** Dice e IoU por clase, tasa de detección por corte (cuántos tumores pequeños se escapan), análisis de falsos positivos y negativos, y comparación por pares con reentrenamientos controlados. Los puntos de control se seleccionan en validación y se evalúan una sola vez en prueba.

## Datos

Jun Cheng, *brain tumor dataset*, figshare (2015), <https://doi.org/10.6084/m9.figshare.1512427>
— 3.064 cortes T1 con contraste de 233 pacientes (708 meningioma, 1.426 glioma, 930 hipófisis), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Los artículos asociados son Cheng et al., *PLoS ONE* 10(10) (2015) y 11(6) (2016). El dataset se descarga a través del espejo de Kaggle `ashkhagan/figshare-brain-tumor-dataset` y **no se redistribuye** en este repositorio.

## Estructura del repositorio

```
app/        demo en Streamlit (español/inglés) con el baseline, los modelos E y F
src/        datos, redes U-Net, funciones de pérdida y algoritmo genético
scripts/    entrenamiento, optimización, evaluación y gráficas
tests/      suite de pruebas (pytest)
docs/       guía técnica, figuras del README
```

## Más información

- **[docs/development.md](docs/development.md)** — guía técnica completa: instalación, reproducción de todos los resultados, opciones de entrenamiento, despliegue en Streamlit Cloud y ejecución en Kaggle/Colab.
- El **informe completo (memoria)** acompaña esta entrega como documento aparte.


## Licencia

El código de este repositorio se publica bajo la [licencia MIT](LICENSE). El dataset está bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) de sus autores originales y **no** se redistribuye aquí.

## Aviso médico

Este es un prototipo de investigación y docencia. **No** es un dispositivo médico, no ha sido validado clínicamente y nunca debe usarse para diagnosticar, tratar o descartar una enfermedad. Todas las cifras son experimentos reproducibles sobre un conjunto de datos público.
