# MixMatch en CIFAR-10 (TensorFlow / Keras)

Reimplementación desde cero de **MixMatch** ([Berthelot et al., 2019](https://arxiv.org/abs/1905.02249)), aprendizaje semi-supervisado con pocas etiquetas, sobre CIFAR-10.

## Qué incluye

- WideResNet-28-2 con bloques residuales pre-activados.
- Etiquetas artificiales: K augmentaciones, promedio y sharpening (T = 0.5).
- MixUp modificado (λ' = max(λ, 1 - λ)) entre datos etiquetados y no etiquetados.
- Pérdida L_X (entropía cruzada) + λ_U · L_U (error cuadrático) con rampa lineal de λ_U.
- EMA de los pesos (incluidas las estadísticas de BatchNorm) y evaluación con el modelo EMA.
- Visualizaciones del efecto del sharpening y del MixUp.

## Hiperparámetros

Adam con lr 0.002, batch 64, K = 2, T = 0.5, α = 0.75, λ_U = 75, EMA 0.999.

## Resultados

Error de test (modelo EMA) en los 10.000 ejemplos de test de CIFAR-10, tras la última época. Una sola ejecución (semilla 0), en una GPU Tesla T4 (~5 min por época).

| Etiquetas | Épocas × steps | Error de test | Paper (~1M steps) |
|---|---|---|---|
| 250 | 20 × 1024 | 31.31% | 11.08% |
| 4000 | 20 × 1024 | 11.13% | 6.24% |

- Con 4000 etiquetas la curva se aplana hacia el final; con 250 sigue bajando al terminar, así que ese resultado está limitado por el presupuesto de entrenamiento.
- No hay early stopping ni selección de checkpoints sobre el test: se reporta el error de la última época.
- No se ha comprobado con un entrenamiento más largo que el presupuesto explique toda la diferencia con el paper.

## Diferencias con el paper

- **Presupuesto de entrenamiento:** el paper entrena ~1M de steps; aquí 20 épocas × 1024 steps (~20k) por límites de cómputo. Los resultados no son comparables con los del paper.
- **Rampa de λ_U:** el paper usa 16.000 steps; aquí se reduce a 4096 (20% del entrenamiento) para adaptarla al presupuesto.
- **Weight decay:** el paper indica 0.0004 por actualización; el código oficial aplica `0.02 * lr` (4e-5) solo a los kernels, que es lo que se usa aquí.

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook mixmatch_cifar10.ipynb
```

CIFAR-10 se descarga automáticamente con `tf.keras.datasets`. Se recomienda GPU.
