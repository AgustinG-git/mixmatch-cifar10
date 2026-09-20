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

## Diferencias con el paper (importante)

- **Presupuesto de entrenamiento:** el paper entrena ~1M de steps; aquí 20 épocas × 1024 steps (~20k) por límites de cómputo. Los resultados no son comparables con los del paper.
- **Rampa de λ_U:** el paper usa 16.000 steps; aquí se reduce a 4096 (20% del entrenamiento) para adaptarla al presupuesto.
- **Weight decay:** el paper indica 0.0004 por actualización; el código oficial aplica `0.02 * lr` (4e-5) solo a los kernels, que es lo que se usa aquí.

## Resultados

Pendiente de completar tras el reentrenamiento.

| Etiquetas | Épocas × steps | Error de test (EMA) | Paper (1M steps) |
|---|---|---|---|
| 250 | 20 × 1024 | por rellenar | 11.08% |
| 4000 | 20 × 1024 | por rellenar | 6.24% |

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook mixmatch_cifar10.ipynb
```

CIFAR-10 se descarga automáticamente con `tf.keras.datasets`. Se recomienda GPU.
