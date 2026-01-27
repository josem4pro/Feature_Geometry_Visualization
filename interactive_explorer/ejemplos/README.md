# Ejemplos Interactivos - Feature Geometry Visualization

Esta carpeta contiene ejemplos documentados del paper "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction".

## Estructura

Cada ejemplo tiene su propia subcarpeta con:
- **HTML interactivo**: Visualizacion completa con graficos 2D y 3D
- **README.md**: Documentacion, datos utilizados, y lecciones aprendidas

## Ejemplos Disponibles

| Carpeta | Descripcion | Datos |
|---------|-------------|-------|
| [ejemplo_1](./ejemplo_1/) | Helice de embeddings posicionales GPT-2 | 512 posiciones reales |

## Convencion de Nombres

- `ejemplo_paper_N.html` - Visualizacion interactiva del ejemplo N
- `README.md` - Documentacion del ejemplo, incluyendo bugs encontrados y soluciones

## Importante: Regla de Continuidad

Todas las visualizaciones de helice DEBEN mostrar curvas continuas.
Si hay saltos o discontinuidades, significa que hay un bug en los datos.

Ver [ejemplo_1/README.md](./ejemplo_1/README.md) para documentacion detallada de un bug critico encontrado y corregido.

---

*Proyecto: Feature_Geometry_Visualization*
