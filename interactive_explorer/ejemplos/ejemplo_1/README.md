# Ejemplo 1: Helice de Embeddings Posicionales GPT-2

## Descripcion

Esta visualizacion muestra los **embeddings posicionales reales** de GPT-2, reducidos de 768 dimensiones a 3D mediante PCA. La estructura helicoidal emergente demuestra que el modelo aprendio una representacion geometrica coherente de las posiciones en una secuencia.

## Contenido

- `ejemplo_paper_1.html` - Visualizacion interactiva con graficos 2D y 3D
- `README.md` - Este documento

## Datos Utilizados

- **Fuente**: `model.transformer.wpe.weight` de GPT-2 (HuggingFace)
- **Dimensiones originales**: 1024 posiciones x 768 dimensiones
- **Reduccion**: PCA a 3 componentes (~90% varianza explicada)
- **Posiciones mostradas**: 512 (primeras posiciones)

---

## BUG CRITICO DOCUMENTADO: Discontinuidad en la Helice

### Fecha de Deteccion
27 de Enero de 2026

### Sintoma
La helice mostraba un **salto abrupto** visible - dos lineas separadas conectadas por una linea recta, en lugar de una curva continua.

### Capturas del Bug

El usuario observo que la helice tenia una discontinuidad evidente:
- La primera parte de la helice se veia correcta
- Habia un "salto" hacia una segunda seccion
- Las dos secciones no formaban una curva continua

### Causa Raiz

En el archivo HTML habia codigo **residual** que generaba datos sinteticos DESPUES de cargar los datos reales:

```javascript
// CODIGO PROBLEMATICO (ELIMINADO)
// Agregar mas puntos para completar la helice
for (let i = 64; i < 256; i++) {
    const t = i / 256 * Math.PI * 4;
    const r = 2.0 - i / 256 * 0.5;
    posPCAData.push({
        x: r * Math.cos(t) * 0.8,
        y: r * Math.sin(t) * 0.6,
        z: (i - 128) / 128 * 0.8
    });
}
```

Este codigo:
1. Tomaba los 512 puntos reales de GPT-2
2. **Agregaba 192 puntos sinteticos** con una formula matematica diferente
3. Creaba una mezcla de datos reales + sinteticos = discontinuidad visual

### Solucion Aplicada

Se elimino completamente el bloque de generacion sintetica:

```javascript
// CODIGO CORREGIDO
// Datos reales de GPT-2 (512 posiciones) - SIN datos sinteticos
// La helice debe ser continua, representando embeddings posicionales reales
```

### Commit de la Correccion
```
fbdd5fe - Fix: Remove synthetic data generation causing helix discontinuity
```

---

## REGLA DE ORO: Continuidad de la Helice

### Por que la helice DEBE ser continua

Los embeddings posicionales de GPT-2 representan como el modelo "entiende" la posicion de cada token en una secuencia. Estas son propiedades aprendidas, no inventadas:

1. **Posiciones cercanas = embeddings cercanos**: El modelo aprendio que la posicion 50 es similar a la posicion 51
2. **Estructura helicoidal**: La forma de helice emerge naturalmente del entrenamiento
3. **Sin saltos abruptos**: No hay razon fisica para que exista una discontinuidad entre posicion N y N+1

### Como verificar que los datos son correctos

1. **Fuente unica**: Los datos deben venir SOLO de `model.transformer.wpe.weight`
2. **Sin generacion sintetica**: NUNCA agregar puntos calculados con formulas matematicas
3. **Continuidad visual**: La helice debe verse como una curva suave sin saltos
4. **Archivo JSON de referencia**: Usar siempre `real_helix_data.json` como fuente de verdad

### Outlier conocido

La **posicion 1023** (ultima) tiene valores anomalos porque se usa muy raramente durante el entrenamiento. Este es el UNICO punto que puede parecer "saltado" y es un comportamiento real del modelo.

---

## Extraccion de Datos Reales

Para extraer embeddings posicionales reales de GPT-2:

```python
from transformers import GPT2LMHeadModel
from sklearn.decomposition import PCA
import json

# Cargar modelo
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Extraer embeddings posicionales
pos_embeddings = model.transformer.wpe.weight.detach().cpu().numpy()
# Shape: (1024, 768)

# Reducir a 3D con PCA
pca = PCA(n_components=3)
reduced = pca.fit_transform(pos_embeddings)

# Varianza explicada: ~90%
print(f"Varianza explicada: {sum(pca.explained_variance_ratio_):.2%}")

# Guardar como JSON
data = [{"x": round(p[0], 3), "y": round(p[1], 3), "z": round(p[2], 3)}
        for p in reduced]
with open('real_helix_data.json', 'w') as f:
    json.dump(data, f)
```

---

## Estructura de la Visualizacion

El archivo HTML incluye:

1. **Grafico de Atencion vs MLP**: Separacion geometrica de ambos tipos de capas
2. **Ejemplo de Token Processing**: Como se transforma "The quick brown fox..."
3. **Evolucion por Capas**: Slider interactivo mostrando cambios
4. **Helice 3D**: Visualizacion Three.js con rotacion automatica
5. **Navegacion**: Links a otras visualizaciones del proyecto

---

## Lecciones Aprendidas

1. **Siempre usar datos reales**: No mezclar datos extraidos del modelo con datos generados
2. **Verificar visualmente**: Si hay saltos, algo esta mal
3. **Documentar bugs**: Este README sirve como referencia futura
4. **Archivo JSON como fuente de verdad**: `real_helix_data.json` contiene los datos correctos

---

*Documentado como parte del proyecto Feature_Geometry_Visualization*
*Paper: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"*
