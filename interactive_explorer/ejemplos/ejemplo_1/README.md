# Ejemplo 1: Trayectoria de "The quick brown fox jumps over the lazy dog"

## Descripcion

Esta visualizacion muestra como **GPT-2 procesa una frase especifica**, capa por capa. Cada punto representa el estado interno del modelo en una capa, y la trayectoria muestra la evolucion del "pensamiento" desde la entrada hasta la salida.

## Contenido

- `ejemplo_paper_1.html` - Visualizacion interactiva con graficos 3D
- `fox_trajectory_data.json` - Datos extraidos de GPT-2
- `README.md` - Este documento

## Que muestra esta visualizacion?

### Trayectoria por Capas

GPT-2 tiene **13 capas** (embedding + 12 transformer layers). Cuando procesa "The quick brown fox jumps over the lazy dog":

1. **Capa 0 (Embedding)**: Los tokens se convierten en vectores de 768 dimensiones
2. **Capas 1-4**: Procesamiento sintactico - estructura gramatical
3. **Capas 5-8**: Integracion semantica - significado contextual
4. **Capas 9-12**: Preparacion para prediccion del siguiente token

### Posiciones de Tokens

Cada token termina en una posicion diferente en el espacio latente, influenciado por:
- Su significado propio
- El contexto de toda la frase
- Las relaciones con otros tokens

## Datos Tecnicos

| Metrica | Valor |
|---------|-------|
| Modelo | GPT-2 (124M parametros) |
| Frase | "The quick brown fox jumps over the lazy dog" |
| Tokens | 9 |
| Capas | 13 (embedding + 12 transformer) |
| Dimensiones originales | 768 |
| Reduccion | PCA a 3D |
| Varianza explicada | ~99% |

## Extraccion de Datos

Los hidden states se extraen asi:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

sentence = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# hidden_states es una tupla de 13 tensores
# Cada tensor tiene shape (batch, seq_len, 768)
hidden_states = outputs.hidden_states

# Para la trayectoria: promedio de tokens por capa
for layer_idx, layer_hidden in enumerate(hidden_states):
    layer_mean = layer_hidden.mean(dim=1)  # (1, 768)
    # Este vector representa el "estado" de toda la frase en esta capa

# Para posiciones de tokens: ultima capa
final_layer = hidden_states[-1]  # (1, seq_len, 768)
# Cada token tiene su propia representacion final
```

## Hallazgos Clave

### 1. El Salto entre Capas 2-3

La trayectoria muestra un "salto" notable entre las capas 2 y 3. Esto indica una transicion del procesamiento lexico (palabras individuales) al procesamiento semantico (significado integrado).

### 2. Convergencia en Capas Finales

En las capas 9-12, la trayectoria se mueve de manera mas consistente hacia la "zona de prediccion", donde el modelo prepara su output.

### 3. Tokens en Diferentes Posiciones

Observando las posiciones finales de los tokens:
- **"fox"** y **"dog"** (sustantivos animales) estan relativamente cerca
- **"jumps"** (verbo de accion) esta separado
- **"The"** y **"the"** (articulos) estan en posiciones similares

## Comparacion con Ejemplo 2

| Aspecto | Ejemplo 1 | Ejemplo 2 |
|---------|-----------|-----------|
| Frases | 1 ("fox jumps over dog") | 2 (fox/dog intercambiados) |
| Proposito | Ver trayectoria unica | Comparar dos trayectorias |
| Pregunta | "Como procesa GPT-2 esta frase?" | "Como cambia al intercambiar roles?" |

Ver [ejemplo_2](../ejemplo_2/) para la comparacion semantica.

---

## Nota sobre Version Anterior

La version original de este ejemplo mostraba **embeddings posicionales** (la tabla `wpe.weight`), que son independientes del texto procesado. Esta version corregida muestra **hidden states**, que son las representaciones reales generadas cuando GPT-2 procesa esta frase especifica.

---

*Documentado como parte del proyecto Feature_Geometry_Visualization*
*Paper: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"*
