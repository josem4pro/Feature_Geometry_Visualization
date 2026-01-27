# Ejemplo 2: Comparación Semántica - Fox vs Dog

## Descripción

Esta visualización compara cómo GPT-2 procesa dos frases casi idénticas donde se intercambian los roles de "fox" y "dog":

1. **Frase 1**: "The quick brown fox jumps over the lazy dog"
2. **Frase 2**: "The quick brown dog jumps over the lazy fox"

La pregunta clave: ¿Cómo cambia la geometría del espacio latente cuando el "que salta" y el "saltado" intercambian roles?

## Contenido

- `ejemplo_paper_2.html` - Visualización interactiva con tres vistas 3D
- `sentence_comparison_data.json` - Datos extraídos de GPT-2
- `README.md` - Este documento

## Datos Utilizados

### Extracción de Hidden States

A diferencia del Ejemplo 1 (embeddings posicionales), aquí extraemos **hidden states** - las representaciones internas que GPT-2 genera al procesar texto específico.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sklearn.decomposition import PCA
import numpy as np

# Cargar modelo con output_hidden_states=True
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown dog jumps over the lazy fox"
]

all_layer_means = []
all_token_final = []

for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states: tuple de 13 tensores (embedding + 12 layers)
    hidden_states = outputs.hidden_states

    # Para trayectoria por capas: promedio de tokens por capa
    all_layers = torch.stack(hidden_states).squeeze(1).numpy()
    layer_means = all_layers.mean(axis=1)  # (13, 768)
    all_layer_means.append(layer_means)

    # Para posición de tokens: última capa
    token_final = hidden_states[-1].squeeze(0).numpy()  # (n_tokens, 768)
    all_token_final.append(token_final)

# PCA conjunto para comparación justa
combined_layers = np.vstack(all_layer_means)
pca_layers = PCA(n_components=3)
reduced_layers = pca_layers.fit_transform(combined_layers)
# Varianza explicada: ~99%
```

### Estadísticas de los Datos

| Métrica | Valor |
|---------|-------|
| Modelo | GPT-2 (124M parámetros) |
| Capas | 13 (embedding + 12 transformer) |
| Tokens por frase | 9 |
| Dimensiones originales | 768 |
| Dimensiones reducidas | 3 (PCA) |
| Varianza explicada (capas) | 98.99% |
| Varianza explicada (tokens) | 96.62% |

### Tokenización

Ambas frases se tokenizan en 9 tokens:

| Posición | Frase 1 | Frase 2 |
|----------|---------|---------|
| 0 | The | The |
| 1 | quick | quick |
| 2 | brown | brown |
| **3** | **fox** | **dog** |
| 4 | jumps | jumps |
| 5 | over | over |
| 6 | the | the |
| 7 | lazy | lazy |
| **8** | **dog** | **fox** |

Las diferencias están en posiciones 3 y 8 - exactamente donde fox y dog intercambian lugares.

---

## Visualizaciones Incluidas

### 1. Trayectorias por Capas (Vista Separada)

Dos gráficos 3D lado a lado mostrando:
- **Eje de color**: Progresión desde capa 0 (embedding) hasta capa 12 (salida)
- **Forma**: Cómo la representación de toda la frase evoluciona a través del modelo

### 2. Trayectorias Combinadas

Un solo gráfico 3D con ambas trayectorias superpuestas:
- **Azul**: Frase 1 (fox jumps over dog)
- **Rojo**: Frase 2 (dog jumps over fox)
- **Observación clave**: Las trayectorias comienzan similares y divergen en capas intermedias

### 3. Posiciones de Tokens

Visualización de dónde termina cada token en el espacio latente después de todas las capas:
- Permite ver la "distancia semántica" entre tokens
- Los tokens idénticos (The, quick, brown, etc.) aparecen en posiciones similares
- Los tokens intercambiados (fox/dog) muestran patrones interesantes

---

## Hallazgos Clave

### 1. Divergencia Gradual

Las trayectorias de ambas frases:
- **Capas 0-3**: Muy similares (procesamiento léxico básico)
- **Capas 4-8**: Comienzan a diverger (integración semántica)
- **Capas 9-12**: Máxima diferencia (representación final del significado)

### 2. El Rol Semántico Importa

Aunque "fox" y "dog" son palabras similares (ambos animales), su **rol en la oración** cambia completamente cómo GPT-2 las representa:
- Fox como sujeto activo vs. Fox como objeto pasivo
- Dog como sujeto activo vs. Dog como objeto pasivo

### 3. Contexto es Todo

La posición final de cada token no depende solo del token en sí, sino de toda la oración. El mismo token "fox" termina en lugares diferentes dependiendo de si es "el que salta" o "sobre quien se salta".

---

## Diferencias con Ejemplo 1

| Aspecto | Ejemplo 1 | Ejemplo 2 |
|---------|-----------|-----------|
| Datos | Embeddings posicionales | Hidden states |
| Fuente | `wpe.weight` | Forward pass |
| Representa | Cómo GPT-2 entiende posición | Cómo GPT-2 procesa texto específico |
| Puntos | 512-1024 posiciones | 13 capas × 2 frases |
| Forma | Hélice única | Dos trayectorias comparadas |

---

## Código de Extracción Completo

El archivo `sentence_comparison_data.json` fue generado con este proceso:

```python
import json
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA

def extract_sentence_comparison():
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()

    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown dog jumps over the lazy fox"
    ]

    all_layer_means = []
    all_token_final = []
    all_tokens = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        all_tokens.append(tokens)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        all_layers = torch.stack(hidden_states).squeeze(1).numpy()
        layer_means = all_layers.mean(axis=1)
        all_layer_means.append(layer_means)

        token_final = hidden_states[-1].squeeze(0).numpy()
        all_token_final.append(token_final)

    # PCA conjunto
    combined_layers = np.vstack(all_layer_means)
    pca_layers = PCA(n_components=3)
    reduced_layers = pca_layers.fit_transform(combined_layers)

    combined_tokens = np.vstack(all_token_final)
    pca_tokens = PCA(n_components=3)
    reduced_tokens = pca_tokens.fit_transform(combined_tokens)

    # Formatear resultado
    n_layers = all_layer_means[0].shape[0]
    n_tokens_1 = all_token_final[0].shape[0]
    n_tokens_2 = all_token_final[1].shape[0]

    result = {
        "sentence_1": {
            "text": sentences[0],
            "tokens": all_tokens[0],
            "layer_trajectory": [
                {"x": float(round(p[0], 3)), "y": float(round(p[1], 3)), "z": float(round(p[2], 3))}
                for p in reduced_layers[:n_layers]
            ],
            "token_positions": [
                {"x": float(round(p[0], 3)), "y": float(round(p[1], 3)), "z": float(round(p[2], 3))}
                for p in reduced_tokens[:n_tokens_1]
            ]
        },
        "sentence_2": {
            "text": sentences[1],
            "tokens": all_tokens[1],
            "layer_trajectory": [
                {"x": float(round(p[0], 3)), "y": float(round(p[1], 3)), "z": float(round(p[2], 3))}
                for p in reduced_layers[n_layers:]
            ],
            "token_positions": [
                {"x": float(round(p[0], 3)), "y": float(round(p[1], 3)), "z": float(round(p[2], 3))}
                for p in reduced_tokens[n_tokens_1:]
            ]
        },
        "metadata": {
            "model": "gpt2",
            "n_layers": n_layers,
            "variance_explained_layers": float(round(sum(pca_layers.explained_variance_ratio_) * 100, 2)),
            "variance_explained_tokens": float(round(sum(pca_tokens.explained_variance_ratio_) * 100, 2))
        }
    }

    with open('sentence_comparison_data.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result

# Ejecutar
data = extract_sentence_comparison()
```

---

## Lecciones Aprendidas

1. **PCA conjunto**: Para comparar dos frases, aplicar la misma transformación PCA (entrenada con datos combinados) garantiza que las coordenadas sean comparables.

2. **Hidden states vs Embeddings**: Los hidden states capturan el procesamiento contextual, no solo la información posicional.

3. **Serialización JSON**: Numpy float32 no es serializable directamente - usar `float()` para convertir.

4. **Trayectorias como narrativa**: Visualizar la evolución por capas cuenta la "historia" de cómo el modelo construye el significado.

---

*Documentado como parte del proyecto Feature_Geometry_Visualization*
*Paper: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"*
