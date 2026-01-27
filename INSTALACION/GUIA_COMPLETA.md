# Guía de Instalación - Explorador de Geometría del Espacio Latente

## Requisitos del Sistema

### Software Necesario
- **Python 3.8+** con los siguientes paquetes:
  - `torch` (PyTorch)
  - `transformers` (Hugging Face)
  - `numpy`
  - `scikit-learn`
  - `pandas`

### Requisitos del Navegador
- **Google Chrome** (recomendado) o Firefox
- **WebGL habilitado** (ver sección de solución de problemas)

---

## Instalación Rápida

### 1. Clonar/Descargar el repositorio
```bash
cd /ruta/a/tu/directorio
git clone <url-del-repositorio>
cd Feature_Geometry_Visualization/interactive_explorer
```

### 2. Verificar dependencias Python
```bash
python3 -c "import torch; import transformers; import numpy; import sklearn; print('OK')"
```

### 3. Iniciar servidor HTTP local
```bash
cd interactive_explorer
python3 -m http.server 8888
```

### 4. Abrir en el navegador
```
http://localhost:8888/reporte_espanol_v2.html
```

---

## Generación de Reportes

### Generar reporte en español
```bash
python3 generate_html_report_es.py --output mi_reporte.html
```

### Generar reporte en inglés
```bash
python3 generate_html_report.py --output my_report.html
```

### Opciones disponibles
- `--model`: Modelo a usar (default: "gpt2")
- `--output`: Nombre del archivo de salida
- `--max-length`: Longitud máxima de secuencia (default: 64)

---

## Estructura de Archivos

```
interactive_explorer/
├── latent_extractor.py      # Extracción de estados latentes de GPT-2
├── visualizer.py            # Visualizaciones con Plotly (opcional)
├── generate_html_report.py  # Generador de reportes (inglés)
├── generate_html_report_es.py # Generador de reportes (español)
├── app.py                   # Dashboard Streamlit (opcional)
├── requirements.txt         # Dependencias Python
├── setup.sh                 # Script de configuración
├── helix_2d.html           # Visualización 2D de la hélice
├── helix_test.html         # Test de hélice 3D
├── webgl_test.html         # Diagnóstico de WebGL
└── INSTALACION/            # Esta documentación
```

---

## Solución de Problemas

### Problema: "WebGL no disponible"

**Síntoma**: La visualización 3D muestra una caja negra vacía.

**Diagnóstico**:
1. Abrir `http://localhost:8888/webgl_test.html`
2. Si dice "ERROR: WebGL no disponible", seguir los pasos abajo.

**Solución para Chrome**:
1. Abrir `chrome://settings/system`
2. Activar **"Usar aceleración de hardware cuando esté disponible"**
3. Reiniciar Chrome completamente (cerrar todas las ventanas)
4. Volver a probar

**Solución alternativa**:
Si WebGL no puede habilitarse, usar la visualización 2D alternativa:
- `http://localhost:8888/helix_2d.html`

### Problema: "Error creating WebGL context"

**Causa**: Los drivers de GPU no soportan WebGL o están desactualizados.

**Soluciones**:
1. Actualizar drivers de la tarjeta gráfica
2. En Chrome, ir a `chrome://flags` y buscar "WebGL", asegurarse que esté habilitado
3. Probar con Firefox como alternativa

### Problema: ES Modules no cargan (Three.js)

**Síntoma**: La página carga pero la visualización 3D no aparece.

**Causa**: Los ES modules de JavaScript requieren un servidor HTTP.

**Solución**:
- **NO abrir el archivo directamente** (file://)
- **SÍ usar un servidor HTTP**:
```bash
python3 -m http.server 8888
# Luego abrir http://localhost:8888/archivo.html
```

### Problema: Modelo GPT-2 no descarga

**Síntoma**: Error al cargar el modelo.

**Solución**:
```bash
# Verificar conexión a internet
# El modelo se descarga automáticamente de Hugging Face
python3 -c "from transformers import GPT2Model; GPT2Model.from_pretrained('gpt2')"
```

---

## Tecnologías Utilizadas

### Backend (Python)
- **PyTorch**: Ejecución del modelo GPT-2
- **Transformers**: Carga del modelo pre-entrenado
- **scikit-learn**: PCA para reducción de dimensionalidad

### Frontend (HTML/JavaScript)
- **Chart.js**: Gráficos 2D interactivos (barras, scatter)
- **Three.js**: Visualización 3D de la hélice
- **Canvas 2D**: Alternativa sin WebGL

### CDN utilizados
- Chart.js: `https://cdn.jsdelivr.net/npm/chart.js`
- Three.js: `https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`
- OrbitControls: `https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js`

---

## Notas Técnicas

### Por qué es necesario WebGL
Three.js utiliza WebGL para renderizar gráficos 3D acelerados por GPU.
Sin WebGL, el navegador no puede crear el contexto de renderizado 3D.

### Por qué es necesario un servidor HTTP
Los ES modules de JavaScript tienen restricciones de seguridad (CORS) que
impiden cargarlos desde el protocolo `file://`. Un servidor HTTP local
resuelve este problema.

### Extracción de estados latentes
El módulo `latent_extractor.py` usa hooks de PyTorch para capturar
las activaciones intermedias de GPT-2 en 4 puntos por bloque transformer:
1. Salida de atención (antes de residual)
2. Después de sumar residual de atención
3. Salida de MLP (antes de residual)
4. Después de sumar residual de MLP

---

## Contacto y Recursos

- Paper original: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"
- arXiv: https://arxiv.org/abs/2511.21594
