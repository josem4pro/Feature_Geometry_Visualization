# Inicio Rápido

## 3 Pasos para Visualizar

### Paso 1: Iniciar servidor
```bash
cd interactive_explorer
python3 -m http.server 8888
```

### Paso 2: Abrir en Chrome
```
http://localhost:8888/reporte_espanol_v2.html
```

### Paso 3: Si la hélice 3D no se ve
1. Chrome > Configuración > Sistema
2. Activar "Aceleración de hardware"
3. Reiniciar Chrome

---

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `reporte_espanol_v2.html` | Reporte completo en español |
| `helix_2d.html` | Hélice en 2D (sin WebGL) |
| `webgl_test.html` | Diagnóstico de WebGL |

---

## Comandos Útiles

```bash
# Generar nuevo reporte
python3 generate_html_report_es.py --output nuevo_reporte.html

# Test de WebGL
# Abrir: http://localhost:8888/webgl_test.html
```
