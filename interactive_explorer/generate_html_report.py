#!/usr/bin/env python3
"""
Generate HTML Report with Interactive Visualizations

Creates a self-contained HTML file with embedded visualizations
using Chart.js (loaded from CDN). No matplotlib/plotly needed.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, str(Path(__file__).parent))

from latent_extractor import GPT2LatentExtractor, compute_norms


def to_json_list(arr):
    """Convert numpy array to JSON-serializable list."""
    return arr.tolist() if isinstance(arr, np.ndarray) else arr


def generate_html_report(model_name: str = "gpt2", max_length: int = 64, output_path: str = "report.html"):
    """Generate interactive HTML report."""

    print("=" * 60)
    print("LLM Latent Space Geometry - HTML Report Generator")
    print("=" * 60)

    # Load model and extract data
    print(f"\nLoading {model_name}...")
    extractor = GPT2LatentExtractor(model_name)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was darkness, and then there was light.",
        "Machine learning models can discover complex patterns in data.",
        "The theory of relativity changed our understanding of space and time.",
        "Natural language processing enables computers to understand human language.",
        "Deep neural networks learn hierarchical representations of data.",
    ]

    print(f"Extracting latent states from {len(texts)} texts...")
    latent_states, metas, token_ids, token_strings = extractor.extract(texts, max_length=max_length)
    print(f"Shape: {latent_states.shape}")

    n_blocks = extractor.n_layers

    # Compute statistics
    norms = compute_norms(latent_states)
    pos0_mean = float(norms['full'][:, :, 0].mean())
    other_mean = float(norms['full'][:, :, 1:].mean())

    # Prepare data for visualizations
    print("Processing data for visualizations...")

    # 1. Norm by position
    norm_by_pos = norms['by_position'].tolist()

    # 2. Norm by layer
    norm_by_layer = norms['by_layer'].tolist()
    layer_colors = []
    layer_labels = []
    for meta in metas:
        if meta.layer_type == 'embed':
            layer_colors.append('rgba(50, 200, 50, 0.8)')
            layer_labels.append('Embed')
        elif meta.layer_type == 'final_norm':
            layer_colors.append('rgba(200, 200, 50, 0.8)')
            layer_labels.append('Final')
        elif meta.is_attention:
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            layer_colors.append(f'rgba({int(50*intensity)}, {int(100*intensity)}, {int(200+55*(1-intensity))}, 0.8)')
            layer_labels.append(f'B{meta.block_num} Attn' if meta.is_pre_add else f'B{meta.block_num} +Attn')
        else:
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            layer_colors.append(f'rgba({int(200+55*(1-intensity))}, {int(50*intensity)}, {int(50*intensity)}, 0.8)')
            layer_labels.append(f'B{meta.block_num} MLP' if meta.is_pre_add else f'B{meta.block_num} +MLP')

    # 3. Layer evolution PCA
    data = latent_states[:, :, 1:, :].copy()
    layer_norms = np.linalg.norm(data, axis=-1, keepdims=True)
    layer_norms = np.where(layer_norms == 0, 1, layer_norms)
    data = data / layer_norms
    data = data.mean(axis=(1, 2))  # (n_layers, hidden_dim)

    pca = PCA(n_components=2)
    layer_pca = pca.fit_transform(data)
    layer_pca_data = [
        {'x': float(layer_pca[i, 0]), 'y': float(layer_pca[i, 1]), 'label': layer_labels[i]}
        for i in range(len(layer_pca))
    ]

    # 4. Attention vs MLP separation
    attn_indices = [i for i, m in enumerate(metas) if m.is_pre_add and m.is_attention]
    mlp_indices = [i for i, m in enumerate(metas) if m.is_pre_add and m.is_mlp]

    attn_data = latent_states[attn_indices, :, 1:, :].mean(axis=1).reshape(-1, latent_states.shape[-1])
    mlp_data = latent_states[mlp_indices, :, 1:, :].mean(axis=1).reshape(-1, latent_states.shape[-1])

    # Unit vectors
    attn_data = attn_data / np.linalg.norm(attn_data, axis=-1, keepdims=True)
    mlp_data = mlp_data / np.linalg.norm(mlp_data, axis=-1, keepdims=True)

    combined = np.vstack([attn_data, mlp_data])
    pca2 = PCA(n_components=2)
    combined_pca = pca2.fit_transform(combined)

    attn_pca = combined_pca[:len(attn_data)]
    mlp_pca = combined_pca[len(attn_data):]

    # Sample for visualization (too many points)
    n_sample = min(500, len(attn_pca))
    attn_sample = attn_pca[np.random.choice(len(attn_pca), n_sample, replace=False)]
    mlp_sample = mlp_pca[np.random.choice(len(mlp_pca), n_sample, replace=False)]

    attn_pca_data = [{'x': float(p[0]), 'y': float(p[1])} for p in attn_sample]
    mlp_pca_data = [{'x': float(p[0]), 'y': float(p[1])} for p in mlp_sample]

    # 5. Positional embeddings helix
    pos_emb = extractor.get_positional_embeddings()
    pca3 = PCA(n_components=3)
    pos_pca = pca3.fit_transform(pos_emb[:256])  # First 256 positions
    pos_pca_data = [
        {'x': float(pos_pca[i, 0]), 'y': float(pos_pca[i, 1]), 'z': float(pos_pca[i, 2])}
        for i in range(len(pos_pca))
    ]

    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Latent Space Geometry Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #4fc3f7;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #90a4ae;
            font-size: 1.1em;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
        }}
        .stat-box {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px 30px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4fc3f7;
        }}
        .stat-label {{
            color: #90a4ae;
            margin-top: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .chart-title {{
            color: #4fc3f7;
            font-size: 1.2em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .chart-desc {{
            color: #90a4ae;
            font-size: 0.9em;
            margin-bottom: 15px;
            line-height: 1.4;
        }}
        canvas {{
            width: 100% !important;
            height: 350px !important;
        }}
        #helix-container {{
            width: 100%;
            height: 400px;
            border-radius: 8px;
            overflow: hidden;
        }}
        .finding {{
            background: rgba(79, 195, 247, 0.1);
            border-left: 4px solid #4fc3f7;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .finding h3 {{
            color: #4fc3f7;
            margin: 0 0 10px 0;
        }}
        .info {{
            max-width: 1600px;
            margin: 30px auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Latent Space Geometry Explorer</h1>
        <p class="subtitle">Interactive visualizations of Transformer internal representations</p>
        <p class="subtitle">Model: {model_name.upper()} | Layers: {extractor.n_layers} | Hidden: {extractor.hidden_size}</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">{pos0_mean:.0f}</div>
            <div class="stat-label">Position 0 Norm</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{other_mean:.0f}</div>
            <div class="stat-label">Other Positions Norm</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{pos0_mean/other_mean:.1f}x</div>
            <div class="stat-label">Ratio</div>
        </div>
    </div>

    <div class="info">
        <div class="finding">
            <h3>Key Finding: High Norm at Position 0</h3>
            <p>Latent states at the first sequence position have significantly higher norms than other positions.
            This acts as a learned bias term in the model.</p>
        </div>
    </div>

    <div class="grid">
        <div class="chart-container">
            <div class="chart-title">Norm by Sequence Position</div>
            <div class="chart-desc">Mean L2 norm of latent states at each position. Notice the dramatic spike at position 0.</div>
            <canvas id="normByPos"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Norm by Layer</div>
            <div class="chart-desc">Mean norm increases through layers. Blue = Attention, Red = MLP.</div>
            <canvas id="normByLayer"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Layer Evolution (PCA)</div>
            <div class="chart-desc">How latent states evolve through layers projected to 2D. The trajectory shows progression from embedding to final norm.</div>
            <canvas id="layerPCA"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Attention vs MLP Separation</div>
            <div class="chart-desc"><strong>Key Finding:</strong> Attention (blue) and MLP (red) outputs occupy geometrically distinct regions in latent space.</div>
            <canvas id="attnMLP"></canvas>
        </div>

        <div class="chart-container" style="grid-column: span 2;">
            <div class="chart-title">Positional Embeddings (3D Helix)</div>
            <div class="chart-desc"><strong>Key Finding:</strong> GPT-2's learned positional embeddings form a helical structure in high-dimensional space. Drag to rotate, scroll to zoom.</div>
            <div id="helix-container"></div>
        </div>
    </div>

    <div class="info">
        <div class="finding">
            <h3>About This Visualization</h3>
            <p>Based on the paper: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"
            by Ning, Rangaraju & Kuo (2025). <a href="https://arxiv.org/abs/2511.21594" style="color:#4fc3f7">arXiv:2511.21594</a></p>
        </div>
    </div>

    <script>
        // Data
        const normByPosData = {json.dumps(norm_by_pos)};
        const normByLayerData = {json.dumps(norm_by_layer)};
        const layerColors = {json.dumps(layer_colors)};
        const layerLabels = {json.dumps(layer_labels)};
        const layerPCAData = {json.dumps(layer_pca_data)};
        const attnPCAData = {json.dumps(attn_pca_data)};
        const mlpPCAData = {json.dumps(mlp_pca_data)};
        const posPCAData = {json.dumps(pos_pca_data)};

        // Chart defaults
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

        // 1. Norm by Position
        new Chart(document.getElementById('normByPos'), {{
            type: 'bar',
            data: {{
                labels: normByPosData.map((_, i) => i),
                datasets: [{{
                    data: normByPosData,
                    backgroundColor: normByPosData.map((_, i) => i === 0 ? 'rgba(244, 67, 54, 0.8)' : 'rgba(79, 195, 247, 0.6)'),
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Mean Norm' }} }},
                    x: {{ title: {{ display: true, text: 'Sequence Position' }} }}
                }}
            }}
        }});

        // 2. Norm by Layer
        new Chart(document.getElementById('normByLayer'), {{
            type: 'bar',
            data: {{
                labels: layerLabels,
                datasets: [{{
                    data: normByLayerData,
                    backgroundColor: layerColors,
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Mean Norm' }} }},
                    x: {{ ticks: {{ maxRotation: 90, minRotation: 45, font: {{ size: 8 }} }} }}
                }}
            }}
        }});

        // 3. Layer PCA
        new Chart(document.getElementById('layerPCA'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    data: layerPCAData,
                    backgroundColor: layerColors,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }}, {{
                    data: layerPCAData,
                    type: 'line',
                    borderColor: 'rgba(255,255,255,0.3)',
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => ctx.raw.label || ''
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'PC1' }} }},
                    y: {{ title: {{ display: true, text: 'PC2' }} }}
                }}
            }}
        }});

        // 4. Attention vs MLP
        new Chart(document.getElementById('attnMLP'), {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Attention',
                    data: attnPCAData,
                    backgroundColor: 'rgba(79, 195, 247, 0.5)',
                    pointRadius: 3
                }}, {{
                    label: 'MLP',
                    data: mlpPCAData,
                    backgroundColor: 'rgba(244, 67, 54, 0.5)',
                    pointRadius: 3
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'PC1' }} }},
                    y: {{ title: {{ display: true, text: 'PC2' }} }}
                }}
            }}
        }});

    </script>

    <!-- 5. 3D Helix with Three.js (ES Modules) -->
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const posPCAData = {json.dumps(pos_pca_data)};

        const container = document.getElementById('helix-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        const camera = new THREE.PerspectiveCamera(50, width/height, 0.1, 1000);
        camera.position.z = 5;

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);

        // Create helix points
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        const data = posPCAData;
        const scale = 0.02;

        for (let i = 0; i < data.length; i++) {{
            positions.push(data[i].x * scale, data[i].y * scale, data[i].z * scale);
            // Color gradient from blue to yellow
            const t = i / data.length;
            colors.push(0.3 + t * 0.7, 0.3 + t * 0.4, 0.9 - t * 0.6);
        }}

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{
            size: 0.08,
            vertexColors: true
        }});

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Add connecting line
        const lineGeom = new THREE.BufferGeometry();
        lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const lineMat = new THREE.LineBasicMaterial({{ color: 0x888888, opacity: 0.6, transparent: true }});
        const line = new THREE.Line(lineGeom, lineMat);
        scene.add(line);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            const h = container.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});
    </script>
</body>
</html>'''

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"\n{'='*60}")
    print("Report generated successfully!")
    print(f"{'='*60}")
    print(f"\nOutput: {output_path}")
    print("\nOpen in your web browser to explore the visualizations!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", default="report.html")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    generate_html_report(args.model, args.max_length, args.output)
