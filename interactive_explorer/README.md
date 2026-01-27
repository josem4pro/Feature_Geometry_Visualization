# Interactive LLM Latent Space Geometry Explorer

This is a companion visualization tool based on the paper:

**"Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"**
by Ning, Rangaraju & Kuo (2025)

## Key Findings You Can Explore

### 1. Attention vs MLP Separation
The paper discovered that **attention and MLP component outputs occupy geometrically distinct regions** in latent space. Blue points (attention) cluster separately from red points (MLP).

### 2. High Norm at Position 0
Latent states at sequence position 0 have **significantly higher norms** than other positions. This acts as a learned bias term in the model.

### 3. Helical Positional Embeddings
GPT-2's learned positional embeddings form a **helical structure** in high-dimensional space when visualized through PCA.

### 4. Layer-wise Evolution
Representations **evolve gradually through layers**, with clear geometric progression from early to late layers.

## Quick Start

### Installation

```bash
cd interactive_explorer
pip install -r requirements.txt
```

### Option 1: Streamlit Dashboard (Recommended)

Launch the interactive web dashboard:

```bash
streamlit run app.py
```

This opens a browser with:
- Model selection (GPT-2 variants)
- Custom text input
- 6 interactive visualization tabs
- Real-time exploration

### Option 2: Generate Static HTML

Create all visualizations as interactive HTML files:

```bash
python generate_visualizations.py --model gpt2 --output ./my_visualizations
```

Then open the HTML files in your browser.

## Visualizations

| Tab | What it Shows |
|-----|---------------|
| **Layer Evolution** | 3D scatter showing how latent states evolve through layers |
| **Attention vs MLP** | 2D separation between attention and MLP outputs |
| **Position 0 Norm** | Analysis of the high-norm phenomenon at position 0 |
| **Positional Helix** | 3D view of GPT-2's helical positional embeddings |
| **Layer Norms** | Bar chart of mean norms per layer |
| **Animation** | Animated view of layer-by-layer evolution |

## Architecture

```
interactive_explorer/
├── app.py                     # Streamlit dashboard
├── latent_extractor.py        # GPT-2 latent state extraction
├── visualizer.py              # Plotly visualizations
├── generate_visualizations.py # CLI for static HTML generation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## How It Works

1. **Extraction** (`latent_extractor.py`):
   - Loads GPT-2 using HuggingFace Transformers
   - Registers forward hooks to capture activations
   - Captures 4 points per transformer block:
     - Pre-add attention output
     - Post-add attention (after residual)
     - Pre-add MLP output
     - Post-add MLP (after residual)

2. **Processing**:
   - Optional conversion to unit vectors
   - Optional averaging over samples/sequence
   - PCA dimensionality reduction

3. **Visualization** (`visualizer.py`):
   - Plotly-based interactive charts
   - Color coding: blue=attention, red=MLP
   - Gradients indicate layer depth

## Memory Requirements

| Model | Parameters | Approx. Memory |
|-------|-----------|----------------|
| gpt2 | 124M | ~2 GB |
| gpt2-medium | 355M | ~4 GB |
| gpt2-large | 774M | ~6 GB |

## References

- [Paper (arXiv)](https://arxiv.org/abs/2511.21594)
- [Original Repository](https://github.com/Vainateya/Feature_Geometry_Visualization)

## License

Same as the original repository.
