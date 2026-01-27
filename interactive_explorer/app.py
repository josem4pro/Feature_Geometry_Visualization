"""
Interactive LLM Latent Space Geometry Explorer

A Streamlit dashboard to visualize and explore the geometric patterns
in LLM latent spaces, based on the paper:
"Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"

Key findings you can explore:
1. Layer-wise evolution of representations
2. Attention vs MLP geometric separation
3. High norm at position 0 phenomenon
4. Helical structure of positional embeddings
"""

import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from latent_extractor import GPT2LatentExtractor, compute_norms
from visualizer import (
    LatentSpaceVisualizer,
    create_animated_layer_evolution
)

# Page config
st.set_page_config(
    page_title="LLM Latent Space Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .finding-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stPlotlyChart {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str):
    """Load and cache the GPT-2 model."""
    return GPT2LatentExtractor(model_name)


@st.cache_data
def extract_latents(_extractor, texts: tuple, max_length: int):
    """Extract and cache latent states."""
    return _extractor.extract(list(texts), max_length=max_length)


@st.cache_data
def get_positional_embeddings(_extractor):
    """Get and cache positional embeddings."""
    return _extractor.get_positional_embeddings()


def main():
    # Header
    st.markdown('<div class="main-header">🧠 LLM Latent Space Geometry Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Visualizing the internal representations of Transformer models</div>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    st.sidebar.title("Configuration")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gpt2", "gpt2-medium", "gpt2-large"],
        index=0,
        help="Larger models have more layers and hidden dimensions"
    )

    # Load model
    with st.spinner(f"Loading {model_name}..."):
        extractor = load_model(model_name)

    st.sidebar.success(f"Model loaded: {extractor.n_layers} layers, {extractor.hidden_size} dim")

    # Text input options
    st.sidebar.subheader("Input Text")

    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Sample Texts", "Custom Text"],
        help="Use preset samples or enter your own text"
    )

    if input_mode == "Sample Texts":
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was darkness, and then there was light.",
            "Machine learning models can discover complex patterns in data.",
            "The theory of relativity changed our understanding of space and time.",
        ]
        texts = sample_texts
        st.sidebar.text_area("Sample texts:", "\n".join(texts), height=150, disabled=True)
    else:
        custom_text = st.sidebar.text_area(
            "Enter your text(s):",
            "Enter one or more sentences here.\nEach line will be treated as a separate sample.",
            height=150
        )
        texts = [t.strip() for t in custom_text.strip().split('\n') if t.strip()]

    max_length = st.sidebar.slider(
        "Max Sequence Length",
        min_value=16,
        max_value=256,
        value=64,
        step=16,
        help="Maximum number of tokens per sample"
    )

    # Extract latents
    if st.sidebar.button("Extract Latent States", type="primary"):
        with st.spinner("Extracting latent states..."):
            latent_states, metas, token_ids, token_strings = extract_latents(
                extractor, tuple(texts), max_length
            )
            st.session_state.latent_states = latent_states
            st.session_state.metas = metas
            st.session_state.token_ids = token_ids
            st.session_state.token_strings = token_strings
            st.session_state.n_blocks = extractor.n_layers
            st.sidebar.success(f"Extracted: {latent_states.shape}")

    # Check if we have data
    if 'latent_states' not in st.session_state:
        st.info("👈 Configure the model and click 'Extract Latent States' to begin exploring!")

        # Show paper findings
        st.markdown("### Key Findings from the Paper")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="finding-box">
            <b>🔵 Attention vs 🔴 MLP Separation</b><br>
            The paper discovered that attention and MLP component outputs occupy
            <b>distinct geometric regions</b> in latent space. This pattern appears
            consistently across intermediate layers.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="finding-box">
            <b>📈 High Norm at Position 0</b><br>
            Latent states at the first sequence position have <b>significantly higher norms</b>
            than other positions. This acts as a learned bias term in the model.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="finding-box">
            <b>🌀 Helical Positional Embeddings</b><br>
            GPT-2's learned positional embeddings form a <b>helical structure</b>
            in high-dimensional space when visualized through PCA.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="finding-box">
            <b>📊 Layer-wise Evolution</b><br>
            Latent states evolve <b>gradually through layers</b>, with representations
            progressively changing as information flows through the transformer.
            </div>
            """, unsafe_allow_html=True)

        return

    # Create visualizer
    visualizer = LatentSpaceVisualizer(
        st.session_state.latent_states,
        st.session_state.metas,
        st.session_state.token_strings
    )

    # Visualization tabs
    tabs = st.tabs([
        "🔄 Layer Evolution",
        "⚡ Attention vs MLP",
        "📊 Position 0 Norm",
        "🌀 Positional Helix",
        "📉 Layer Norms",
        "🎬 Animation"
    ])

    # Tab 1: Layer Evolution
    with tabs[0]:
        st.subheader("Layer-wise Evolution of Latent States")
        st.markdown("""
        This 3D visualization shows how representations evolve through the transformer layers.
        - **Blue shades**: Attention-related layers (darker = earlier)
        - **Red shades**: MLP-related layers (darker = earlier)
        - **Dashed line**: Trajectory of layer centroids
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            mean_samples = st.checkbox("Mean over samples", value=True, key="le_samples")
        with col2:
            mean_seq = st.checkbox("Mean over sequence", value=True, key="le_seq")
        with col3:
            unit_vectors = st.checkbox("Unit vectors", value=True, key="le_unit")

        fig = visualizer.plot_layerwise_evolution_3d(
            mean_over_samples=mean_samples,
            mean_over_sequence=mean_seq,
            convert_to_unit_vectors=unit_vectors
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Attention vs MLP
    with tabs[1]:
        st.subheader("Attention vs MLP Geometric Separation")
        st.markdown("""
        **Key Finding**: Pre-add outputs from attention components (blue) occupy a
        *geometrically distinct region* from MLP outputs (red) in latent space.

        This separation is especially clear in intermediate layers.
        """)

        col1, col2 = st.columns(2)
        with col1:
            intermediate_only = st.checkbox("Intermediate blocks only", value=True, key="am_int")
        with col2:
            am_unit = st.checkbox("Unit vectors", value=True, key="am_unit")

        fig = visualizer.plot_attention_vs_mlp_2d(
            intermediate_blocks_only=intermediate_only,
            convert_to_unit_vectors=am_unit
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Position 0 Norm
    with tabs[2]:
        st.subheader("High Norm at Position 0 Phenomenon")
        st.markdown("""
        **Key Finding**: Latent states at sequence position 0 have significantly
        higher norms than other positions. This pattern emerges after the first
        few layers and persists through intermediate layers.

        The high-norm position 0 acts as a kind of **learned bias term** that
        affects the model's computations.
        """)

        fig = visualizer.plot_position_0_norm_analysis()
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        norms = compute_norms(st.session_state.latent_states)
        pos0_mean = norms['full'][:, :, 0].mean()
        other_mean = norms['full'][:, :, 1:].mean()
        ratio = pos0_mean / other_mean

        col1, col2, col3 = st.columns(3)
        col1.metric("Position 0 Mean Norm", f"{pos0_mean:.2f}")
        col2.metric("Other Positions Mean", f"{other_mean:.2f}")
        col3.metric("Ratio", f"{ratio:.2f}x")

    # Tab 4: Positional Helix
    with tabs[3]:
        st.subheader("Helical Structure of Positional Embeddings")
        st.markdown("""
        **Key Finding**: GPT-2's learned positional embeddings form a **helical pattern**
        when projected into 3D via PCA. This geometric structure encodes position
        information in a way that allows the model to understand sequence order.

        Colors indicate position: darker = earlier, lighter = later.
        """)

        pos_emb = get_positional_embeddings(extractor)
        fig = visualizer.plot_positional_embeddings_helix(pos_emb)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Showing {len(pos_emb)} positional embeddings ({extractor.hidden_size} dimensions each)")

    # Tab 5: Layer Norms
    with tabs[4]:
        st.subheader("Mean Norm by Layer")
        st.markdown("""
        This bar chart shows how the mean norm of latent states varies across layers.

        Notice:
        - Norms tend to **increase** through intermediate layers
        - There's often a **spike** in the final block
        - The final layer norm brings norms back down
        """)

        exclude_p0 = st.checkbox("Exclude position 0", value=True, key="ln_p0")

        fig = visualizer.plot_layer_norms_bar(exclude_position_0=exclude_p0)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 6: Animation
    with tabs[5]:
        st.subheader("Animated Layer Evolution")
        st.markdown("""
        Watch how the latent state distribution changes as we move through layers.

        - Each frame shows one layer's latent states projected to 2D
        - Use the slider or play button to step through layers
        """)

        fig = create_animated_layer_evolution(
            st.session_state.latent_states,
            st.session_state.metas,
            st.session_state.n_blocks
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    Based on the paper:

    **"Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"**

    by Ning, Rangaraju & Kuo (2025)

    [arXiv:2511.21594](https://arxiv.org/abs/2511.21594)
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Data Shape:**
    - Layers: {}
    - Samples: {}
    - Sequence: {}
    - Hidden: {}
    """.format(*st.session_state.latent_states.shape))


if __name__ == "__main__":
    main()
