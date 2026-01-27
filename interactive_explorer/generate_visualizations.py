#!/usr/bin/env python3
"""
Generate Static Visualizations for LLM Latent Space Geometry

This script creates all key visualizations from the paper and saves them as HTML files
that can be opened in any browser for interactive exploration.

Usage:
    python generate_visualizations.py --model gpt2 --output ./visualizations
"""

import argparse
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from latent_extractor import GPT2LatentExtractor, compute_norms
from visualizer import (
    LatentSpaceVisualizer,
    create_animated_layer_evolution
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM latent space visualizations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model variant to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualizations",
        help="Output directory for HTML files"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=None,
        help="Custom input texts (default: use sample texts)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print("LLM Latent Space Geometry Visualization Generator")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    extractor = GPT2LatentExtractor(args.model)

    # Sample texts
    if args.texts:
        texts = args.texts
    else:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was darkness, and then there was light.",
            "Machine learning models can discover complex patterns in data.",
            "The theory of relativity changed our understanding of space and time.",
            "Natural language processing enables computers to understand human language.",
            "Deep neural networks learn hierarchical representations of data.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Transformers have revolutionized the field of artificial intelligence.",
        ]

    print(f"Using {len(texts)} input texts")
    print(f"Max sequence length: {args.max_length}")
    print()

    # Extract latent states
    print("Extracting latent states...")
    latent_states, metas, token_ids, token_strings = extractor.extract(
        texts, max_length=args.max_length
    )
    print(f"Extracted shape: {latent_states.shape}")
    print()

    # Create visualizer
    visualizer = LatentSpaceVisualizer(latent_states, metas, token_strings)

    # Generate visualizations
    print("Generating visualizations...")
    print()

    # 1. Layer Evolution 3D
    print("  [1/6] Layer-wise Evolution (3D)...")
    fig = visualizer.plot_layerwise_evolution_3d(
        title=f"{args.model.upper()} Layer-wise Evolution of Latent States"
    )
    fig.write_html(output_dir / "01_layer_evolution_3d.html")

    # 2. Attention vs MLP
    print("  [2/6] Attention vs MLP Separation...")
    fig = visualizer.plot_attention_vs_mlp_2d(
        title=f"{args.model.upper()} Attention vs MLP Geometric Separation"
    )
    fig.write_html(output_dir / "02_attention_vs_mlp.html")

    # 3. Position 0 Norm
    print("  [3/6] Position 0 Norm Analysis...")
    fig = visualizer.plot_position_0_norm_analysis(
        title=f"{args.model.upper()} High Norm at Position 0"
    )
    fig.write_html(output_dir / "03_position_0_norm.html")

    # 4. Sequence Position Geometry
    print("  [4/6] Sequence Position Geometry...")
    fig = visualizer.plot_sequence_position_geometry(
        title=f"{args.model.upper()} Sequence Position Geometry"
    )
    fig.write_html(output_dir / "04_sequence_geometry.html")

    # 5. Positional Embeddings Helix
    print("  [5/6] Positional Embeddings Helix...")
    pos_emb = extractor.get_positional_embeddings()
    fig = visualizer.plot_positional_embeddings_helix(
        pos_emb,
        title=f"{args.model.upper()} Positional Embeddings Helical Structure"
    )
    fig.write_html(output_dir / "05_positional_helix.html")

    # 6. Layer Norms Bar Chart
    print("  [6/6] Layer Norms...")
    fig = visualizer.plot_layer_norms_bar(
        title=f"{args.model.upper()} Mean Latent State Norm by Layer"
    )
    fig.write_html(output_dir / "06_layer_norms.html")

    # 7. Animated Evolution
    print("  [Bonus] Animated Layer Evolution...")
    fig = create_animated_layer_evolution(
        latent_states, metas, extractor.n_layers
    )
    fig.write_html(output_dir / "07_animation.html")

    print()
    print(f"=" * 60)
    print("Visualizations generated successfully!")
    print(f"=" * 60)
    print()
    print("Open these HTML files in your browser:")
    for f in sorted(output_dir.glob("*.html")):
        print(f"  - {f}")
    print()

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 40)
    norms = compute_norms(latent_states)
    print(f"  Position 0 mean norm: {norms['full'][:, :, 0].mean():.2f}")
    print(f"  Other positions mean: {norms['full'][:, :, 1:].mean():.2f}")
    print(f"  Ratio: {norms['full'][:, :, 0].mean() / norms['full'][:, :, 1:].mean():.2f}x")
    print()


if __name__ == "__main__":
    main()
