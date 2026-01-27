#!/usr/bin/env python3
"""
Quick Visualization Script - Uses matplotlib (pre-installed)

Generates key visualizations from the paper without requiring plotly/streamlit.
Output: PNG images that demonstrate the paper's findings.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from latent_extractor import GPT2LatentExtractor, compute_norms, LayerMeta
from typing import List


def get_layer_colors(metas: List[LayerMeta], n_blocks: int):
    """Generate colors: blue for attention, red for MLP."""
    colors = []
    for meta in metas:
        if meta.layer_type == 'embed':
            colors.append((0.2, 0.8, 0.2))  # Green
        elif meta.layer_type == 'final_norm':
            colors.append((0.8, 0.8, 0.2))  # Yellow
        elif meta.is_attention:
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            colors.append((0.2 * intensity, 0.4 * intensity, 0.8 + 0.2 * (1 - intensity)))
        elif meta.is_mlp:
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            colors.append((0.8 + 0.2 * (1 - intensity), 0.2 * intensity, 0.2 * intensity))
        else:
            colors.append((0.5, 0.5, 0.5))
    return colors


def plot_layer_evolution_3d(latent_states, metas, n_blocks, output_path):
    """Create 3D scatter of layer evolution."""
    print("  Creating layer evolution 3D plot...")

    # Process data
    data = latent_states[:, :, 1:, :].copy()  # Exclude pos 0
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data = data / norms  # Unit vectors
    data = data.mean(axis=(1, 2))  # Mean over samples and sequence

    # PCA
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(data)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = get_layer_colors(metas, n_blocks)

    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
               c=colors, s=50, alpha=0.8)

    # Add trajectory line
    ax.plot(reduced[:, 0], reduced[:, 1], reduced[:, 2],
            'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('Layer-wise Evolution of Latent States (3D PCA)\nBlue=Attention, Red=MLP', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_attention_vs_mlp(latent_states, metas, n_blocks, output_path):
    """Visualize attention vs MLP separation."""
    print("  Creating attention vs MLP separation plot...")

    # Filter to pre-add layers in intermediate blocks
    attn_indices = [i for i, m in enumerate(metas)
                    if m.is_pre_add and m.is_attention and 2 <= m.block_num <= n_blocks - 3]
    mlp_indices = [i for i, m in enumerate(metas)
                   if m.is_pre_add and m.is_mlp and 2 <= m.block_num <= n_blocks - 3]

    if not attn_indices or not mlp_indices:
        # Fallback: use all pre-add layers
        attn_indices = [i for i, m in enumerate(metas) if m.is_pre_add and m.is_attention]
        mlp_indices = [i for i, m in enumerate(metas) if m.is_pre_add and m.is_mlp]

    # Get data
    attn_data = latent_states[attn_indices, :, 1:, :].copy()
    mlp_data = latent_states[mlp_indices, :, 1:, :].copy()

    # Unit vectors
    for d in [attn_data, mlp_data]:
        norms = np.linalg.norm(d, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        d /= norms

    # Mean over samples
    attn_mean = attn_data.mean(axis=1)  # (n_attn_layers, seq, hidden)
    mlp_mean = mlp_data.mean(axis=1)

    # Combine and PCA
    combined = np.vstack([
        attn_mean.reshape(-1, attn_mean.shape[-1]),
        mlp_mean.reshape(-1, mlp_mean.shape[-1])
    ])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    n_attn = attn_mean.reshape(-1, attn_mean.shape[-1]).shape[0]
    attn_reduced = reduced[:n_attn]
    mlp_reduced = reduced[n_attn:]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(attn_reduced[:, 0], attn_reduced[:, 1],
               c='steelblue', alpha=0.5, s=10, label='Attention outputs')
    ax.scatter(mlp_reduced[:, 0], mlp_reduced[:, 1],
               c='indianred', alpha=0.5, s=10, label='MLP outputs')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Attention vs MLP Geometric Separation (2D PCA)\nIntermediate Blocks, Pre-add Outputs', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_position_0_norm(latent_states, output_path):
    """Analyze high norm at position 0."""
    print("  Creating position 0 norm analysis...")

    norms = np.linalg.norm(latent_states, axis=-1)
    mean_by_pos = norms.mean(axis=(0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Bar chart of mean norm by position
    ax = axes[0, 0]
    colors = ['red' if i == 0 else 'steelblue' for i in range(len(mean_by_pos))]
    ax.bar(range(len(mean_by_pos)), mean_by_pos, color=colors)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Mean Norm')
    ax.set_title('Mean Norm by Sequence Position')

    # 2. Log scale
    ax = axes[0, 1]
    ax.semilogy(range(len(mean_by_pos)), mean_by_pos, 'o-', markersize=4)
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Mean Norm (log scale)')
    ax.set_title('Mean Norm by Position (Log Scale)')
    ax.grid(True, alpha=0.3)

    # 3. Histogram comparison
    ax = axes[1, 0]
    pos_0_norms = norms[:, :, 0].flatten()
    other_norms = norms[:, :, 1:].flatten()
    ax.hist(pos_0_norms, bins=50, alpha=0.7, label='Position 0', color='red')
    ax.hist(other_norms, bins=50, alpha=0.7, label='Other positions', color='steelblue')
    ax.set_xlabel('Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Norm Distribution: Position 0 vs Others')
    ax.legend()

    # 4. Heatmap
    ax = axes[1, 1]
    mean_by_layer_pos = norms.mean(axis=1)
    im = ax.imshow(mean_by_layer_pos, aspect='auto', cmap='viridis')
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Layer')
    ax.set_title('Norm by Layer and Position')
    plt.colorbar(im, ax=ax, label='Mean Norm')

    plt.suptitle('High Norm at Position 0 Phenomenon', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_positional_helix(pos_embeddings, output_path):
    """Visualize helical structure of positional embeddings."""
    print("  Creating positional embeddings helix plot...")

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(pos_embeddings)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    n_pos = len(pos_embeddings)
    colors = plt.cm.viridis(np.linspace(0, 1, n_pos))

    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
               c=colors, s=10, alpha=0.8)
    ax.plot(reduced[:, 0], reduced[:, 1], reduced[:, 2],
            'gray', alpha=0.3, linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('GPT-2 Positional Embeddings Helical Structure\nColor: Position (dark=early, light=late)', fontsize=12)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=n_pos))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Position')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_layer_norms(latent_states, metas, n_blocks, output_path):
    """Bar chart of mean norms by layer."""
    print("  Creating layer norms bar chart...")

    data = latent_states[:, :, 1:, :]  # Exclude pos 0
    norms = np.linalg.norm(data, axis=-1)
    mean_by_layer = norms.mean(axis=(1, 2))

    colors = get_layer_colors(metas, n_blocks)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(range(len(mean_by_layer)), mean_by_layer, color=colors)

    # Create simplified labels
    labels = []
    for i, meta in enumerate(metas):
        if meta.layer_type == 'embed':
            labels.append('E')
        elif meta.layer_type == 'final_norm':
            labels.append('F')
        elif i % 4 == 0:  # Show every 4th
            labels.append(f'{meta.block_num}')
        else:
            labels.append('')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_xlabel('Layer (E=Embed, numbers=block, F=Final)')
    ax.set_ylabel('Mean Norm')
    ax.set_title('Mean Latent State Norm by Layer (excluding position 0)\nBlue=Attention, Red=MLP', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_sequence_geometry(latent_states, metas, n_blocks, output_path):
    """Multi-panel visualization of sequence position geometry."""
    print("  Creating sequence position geometry plot...")

    # Use post-add intermediate layers
    layer_indices = [i for i, m in enumerate(metas)
                     if not m.is_pre_add and 2 <= m.block_num <= n_blocks - 3]

    if not layer_indices:
        layer_indices = [i for i, m in enumerate(metas) if not m.is_pre_add]

    data = latent_states[layer_indices].copy()

    # Unit vectors
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data = data / norms

    # Mean over samples and layers
    data = data.mean(axis=(0, 1))  # (seq_len, hidden_dim)

    # PCA to 6 dimensions
    pca = PCA(n_components=6)
    reduced = pca.fit_transform(data)

    # Create grid of 2D projections (15 pairs from 6 dims)
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()

    seq_len = reduced.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, seq_len))

    pair_idx = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if pair_idx >= 15:
                break
            ax = axes[pair_idx]
            ax.scatter(reduced[:, i], reduced[:, j], c=colors, s=15, alpha=0.8)
            ax.plot(reduced[:, i], reduced[:, j], 'gray', alpha=0.2, linewidth=0.5)
            ax.set_xlabel(f'PC{i}', fontsize=8)
            ax.set_ylabel(f'PC{j}', fontsize=8)
            ax.tick_params(labelsize=7)
            pair_idx += 1

    plt.suptitle('Sequence Position Geometry (Multi-dimensional PCA)\nColor: Position (dark=early, light=late)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM latent space visualizations")
    parser.add_argument("--model", type=str, default="gpt2", help="GPT-2 model variant")
    parser.add_argument("--output", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--max-length", type=int, default=64, help="Max sequence length")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLM Latent Space Geometry Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    extractor = GPT2LatentExtractor(args.model)
    n_blocks = extractor.n_layers

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was darkness, and then there was light.",
        "Machine learning models can discover complex patterns in data.",
        "The theory of relativity changed our understanding of space and time.",
        "Natural language processing enables computers to understand human language.",
        "Deep neural networks learn hierarchical representations of data.",
    ]

    # Extract latent states
    print(f"\nExtracting latent states from {len(texts)} texts...")
    latent_states, metas, token_ids, token_strings = extractor.extract(
        texts, max_length=args.max_length
    )
    print(f"Shape: {latent_states.shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_layer_evolution_3d(
        latent_states, metas, n_blocks,
        output_dir / "01_layer_evolution_3d.png"
    )

    plot_attention_vs_mlp(
        latent_states, metas, n_blocks,
        output_dir / "02_attention_vs_mlp.png"
    )

    plot_position_0_norm(
        latent_states,
        output_dir / "03_position_0_norm.png"
    )

    print("  Getting positional embeddings...")
    pos_emb = extractor.get_positional_embeddings()
    plot_positional_helix(
        pos_emb,
        output_dir / "04_positional_helix.png"
    )

    plot_layer_norms(
        latent_states, metas, n_blocks,
        output_dir / "05_layer_norms.png"
    )

    plot_sequence_geometry(
        latent_states, metas, n_blocks,
        output_dir / "06_sequence_geometry.png"
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    norms = compute_norms(latent_states)
    pos0_mean = norms['full'][:, :, 0].mean()
    other_mean = norms['full'][:, :, 1:].mean()
    print(f"Position 0 mean norm:    {pos0_mean:.2f}")
    print(f"Other positions mean:    {other_mean:.2f}")
    print(f"Ratio (pos0/others):     {pos0_mean/other_mean:.2f}x")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
