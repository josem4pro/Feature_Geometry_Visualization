"""
Interactive Visualizations for LLM Latent Space Geometry

Creates Plotly-based interactive visualizations to explore:
1. Layer-wise evolution of latent states
2. Attention vs MLP geometric separation
3. Positional embedding structure (helical)
4. High norm at position 0 phenomenon
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple, Dict
import colorsys

from latent_extractor import LayerMeta


def get_layer_colors(metas: List[LayerMeta], n_blocks: int) -> List[str]:
    """
    Generate colors for layers: blue gradient for attention, red gradient for MLP.

    Args:
        metas: List of layer metadata
        n_blocks: Number of transformer blocks

    Returns:
        List of RGB color strings
    """
    colors = []

    for meta in metas:
        if meta.layer_type == 'embed':
            colors.append('rgb(50, 200, 50)')  # Green
        elif meta.layer_type == 'final_norm':
            colors.append('rgb(200, 200, 50)')  # Yellow
        elif meta.is_attention:
            # Blue gradient: darker early, lighter late
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            r = int(50 * intensity)
            g = int(100 * intensity)
            b = int(200 + 55 * (1 - intensity))
            colors.append(f'rgb({r}, {g}, {b})')
        elif meta.is_mlp:
            # Red gradient: darker early, lighter late
            intensity = 0.3 + 0.7 * (meta.block_num / max(n_blocks - 1, 1))
            r = int(200 + 55 * (1 - intensity))
            g = int(50 * intensity)
            b = int(50 * intensity)
            colors.append(f'rgb({r}, {g}, {b})')
        else:
            colors.append('rgb(128, 128, 128)')  # Gray

    return colors


def get_sequence_colors(seq_len: int, colorscale: str = 'Viridis') -> List[str]:
    """Generate gradient colors for sequence positions."""
    colors = px.colors.sample_colorscale(colorscale, seq_len)
    return colors


class LatentSpaceVisualizer:
    """Interactive visualizations for LLM latent space geometry."""

    def __init__(
        self,
        latent_states: np.ndarray,
        metas: List[LayerMeta],
        token_strings: Optional[List[List[str]]] = None
    ):
        """
        Initialize visualizer.

        Args:
            latent_states: Shape (n_layers, n_samples, seq_len, hidden_dim)
            metas: Layer metadata
            token_strings: Optional token strings for hover info
        """
        self.latent_states = latent_states
        self.metas = metas
        self.token_strings = token_strings

        self.n_layers, self.n_samples, self.seq_len, self.hidden_dim = latent_states.shape
        self.n_blocks = max(m.block_num for m in metas if m.block_num >= 0) + 1

        print(f"Visualizer initialized: {self.n_layers} layers, {self.n_samples} samples, {self.seq_len} seq_len")

    def _apply_pca(
        self,
        data: np.ndarray,
        n_components: int = 3,
        fit_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA dimensionality reduction.

        Args:
            data: Data to transform (will be flattened to 2D)
            n_components: Number of PCA components
            fit_data: Data to fit PCA on (uses data if None)

        Returns:
            Transformed data and fitted PCA object
        """
        original_shape = data.shape[:-1]
        flat_data = data.reshape(-1, data.shape[-1])

        pca = PCA(n_components=n_components)

        if fit_data is not None:
            fit_flat = fit_data.reshape(-1, fit_data.shape[-1])
            pca.fit(fit_flat)
        else:
            pca.fit(flat_data)

        transformed = pca.transform(flat_data)
        return transformed.reshape(*original_shape, n_components), pca

    def plot_layerwise_evolution_3d(
        self,
        mean_over_samples: bool = True,
        mean_over_sequence: bool = True,
        exclude_position_0: bool = True,
        convert_to_unit_vectors: bool = True,
        title: str = "Layer-wise Evolution of Latent States (3D PCA)"
    ) -> go.Figure:
        """
        Create 3D scatter plot showing how latent states evolve through layers.

        Args:
            mean_over_samples: Average across samples
            mean_over_sequence: Average across sequence positions
            exclude_position_0: Exclude position 0 (has high norm)
            convert_to_unit_vectors: Normalize to unit length
            title: Plot title

        Returns:
            Plotly Figure
        """
        data = self.latent_states.copy()

        # Exclude position 0 if requested
        if exclude_position_0 and data.shape[2] > 1:
            data = data[:, :, 1:, :]

        # Convert to unit vectors
        if convert_to_unit_vectors:
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            data = data / norms

        # Average over dimensions
        if mean_over_samples:
            data = data.mean(axis=1, keepdims=True)
        if mean_over_sequence:
            data = data.mean(axis=2, keepdims=True)

        # Apply PCA
        reduced, pca = self._apply_pca(data, n_components=3)

        # Flatten for plotting
        points = reduced.reshape(self.n_layers, -1, 3)
        n_points_per_layer = points.shape[1]

        # Create figure
        fig = go.Figure()

        colors = get_layer_colors(self.metas, self.n_blocks)

        # Add points for each layer
        for i, (meta, color) in enumerate(zip(self.metas, colors)):
            layer_points = points[i]

            if meta.layer_type == 'embed':
                name = "Embedding"
            elif meta.layer_type == 'final_norm':
                name = "Final Norm"
            elif meta.is_pre_add:
                name = f"Block {meta.block_num} {meta.layer_type.title()} (pre-add)"
            else:
                name = f"Block {meta.block_num} Post-{meta.layer_type.replace('post_', '').title()}"

            fig.add_trace(go.Scatter3d(
                x=layer_points[:, 0],
                y=layer_points[:, 1],
                z=layer_points[:, 2],
                mode='markers',
                marker=dict(size=6, color=color, opacity=0.8),
                name=name,
                hovertemplate=f"{name}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<br>PC3: %{{z:.3f}}<extra></extra>"
            ))

        # Add trajectory lines connecting layer means
        layer_means = points.mean(axis=1)
        fig.add_trace(go.Scatter3d(
            x=layer_means[:, 0],
            y=layer_means[:, 1],
            z=layer_means[:, 2],
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name='Layer progression',
            hoverinfo='skip'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
            ),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=1.02
            ),
            margin=dict(l=0, r=200, t=50, b=0)
        )

        return fig

    def plot_attention_vs_mlp_2d(
        self,
        intermediate_blocks_only: bool = True,
        mean_over_samples: bool = True,
        exclude_position_0: bool = True,
        convert_to_unit_vectors: bool = True,
        title: str = "Attention vs MLP Separation (2D PCA)"
    ) -> go.Figure:
        """
        Visualize the geometric separation between attention and MLP outputs.

        This is one of the key findings of the paper: attention and MLP outputs
        occupy distinct regions in latent space.

        Args:
            intermediate_blocks_only: Only show intermediate blocks
            mean_over_samples: Average across samples
            exclude_position_0: Exclude position 0
            convert_to_unit_vectors: Normalize to unit length
            title: Plot title

        Returns:
            Plotly Figure
        """
        # Filter to pre-add attention and MLP layers only
        layer_indices = []
        filtered_metas = []

        for i, meta in enumerate(self.metas):
            if not meta.is_pre_add:
                continue
            if intermediate_blocks_only:
                # Skip first and last few blocks
                if self.n_blocks <= 6:
                    include = True
                else:
                    include = 2 <= meta.block_num <= self.n_blocks - 3
                if not include:
                    continue

            layer_indices.append(i)
            filtered_metas.append(meta)

        if not layer_indices:
            # Fallback: use all pre-add layers
            for i, meta in enumerate(self.metas):
                if meta.is_pre_add:
                    layer_indices.append(i)
                    filtered_metas.append(meta)

        data = self.latent_states[layer_indices].copy()

        # Exclude position 0
        if exclude_position_0 and data.shape[2] > 1:
            data = data[:, :, 1:, :]

        # Convert to unit vectors
        if convert_to_unit_vectors:
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            data = data / norms

        # Average over samples
        if mean_over_samples:
            data = data.mean(axis=1, keepdims=True)

        # Apply PCA
        reduced, pca = self._apply_pca(data, n_components=2)
        points = reduced.reshape(len(layer_indices), -1, 2)

        # Create figure
        fig = go.Figure()

        # Separate attention and MLP
        attn_points = []
        attn_colors = []
        mlp_points = []
        mlp_colors = []

        colors = get_layer_colors(filtered_metas, self.n_blocks)

        for i, (meta, color) in enumerate(zip(filtered_metas, colors)):
            layer_pts = points[i]
            if meta.is_attention:
                attn_points.append(layer_pts)
                attn_colors.extend([color] * len(layer_pts))
            else:
                mlp_points.append(layer_pts)
                mlp_colors.extend([color] * len(layer_pts))

        # Plot attention points
        if attn_points:
            attn_all = np.vstack(attn_points)
            fig.add_trace(go.Scatter(
                x=attn_all[:, 0],
                y=attn_all[:, 1],
                mode='markers',
                marker=dict(size=5, color=attn_colors, opacity=0.7),
                name='Attention outputs',
                hovertemplate='Attention<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))

        # Plot MLP points
        if mlp_points:
            mlp_all = np.vstack(mlp_points)
            fig.add_trace(go.Scatter(
                x=mlp_all[:, 0],
                y=mlp_all[:, 1],
                mode='markers',
                marker=dict(size=5, color=mlp_colors, opacity=0.7),
                name='MLP outputs',
                hovertemplate='MLP<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            width=800,
            height=600
        )

        return fig

    def plot_position_0_norm_analysis(
        self,
        title: str = "High Norm at Position 0 Phenomenon"
    ) -> go.Figure:
        """
        Visualize the high norm phenomenon at sequence position 0.

        The paper found that latent states at position 0 have significantly
        higher norms than other positions.

        Returns:
            Plotly Figure with subplots
        """
        norms = np.linalg.norm(self.latent_states, axis=-1)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Norm by Sequence Position',
                'Norm by Position (Log Scale)',
                'Norm Distribution: Position 0 vs Others',
                'Layer-wise Norm by Position'
            ),
            specs=[[{}, {}], [{}, {}]]
        )

        # 1. Mean norm by position
        mean_by_pos = norms.mean(axis=(0, 1))
        fig.add_trace(
            go.Bar(
                x=list(range(len(mean_by_pos))),
                y=mean_by_pos,
                marker_color=['red' if i == 0 else 'steelblue' for i in range(len(mean_by_pos))],
                name='Mean Norm'
            ),
            row=1, col=1
        )

        # 2. Log scale version
        fig.add_trace(
            go.Scatter(
                x=list(range(len(mean_by_pos))),
                y=mean_by_pos,
                mode='lines+markers',
                marker=dict(color=['red' if i == 0 else 'steelblue' for i in range(len(mean_by_pos))]),
                name='Mean Norm (log)',
                showlegend=False
            ),
            row=1, col=2
        )
        fig.update_yaxes(type="log", row=1, col=2)

        # 3. Distribution comparison
        pos_0_norms = norms[:, :, 0].flatten()
        other_norms = norms[:, :, 1:].flatten()

        fig.add_trace(
            go.Histogram(
                x=pos_0_norms,
                name='Position 0',
                opacity=0.7,
                marker_color='red',
                nbinsx=50
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=other_norms,
                name='Other positions',
                opacity=0.7,
                marker_color='steelblue',
                nbinsx=50
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Norm", row=2, col=1)

        # 4. Heatmap of norms by layer and position
        mean_by_layer_pos = norms.mean(axis=1)  # (n_layers, seq_len)

        fig.add_trace(
            go.Heatmap(
                z=mean_by_layer_pos,
                x=list(range(self.seq_len)),
                y=[f"L{i}" for i in range(self.n_layers)],
                colorscale='Viridis',
                name='Norm heatmap'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=700,
            width=1000,
            showlegend=True
        )

        # Add axis labels
        fig.update_xaxes(title_text="Sequence Position", row=1, col=1)
        fig.update_yaxes(title_text="Mean Norm", row=1, col=1)
        fig.update_xaxes(title_text="Sequence Position", row=1, col=2)
        fig.update_xaxes(title_text="Sequence Position", row=2, col=2)
        fig.update_yaxes(title_text="Layer", row=2, col=2)

        return fig

    def plot_sequence_position_geometry(
        self,
        mean_over_samples: bool = True,
        mean_over_layers: bool = True,
        intermediate_layers_only: bool = True,
        convert_to_unit_vectors: bool = True,
        n_pca_dims: int = 6,
        title: str = "Sequence Position Geometry (Multi-dimensional PCA)"
    ) -> go.Figure:
        """
        Visualize the geometric pattern formed by sequence positions.

        For GPT-2, this reveals the helical structure of positional embeddings.

        Args:
            mean_over_samples: Average across samples
            mean_over_layers: Average across layers
            intermediate_layers_only: Use only intermediate blocks
            convert_to_unit_vectors: Normalize to unit length
            n_pca_dims: Number of PCA dimensions to visualize
            title: Plot title

        Returns:
            Plotly Figure with grid of 2D projections
        """
        # Filter layers if requested
        if intermediate_layers_only and self.n_blocks > 6:
            layer_indices = [
                i for i, m in enumerate(self.metas)
                if not m.is_pre_add and 2 <= m.block_num <= self.n_blocks - 3
            ]
        else:
            layer_indices = [
                i for i, m in enumerate(self.metas)
                if not m.is_pre_add  # Post-add only for cleaner visualization
            ]

        data = self.latent_states[layer_indices].copy()

        # Convert to unit vectors
        if convert_to_unit_vectors:
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            data = data / norms

        # Average
        if mean_over_samples:
            data = data.mean(axis=1, keepdims=True)
        if mean_over_layers:
            data = data.mean(axis=0, keepdims=True)

        # Apply PCA
        reduced, pca = self._apply_pca(data, n_components=n_pca_dims)
        points = reduced.reshape(-1, n_pca_dims)  # (seq_len, n_dims)

        # Create grid of 2D plots
        n_pairs = n_pca_dims * (n_pca_dims - 1) // 2
        n_cols = min(5, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        subplot_titles = []
        for i in range(n_pca_dims):
            for j in range(i + 1, n_pca_dims):
                subplot_titles.append(f"PC{i} vs PC{j}")

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles[:n_rows * n_cols]
        )

        colors = get_sequence_colors(self.seq_len)

        pair_idx = 0
        for i in range(n_pca_dims):
            for j in range(i + 1, n_pca_dims):
                row = pair_idx // n_cols + 1
                col = pair_idx % n_cols + 1

                fig.add_trace(
                    go.Scatter(
                        x=points[:, i],
                        y=points[:, j],
                        mode='markers+lines',
                        marker=dict(
                            size=4,
                            color=list(range(self.seq_len)),
                            colorscale='Viridis',
                            showscale=(pair_idx == 0)
                        ),
                        line=dict(color='lightgray', width=0.5),
                        hovertemplate=f"Pos %{{customdata}}<br>PC{i}: %{{x:.3f}}<br>PC{j}: %{{y:.3f}}<extra></extra>",
                        customdata=list(range(self.seq_len)),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                pair_idx += 1
                if pair_idx >= n_rows * n_cols:
                    break
            if pair_idx >= n_rows * n_cols:
                break

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=200 * n_rows + 100,
            width=200 * n_cols + 100,
        )

        return fig

    def plot_positional_embeddings_helix(
        self,
        positional_embeddings: np.ndarray,
        title: str = "GPT-2 Positional Embeddings Helical Structure"
    ) -> go.Figure:
        """
        Visualize the helical structure of GPT-2's learned positional embeddings.

        Args:
            positional_embeddings: Shape (max_seq_len, hidden_dim)
            title: Plot title

        Returns:
            Plotly 3D Figure
        """
        # Apply PCA
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(positional_embeddings)

        # Create 3D scatter with trajectory
        fig = go.Figure()

        # Color by position
        n_pos = len(positional_embeddings)
        colors = list(range(n_pos))

        # Add points
        fig.add_trace(go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title='Position'),
                opacity=0.8
            ),
            hovertemplate='Position: %{customdata}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
            customdata=list(range(n_pos)),
            name='Positional embeddings'
        ))

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='skip',
            name='Trajectory'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
            ),
            width=800,
            height=700
        )

        return fig

    def plot_layer_norms_bar(
        self,
        exclude_position_0: bool = False,
        title: str = "Mean Latent State Norm by Layer"
    ) -> go.Figure:
        """
        Create bar chart of mean norms per layer.

        Args:
            exclude_position_0: Exclude position 0 from calculation
            title: Plot title

        Returns:
            Plotly Figure
        """
        data = self.latent_states
        if exclude_position_0 and data.shape[2] > 1:
            data = data[:, :, 1:, :]

        norms = np.linalg.norm(data, axis=-1)
        mean_by_layer = norms.mean(axis=(1, 2))

        colors = get_layer_colors(self.metas, self.n_blocks)

        # Create labels
        labels = []
        for meta in self.metas:
            if meta.layer_type == 'embed':
                labels.append('Embed')
            elif meta.layer_type == 'final_norm':
                labels.append('Final')
            elif meta.is_pre_add:
                short_type = 'A' if meta.is_attention else 'M'
                labels.append(f'{meta.block_num}{short_type}')
            else:
                short_type = 'a' if meta.is_attention else 'm'
                labels.append(f'{meta.block_num}{short_type}')

        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(mean_by_layer))),
                y=mean_by_layer,
                marker_color=colors,
                hovertemplate='%{customdata}<br>Norm: %{y:.2f}<extra></extra>',
                customdata=labels
            )
        ])

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title='Layer',
            yaxis_title='Mean Norm',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(labels))),
                ticktext=labels,
                tickangle=45
            ),
            width=1000,
            height=500
        )

        return fig


def create_animated_layer_evolution(
    latent_states: np.ndarray,
    metas: List[LayerMeta],
    n_blocks: int,
    fps: int = 2
) -> go.Figure:
    """
    Create animated visualization of layer evolution.

    Args:
        latent_states: Shape (n_layers, n_samples, seq_len, hidden_dim)
        metas: Layer metadata
        n_blocks: Number of transformer blocks
        fps: Frames per second

    Returns:
        Animated Plotly Figure
    """
    # Prepare data
    data = latent_states[:, :, 1:, :].copy()  # Exclude pos 0

    # Unit vectors
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data = data / norms

    # Mean over samples
    data = data.mean(axis=1)  # (n_layers, seq_len, hidden_dim)

    # PCA on all data
    flat = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flat)
    reduced = reduced.reshape(data.shape[0], data.shape[1], 2)

    # Create frames
    frames = []
    colors = get_layer_colors(metas, n_blocks)

    for i, (layer_data, meta, color) in enumerate(zip(reduced, metas, colors)):
        if meta.layer_type == 'embed':
            name = "Initial Embedding"
        elif meta.layer_type == 'final_norm':
            name = "Final Layer Norm"
        elif meta.is_pre_add:
            name = f"Block {meta.block_num} {meta.layer_type.title()} (pre-add)"
        else:
            name = f"Block {meta.block_num} Post-{meta.layer_type.replace('post_', '').title()}"

        frames.append(go.Frame(
            data=[go.Scatter(
                x=layer_data[:, 0],
                y=layer_data[:, 1],
                mode='markers',
                marker=dict(size=5, color=color),
            )],
            name=str(i),
            layout=go.Layout(title_text=name)
        ))

    # Initial frame
    fig = go.Figure(
        data=[go.Scatter(
            x=reduced[0, :, 0],
            y=reduced[0, :, 1],
            mode='markers',
            marker=dict(size=5, color=colors[0]),
        )],
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        title="Layer Evolution Animation",
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.5,
            xanchor='center',
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(
                        frame=dict(duration=1000//fps, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                ),
                dict(
                    label='Pause',
                    method='animate',
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode='immediate'
                    )]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                method='animate',
                args=[[str(i)], dict(
                    frame=dict(duration=0, redraw=True),
                    mode='immediate'
                )],
                label=str(i)
            ) for i in range(len(frames))],
            x=0.1,
            y=0,
            len=0.8
        )],
        width=800,
        height=600
    )

    return fig
