"""
Interactive LLM Latent Space Geometry Explorer

A visualization toolkit for exploring the geometric patterns in
Transformer-based language model latent spaces.

Based on the paper:
"Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"
"""

from .latent_extractor import GPT2LatentExtractor, LayerMeta, compute_norms
from .visualizer import LatentSpaceVisualizer, create_animated_layer_evolution

__all__ = [
    'GPT2LatentExtractor',
    'LayerMeta',
    'compute_norms',
    'LatentSpaceVisualizer',
    'create_animated_layer_evolution',
]
