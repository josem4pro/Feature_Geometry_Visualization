"""
Latent State Extractor for GPT-2
Extracts hidden states from transformer blocks for visualization.

Based on the paper: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings


@dataclass
class LayerMeta:
    """Metadata for a captured latent state layer."""
    block_num: int
    layer_type: str  # 'attention', 'mlp', 'post_attn', 'post_mlp', 'embed', 'final_norm'
    is_pre_add: bool  # True = raw component output, False = after residual addition

    @property
    def is_attention(self) -> bool:
        return self.layer_type in ('attention', 'post_attn')

    @property
    def is_mlp(self) -> bool:
        return self.layer_type in ('mlp', 'post_mlp')

    @property
    def color(self) -> str:
        """Return color for visualization: blue for attention, red for MLP."""
        if self.layer_type in ('attention', 'post_attn'):
            return 'blue'
        elif self.layer_type in ('mlp', 'post_mlp'):
            return 'red'
        elif self.layer_type == 'embed':
            return 'green'
        else:
            return 'gray'


class GPT2LatentExtractor:
    """
    Extract latent states from GPT-2 at multiple points within each transformer block.

    Capture points per block:
    1. After attention (pre-add to residual)
    2. After attention + residual addition (post-add)
    3. After MLP (pre-add to residual)
    4. After MLP + residual addition (post-add)
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            device: Device to use (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

        self.n_layers = self.model.config.n_layer
        self.hidden_size = self.model.config.n_embd
        self.max_seq_length = self.model.config.n_positions

        print(f"Model loaded: {self.n_layers} layers, {self.hidden_size} hidden dim, max {self.max_seq_length} tokens")

        # Storage for captured activations
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _register_hooks(self, capture_norms: bool = False):
        """Register forward hooks to capture activations."""
        self._activations.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Capture initial embeddings (after wte + wpe)
        def embed_hook(module, input, output):
            self._activations['embed'] = output.detach().cpu()

        self._hooks.append(
            self.model.transformer.drop.register_forward_hook(embed_hook)
        )

        # Capture at each transformer block
        for block_idx, block in enumerate(self.model.transformer.h):
            # Capture attention output (pre-add)
            def make_attn_hook(idx):
                def hook(module, input, output):
                    # output[0] is the attention output
                    self._activations[f'block_{idx}_attn'] = output[0].detach().cpu()
                return hook

            self._hooks.append(
                block.attn.register_forward_hook(make_attn_hook(block_idx))
            )

            # Capture MLP output (pre-add)
            def make_mlp_hook(idx):
                def hook(module, input, output):
                    self._activations[f'block_{idx}_mlp'] = output.detach().cpu()
                return hook

            self._hooks.append(
                block.mlp.register_forward_hook(make_mlp_hook(block_idx))
            )

        # Capture final layer norm output
        def final_norm_hook(module, input, output):
            self._activations['final_norm'] = output.detach().cpu()

        self._hooks.append(
            self.model.transformer.ln_f.register_forward_hook(final_norm_hook)
        )

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        max_length: int = 128,
        include_embeddings: bool = True,
    ) -> Tuple[np.ndarray, List[LayerMeta], np.ndarray, List[List[str]]]:
        """
        Extract latent states from text inputs.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            include_embeddings: Whether to include initial embeddings

        Returns:
            Tuple of:
            - latent_states: Array of shape (n_layers, n_samples, seq_len, hidden_dim)
            - metas: List of LayerMeta for each layer
            - token_ids: Array of token IDs (n_samples, seq_len)
            - token_strings: List of token strings per sample
        """
        self._register_hooks()

        try:
            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Forward pass
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Collect latent states
            latent_states = []
            metas = []

            if include_embeddings:
                latent_states.append(self._activations['embed'].numpy())
                metas.append(LayerMeta(block_num=-1, layer_type='embed', is_pre_add=False))

            # For each block, we capture attention and MLP outputs
            # We compute post-add states from the residual stream
            residual = self._activations['embed']

            for block_idx in range(self.n_layers):
                attn_out = self._activations[f'block_{block_idx}_attn']
                mlp_out = self._activations[f'block_{block_idx}_mlp']

                # Attention output (pre-add)
                latent_states.append(attn_out.numpy())
                metas.append(LayerMeta(block_num=block_idx, layer_type='attention', is_pre_add=True))

                # Post-attention residual
                post_attn = residual + attn_out
                latent_states.append(post_attn.numpy())
                metas.append(LayerMeta(block_num=block_idx, layer_type='post_attn', is_pre_add=False))

                # MLP output (pre-add)
                latent_states.append(mlp_out.numpy())
                metas.append(LayerMeta(block_num=block_idx, layer_type='mlp', is_pre_add=True))

                # Post-MLP residual
                post_mlp = post_attn + mlp_out
                latent_states.append(post_mlp.numpy())
                metas.append(LayerMeta(block_num=block_idx, layer_type='post_mlp', is_pre_add=False))

                # Update residual for next block
                residual = post_mlp

            # Final layer norm
            latent_states.append(self._activations['final_norm'].numpy())
            metas.append(LayerMeta(block_num=self.n_layers, layer_type='final_norm', is_pre_add=False))

            # Stack into (n_layers, n_samples, seq_len, hidden_dim)
            latent_array = np.stack(latent_states, axis=0)

            # Get token info
            token_ids = input_ids.cpu().numpy()
            token_strings = [
                self.tokenizer.convert_ids_to_tokens(ids.tolist())
                for ids in input_ids.cpu()
            ]

            return latent_array, metas, token_ids, token_strings

        finally:
            self._remove_hooks()

    def get_positional_embeddings(self) -> np.ndarray:
        """
        Extract the learned positional embeddings matrix.

        Returns:
            Array of shape (max_seq_len, hidden_dim)
        """
        return self.model.transformer.wpe.weight.detach().cpu().numpy()

    def get_token_embeddings(self) -> np.ndarray:
        """
        Extract the token embedding matrix.

        Returns:
            Array of shape (vocab_size, hidden_dim)
        """
        return self.model.transformer.wte.weight.detach().cpu().numpy()


def compute_norms(latent_states: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute various norm statistics from latent states.

    Args:
        latent_states: Array of shape (n_layers, n_samples, seq_len, hidden_dim)

    Returns:
        Dictionary with:
        - 'by_layer': Mean norm per layer (n_layers,)
        - 'by_position': Mean norm per sequence position (seq_len,)
        - 'by_sample': Mean norm per sample (n_samples,)
        - 'full': All norms (n_layers, n_samples, seq_len)
    """
    norms = np.linalg.norm(latent_states, axis=-1)

    return {
        'by_layer': norms.mean(axis=(1, 2)),
        'by_position': norms.mean(axis=(0, 1)),
        'by_sample': norms.mean(axis=(0, 2)),
        'full': norms
    }


if __name__ == "__main__":
    # Quick test
    extractor = GPT2LatentExtractor("gpt2")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn complex patterns.",
    ]

    latent_states, metas, token_ids, token_strings = extractor.extract(texts, max_length=32)

    print(f"\nExtracted shape: {latent_states.shape}")
    print(f"Number of layers captured: {len(metas)}")
    print(f"\nLayer types:")
    for i, meta in enumerate(metas[:10]):
        print(f"  {i}: block {meta.block_num}, {meta.layer_type}, pre_add={meta.is_pre_add}")

    # Test norms
    norms = compute_norms(latent_states)
    print(f"\nNorm by layer shape: {norms['by_layer'].shape}")
    print(f"Norm by position shape: {norms['by_position'].shape}")
    print(f"Position 0 mean norm: {norms['full'][:, :, 0].mean():.2f}")
    print(f"Position 1+ mean norm: {norms['full'][:, :, 1:].mean():.2f}")
