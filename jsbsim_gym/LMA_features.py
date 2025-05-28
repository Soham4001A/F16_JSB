# stashed locally

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from typing import Type # For type hinting nn.Module subclasses
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces as gymnasium_spaces

# Attempt to import JSBSimFeatureExtractor, handling potential relative import issues.
try:
    from features import JSBSimFeatureExtractor
except ImportError:
    from .features import JSBSimFeatureExtractor


# ===============================================
# Helper Function for LMA Configuration
# ===============================================
def find_closest_divisor(total_value: int, target_divisor: int, max_delta: int = 100) -> int:
    """
    Finds a divisor of `total_value` that is closest to `target_divisor`.

    Searches within `target_divisor +/- max_delta`. If no divisor is found
    in this range, it performs an exhaustive search for the closest divisor.

    Args:
        total_value: The positive integer whose divisor is sought.
        target_divisor: The desired divisor.
        max_delta: The maximum deviation from `target_divisor` to search initially.

    Returns:
        The divisor of `total_value` closest to `target_divisor`.

    Raises:
        ValueError: If inputs are not valid positive integers or max_delta is negative.
    """
    if not isinstance(total_value, int) or total_value <= 0:
        raise ValueError(f"total_value ({total_value}) must be a positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        # Ensure target_divisor is positive for modulo operations and sensible search.
        target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0:
        raise ValueError(f"max_delta ({max_delta}) must be a non-negative integer.")

    if total_value == 0: return 1 # Technically not a divisor, but avoids division by zero issues.

    # Check target_divisor first
    if total_value % target_divisor == 0:
        return target_divisor

    # Search in the vicinity of target_divisor
    search_start = target_divisor
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0:
            return candidate_minus
        candidate_plus = search_start + delta
        # No need to check candidate_plus > 0 as search_start >=1 and delta >=1
        if total_value % candidate_plus == 0:
            return candidate_plus

    # If no divisor found in the delta range, perform a broader search
    print(f"Warning: No divisor found near {target_divisor} for {total_value} within delta={max_delta}. Searching all divisors.")
    best_divisor = 1
    min_diff = abs(target_divisor - 1)

    # Check divisors up to sqrt(total_value)
    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1 = i
            div2 = total_value // i
            diff1 = abs(target_divisor - div1)
            diff2 = abs(target_divisor - div2)
            if diff1 < min_diff:
                min_diff = diff1
                best_divisor = div1
            if diff2 < min_diff:
                min_diff = diff2
                best_divisor = div2

    # Check total_value itself as a divisor
    diff_total = abs(target_divisor - total_value)
    if diff_total < min_diff:
        best_divisor = total_value

    print(f"Using {best_divisor} as fallback divisor for {total_value} (target: {target_divisor}).")
    return best_divisor

# ===============================================
# LMA (Latent Modulated Attention) Core Components
# Adapted for Reinforcement Learning (RL)
# ===============================================

@dataclass
class LMAConfigRL:
    """
    Configuration for the Latent Modulated Attention (LMA) architecture,
    tailored for reinforcement learning feature extraction.

    This class defines the dimensionalities and head counts for the two main
    stages of LMA: the initial stacking/reshaping stage and the subsequent
    latent attention stage. It automatically calculates the reshaped sequence
    length (`L_new`) and feature dimensionality (`C_new`) based on the input
    parameters and `find_closest_divisor` to ensure valid reshaping.

    Attributes:
        seq_len: Input sequence length (e.g., number of stacked frames from env).
        embed_dim: Embedding dimension for each item in the input sequence after initial projection.
        num_heads_stacking: Number of heads for the multi-head stacking operation in the initial transform.
                           `embed_dim` must be divisible by this.
        target_l_new: Target sequence length after the initial stacking and reshaping.
                      The actual `L_new` will be a divisor of `seq_len * embed_dim` close to this target.
        d_new: Dimensionality of features after the second embedding layer (input to latent attention).
        num_heads_latent: Number of attention heads in the latent self-attention blocks.
                          `d_new` must be divisible by this.
        L_new: Calculated sequence length after initial transform (reshaping). Initialized by `__post_init__`.
        C_new: Calculated feature dimensionality after initial transform (reshaping). Initialized by `__post_init__`.
    """
    seq_len: int
    embed_dim: int
    num_heads_stacking: int
    target_l_new: int
    d_new: int
    num_heads_latent: int
    L_new: int = field(init=False)
    C_new: int = field(init=False)

    def __post_init__(self):
        """Validates configuration and computes L_new, C_new."""
        if not all(x > 0 for x in [self.seq_len, self.embed_dim, self.num_heads_stacking,
                                   self.target_l_new, self.d_new, self.num_heads_latent]):
            raise ValueError("All LMAConfigRL numeric inputs must be positive.")

        if self.embed_dim % self.num_heads_stacking != 0:
            raise ValueError(
                f"LMA embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads_stacking ({self.num_heads_stacking})."
            )
        if self.d_new % self.num_heads_latent != 0:
            raise ValueError(
                f"LMA d_new ({self.d_new}) must be divisible by "
                f"num_heads_latent ({self.num_heads_latent})."
            )

        total_features = self.seq_len * self.embed_dim
        if total_features == 0:
            raise ValueError("LMA total features (seq_len * embed_dim) cannot be zero.")

        try:
            self.L_new = find_closest_divisor(total_features, self.target_l_new)
            if self.L_new != self.target_l_new:
                print(f"LMAConfigRL ADJUSTMENT: L_new changed from target {self.target_l_new} to {self.L_new} "
                      f"to be a divisor of total_features ({total_features}).")
            if self.L_new <= 0: # Should be caught by find_closest_divisor for total_value > 0
                raise ValueError("Calculated L_new must be positive.")
            if total_features % self.L_new != 0: # Should not happen if find_closest_divisor is correct
                raise RuntimeError(
                    f"Internal Error: total_features ({total_features}) "
                    f"not divisible by final L_new ({self.L_new})."
                )
            self.C_new = total_features // self.L_new
            if self.C_new <= 0:
                raise ValueError("Calculated C_new must be positive.")
        except ValueError as e:
            raise ValueError(f"Error in LMAConfigRL calculating L_new/C_new: {e}") from e


class LayerNorm(nn.Module):
    """A standard Layer Normalization module.

    Args:
        ndim (int): The number of features to normalize over (typically the last dimension).
        bias (bool): If True, add a learnable bias to the C_out. Default: True.
    """
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class LMA_InitialTransform_RL(nn.Module):
    """
    Initial transformation stage of the LMA model for RL.
    It performs embedding, positional encoding, multi-head stacking,
    and reshaping of the input sequence.

    Args:
        features_per_step (int): Dimensionality of features for each item in the input sequence.
        lma_config (LMAConfigRL): Configuration object for LMA.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
        super().__init__()
        self.lma_config = lma_config

        self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=bias)
        self.input_embedding_act = nn.ReLU()
        self.embedding_dropout = nn.Dropout(p=dropout)

        # Second embedding layer after reshaping
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=bias)
        self.embed_layer_2_act = nn.ReLU()

    def _positional_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        """Generates sinusoidal positional encodings."""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SeqLen, FeaturesPerStep).

        Returns:
            torch.Tensor: Transformed tensor of shape (Batch, L_new, d_new).
        """
        B, L, _ = x.shape # L is the original sequence length
        if L != self.lma_config.seq_len:
            raise ValueError(
                f"Input sequence length ({L}) does not match "
                f"LMAConfigRL.seq_len ({self.lma_config.seq_len})."
            )

        # Initial embedding and positional encoding
        y = self.input_embedding_act(self.input_embedding(x))
        y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device)
        y = self.embedding_dropout(y)

        # Multi-head stacking: Reshape (B, L, D0) -> (B, L*num_heads_stacking, D0/num_heads_stacking)
        d0 = self.lma_config.embed_dim
        nh_stack = self.lma_config.num_heads_stacking
        dk_stack = d0 // nh_stack # Dimension per stacking head
        # Split into heads views: list of (B, L, dk_stack) tensors
        head_views = torch.split(y, dk_stack, dim=2)
        # Concatenate along sequence dimension: (B, L * nh_stack, dk_stack)
        x_stacked_seq = torch.cat(head_views, dim=1) # This was the original LMA logic for sequence stacking

        # The LMA paper also mentions an alternative channel stacking:
        # Reshape y to (B, L, nh_stack, dk_stack)
        # y_reshaped_for_channel_stack = y.view(B, L, nh_stack, dk_stack)
        # Transpose to (B, dk_stack, nh_stack, L) -> (B, dk_stack, nh_stack*L) not quite
        # Transpose to (B, L, dk_stack, nh_stack) and view as (B, L*dk_stack, nh_stack) ? No.
        # The intended "stacking" is usually about creating more "tokens" or altering feature dimension.
        # Original LMA seems to split D0 into heads, and then concatenate those head views along the L dimension,
        # effectively making the sequence L*num_heads long, with each token being dk_stack dimensional.
        # x_stacked = x_stacked_seq # Using the sequence stacking from original LMA

        # The current RL implementation seems to do:
        # Split y into chunks along embed_dim (dim=2)
        # head_views = torch.split(y, dk_stack, dim=2) # List of B, L, dk_stack
        # x_stacked = torch.cat(head_views, dim=1) # Concatenates along L -> B, L*num_heads, dk_stack
        # This interpretation seems consistent with your existing code.
        x_stacked = x_stacked_seq


        # Reshape to (B, L_new, C_new)
        L_new = self.lma_config.L_new
        C_new = self.lma_config.C_new
        # Flatten the (B, L*nh_stack, dk_stack) tensor
        x_flat = x_stacked.reshape(B, -1) # Total features: L*nh_stack*dk_stack = L*d0
        # Rechunk into (B, L_new, C_new)
        # Total features must be L_new * C_new
        x_rechunked = x_flat.view(B, L_new, C_new)

        # Second embedding layer
        z = self.embed_layer_2_act(self.embed_layer_2(x_rechunked))
        return z


class LatentAttention_RL(nn.Module):
    """
    Self-attention mechanism operating on the latent (reshaped) sequence.
    Uses scaled dot-product attention, with optional Flash Attention if available.

    Args:
        d_new (int): Dimensionality of input features (and output features).
        num_heads_latent (int): Number of attention heads. `d_new` must be divisible by this.
        dropout (float): Dropout probability for attention weights and output projection.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
        super().__init__()
        assert d_new % num_heads_latent == 0, "d_new must be divisible by num_heads_latent."
        self.d_new = d_new
        self.num_heads = num_heads_latent
        self.head_dim = d_new // num_heads_latent

        self.c_attn = nn.Linear(d_new, 3 * d_new, bias=bias) # Query, Key, Value projection
        self.c_proj = nn.Linear(d_new, d_new, bias=bias)    # Output projection

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout # For Flash Attention if used

        # Check for PyTorch's Flash Attention availability
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash:
            print("LatentAttention_RL: Using F.scaled_dot_product_attention (Flash Attention).")
        else:
            print("LatentAttention_RL: F.scaled_dot_product_attention not found, using manual attention.")


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): Input tensor of shape (Batch, L_new, d_new).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, L_new, d_new).
        """
        B, L_new, _ = z.size() # C is d_new

        # Project to Q, K, V
        q, k, v = self.c_attn(z).split(self.d_new, dim=2)

        # Reshape Q, K, V for multi-head attention
        # (B, L_new, num_heads, head_dim) -> (B, num_heads, L_new, head_dim)
        q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        if self.flash:
            # PyTorch's scaled_dot_product_attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, # No mask for non-causal self-attention
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual implementation
            att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att_probs = F.softmax(att_scores, dim=-1)
            att_probs = self.attn_dropout(att_probs)
            y = att_probs @ v # (B, num_heads, L_new, head_dim)

        # Concatenate heads and project
        # (B, num_heads, L_new, head_dim) -> (B, L_new, num_heads, head_dim) -> (B, L_new, d_new)
        y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new)
        y = self.resid_dropout(self.c_proj(y))
        return y


class LatentMLP_RL(nn.Module):
    """
    Feed-forward network (MLP) applied independently to each token in the latent sequence.
    Typically part of a Transformer block.

    Args:
        d_new (int): Input and output dimensionality.
        ff_latent_hidden (int): Dimensionality of the hidden layer in the MLP.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(d_new, ff_latent_hidden, bias=bias)
        self.gelu = nn.GELU() # GELU activation
        self.c_proj = nn.Linear(ff_latent_hidden, d_new, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMABlock_RL(nn.Module):
    """
    A single block of the Latent Modulated Attention (LMA) architecture.
    Consists of a latent self-attention layer followed by an MLP,
    with LayerNorm and residual connections.

    Args:
        lma_config (LMAConfigRL): Configuration object for LMA.
                                  Uses `d_new` and `num_heads_latent`.
        ff_latent_hidden (int): Dimensionality of the hidden layer in the MLP.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers and LayerNorm.
    """
    def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__()
        self.ln_1 = LayerNorm(lma_config.d_new, bias=bias)
        self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias)
        self.ln_2 = LayerNorm(lma_config.d_new, bias=bias)
        self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ Residual connections: z = z + Attn(LN(z)); z = z + MLP(LN(z)) """
        z = z + self.attn(self.ln_1(z))
        z = z + self.mlp(self.ln_2(z))
        return z


class LMAFeaturesExtractor(BaseFeaturesExtractor):
    """
    A features extractor based on the Latent Modulated Attention (LMA) architecture.
    Designed for processing sequences of features (e.g., flattened image patches,
    or sequences of structured data). This version is adapted for RL, taking a
    flat observation vector which is then reshaped into a sequence.

    The input `observation_space` is expected to be a 1D Box space, where the
    total dimension is `seq_len * features_per_step`.

    Args:
        observation_space (gymnasium_spaces.Box): The observation space from the environment.
            Expected to be 1D, which will be reshaped.
        embed_dim (int): Embedding dimension for each item in the input sequence (D0).
        num_heads_stacking (int): Number of heads for multi-head stacking in initial transform.
        target_l_new (int): Target sequence length after initial reshaping (L').
        d_new (int): Feature dimensionality after the second embedding (D').
        num_heads_latent (int): Number of attention heads in latent self-attention blocks.
        ff_latent_hidden (int): Hidden dimension for the MLP in LMA blocks.
        num_lma_layers (int): Number of LMA blocks.
        seq_len (int): The sequence length to reshape the input observation into.
                       `observation_space.shape[0]` must be divisible by `seq_len`.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(self,
                 observation_space: gymnasium_spaces.Box,
                 embed_dim: int = 64,
                 num_heads_stacking: int = 4,
                 target_l_new: int = 3, # Example: if seq_len=6, embed_dim=64, total=384. 384/3=128. L_new=3, C_new=128.
                 d_new: int = 32,
                 num_heads_latent: int = 4,
                 ff_latent_hidden: int = 64,
                 num_lma_layers: int = 2,
                 seq_len: int = 6, # This is L, the number of "patches" or "time steps"
                 dropout: float = 0.1,
                 bias: bool = True):

        if not isinstance(observation_space, gymnasium_spaces.Box):
            # This warning is more for informational purposes if a user passes an unexpected type.
            print(f"Warning: LMAFeaturesExtractor received observation_space of type "
                  f"{type(observation_space)}, expected gymnasium.spaces.Box.")
        if len(observation_space.shape) != 1:
            raise ValueError(
                f"LMAFeaturesExtractor expects a 1D observation space (flattened sequence), "
                f"but got shape {observation_space.shape}."
            )

        self.input_dim_total = observation_space.shape[0]
        self.seq_len = seq_len # This is L (original sequence length after reshape)
        if self.input_dim_total % seq_len != 0:
            raise ValueError(
                f"Total input dimension ({self.input_dim_total}) must be divisible by "
                f"seq_len ({seq_len}) for reshaping."
            )
        self.features_per_step = self.input_dim_total // seq_len # This is F (features per token/patch)

        # Create LMAConfigRL to calculate L_new, C_new, and ultimately features_dim
        # Note: lma_config.seq_len corresponds to self.seq_len (L)
        #       lma_config.embed_dim corresponds to the argument embed_dim (D0)
        try:
            _lma_config_temp = LMAConfigRL(
                seq_len=self.seq_len,            # L (number of items in sequence)
                embed_dim=embed_dim,             # D0 (embedding dim for each item)
                num_heads_stacking=num_heads_stacking,
                target_l_new=target_l_new,       # Target L'
                d_new=d_new,                     # D'
                num_heads_latent=num_heads_latent
            )
        except ValueError as e:
            raise ValueError(f"Failed to initialize LMAConfigRL for LMAFeaturesExtractor: {e}") from e

        # The output features dimension from LMA is L_new * d_new
        # (after flattening the output of the LMA blocks)
        feature_dim_out = _lma_config_temp.L_new * _lma_config_temp.d_new
        
        super().__init__(observation_space, features_dim=feature_dim_out)
        
        self.lma_config = _lma_config_temp # Store the validated and computed config

        self.initial_transform = LMA_InitialTransform_RL(
            features_per_step=self.features_per_step, # F
            lma_config=self.lma_config,
            dropout=dropout,
            bias=bias
        )
        self.lma_blocks = nn.ModuleList([
            LMABlock_RL(
                lma_config=self.lma_config,
                ff_latent_hidden=ff_latent_hidden,
                dropout=dropout,
                bias=bias
            ) for _ in range(num_lma_layers)
        ])
        self.flatten = nn.Flatten() # Flattens (B, L_new, d_new) to (B, L_new * d_new)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes the flat input observations through the LMA architecture.

        Args:
            observations (torch.Tensor): Input tensor of shape (Batch, input_dim_total).

        Returns:
            torch.Tensor: Extracted features of shape (Batch, features_dim).
        """
        batch_size = observations.shape[0]
        # Reshape flat input (B, L*F) to (B, L, F)
        x_reshaped = observations.view(batch_size, self.seq_len, self.features_per_step)
        
        # Initial transformation and reshaping: (B, L, F) -> (B, L_new, d_new)
        z = self.initial_transform(x_reshaped)
        
        # Pass through LMA blocks
        for block in self.lma_blocks:
            z = block(z)
            
        # Flatten the output for the policy network
        features = self.flatten(z) # (B, L_new * d_new)
        return features


class Transformer(BaseFeaturesExtractor): # Standard Transformer Encoder
    """
    A standard Transformer Encoder features extractor for sequence data.
    The input `observation_space` is expected to be 1D, which will be reshaped
    into a sequence of `seq_len` items, each with `embed_dim` features after an
    initial embedding.

    Args:
        observation_space (gymnasium_spaces.Box): The observation space.
        embed_dim (int): Dimension of the embeddings for each item in the sequence.
        num_heads (int): Number of attention heads in the Transformer encoder layers.
        ff_hidden (int): Dimension of the feed-forward network hidden layer in encoder layers.
        num_layers (int): Number of Transformer encoder layers.
        seq_len (int): The sequence length to reshape the input observation into.
                       `observation_space.shape[0]` must be divisible by `seq_len`.
        dropout (float): Dropout probability.
    """
    def __init__(self,
                 observation_space: gymnasium_spaces.Box,
                 embed_dim: int = 64,
                 num_heads: int = 4, # Changed from 3 to be divisible by common embed_dims
                 ff_hidden: int = 128,
                 num_layers: int = 4,
                 seq_len: int = 6,
                 dropout: float = 0.3): # Corrected default dropout from 0.3 (typo)

        if not isinstance(observation_space, gymnasium_spaces.Box):
            print(f"Warning: Transformer extractor received observation_space of type "
                  f"{type(observation_space)}, expected gymnasium.spaces.Box.")
        if len(observation_space.shape) != 1:
             raise ValueError(
                f"Transformer extractor expects a 1D observation space (flattened sequence), "
                f"but got shape {observation_space.shape}."
            )

        self.input_dim_total = observation_space.shape[0] # Corrected from self.input_dim
        self.seq_len = seq_len
        if self.input_dim_total % seq_len != 0:
            raise ValueError(
                f"Total input dimension ({self.input_dim_total}) must be divisible by "
                f"seq_len ({seq_len}) for reshaping."
            )
        self.features_per_step = self.input_dim_total // seq_len

        # Output dimension will be seq_len * embed_dim after transformer processing
        feature_dim_out = seq_len * embed_dim
        super().__init__(observation_space, features_dim=feature_dim_out)

        self.embed_dim = embed_dim # Store for positional encoding
        self.dropout_p = dropout

        self.input_embedding = nn.Linear(self.features_per_step, embed_dim)
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # Configure TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=self.dropout_p,
            batch_first=True # Expects (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten() # Flattens (B, seq_len, embed_dim) to (B, seq_len * embed_dim)

    def _positional_encoding(self, seq_len: int, embed_dim: int) -> torch.Tensor:
        """Generates sinusoidal positional encodings."""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe # Shape: (seq_len, embed_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes the flat input observations through the Transformer Encoder.

        Args:
            observations (torch.Tensor): Input tensor of shape (Batch, input_dim_total).

        Returns:
            torch.Tensor: Extracted features of shape (Batch, features_dim).
        """
        batch_size = observations.shape[0]
        # Reshape flat input (B, L*F) to (B, L, F)
        x_reshaped = observations.view(batch_size, self.seq_len, self.features_per_step)

        # Embed and add positional encoding
        x_embedded = self.input_embedding(x_reshaped) # (B, L, embed_dim)
        x_embedded = x_embedded + self._positional_encoding(self.seq_len, self.embed_dim).to(x_embedded.device)
        x_processed = self.embedding_dropout(x_embedded)

        # Pass through Transformer encoder
        x_transformed = self.transformer_encoder(x_processed) # (B, L, embed_dim)

        # Flatten the output
        features = self.flatten(x_transformed) # (B, L * embed_dim)
        return features


# ===============================================
# Stacked LMA Feature Extractor
# ===============================================
class StackedLMAFeaturesExtractor(BaseFeaturesExtractor):
    """
    A two-stage features extractor for environments with stacked frame observations.
    It first applies a `JSBSimFeatureExtractor` (or any per-frame extractor) to each
    frame in the input sequence, and then processes the sequence of these extracted
    features using an `LMAFeaturesExtractor`.

    The input `observation_space` is expected to be a 2D Box space of shape
    (num_stacked_frames, raw_features_per_frame).

    Args:
        observation_space (gymnasium_spaces.Box): The 2D observation space from the environment.
        lma_embed_dim_d0 (int): `embed_dim` (D0) for the subsequent LMAFeaturesExtractor.
        lma_num_heads_stacking (int): `num_heads_stacking` for LMA.
        lma_num_heads_latent (int): `num_heads_latent` for LMA.
        lma_ff_latent_hidden (int): `ff_latent_hidden` for LMA's MLP.
        lma_num_layers (int): `num_lma_layers` for LMA.
        lma_dropout (float): Dropout rate for LMA.
        lma_bias (bool): Whether to use bias in LMA's linear layers.
    """
    def __init__(self,
                 observation_space: gymnasium_spaces.Box,
                 lma_embed_dim_d0: int = 64,
                 lma_num_heads_stacking: int = 4,
                 lma_num_heads_latent: int = 4,
                 lma_ff_latent_hidden: int = 128,
                 lma_num_layers: int = 2,
                 lma_dropout: float = 0.1,
                 lma_bias: bool = True):

        assert isinstance(observation_space, gymnasium_spaces.Box), \
            f"Expected gymnasium.spaces.Box, got {type(observation_space)}"
        assert len(observation_space.shape) == 2, \
            "Observation space for StackedLMAFeaturesExtractor must be 2D (num_stacked_frames, raw_features_per_frame)"

        self.env_num_stacked_frames = observation_space.shape[0]  # L_env (e.g., 10 from environment)
        self.env_raw_obs_dim_per_frame = observation_space.shape[1] # F_raw (e.g., 15 from environment state)

        # 1. Determine output dimension of the per-frame feature extractor (JSBSimFeatureExtractor)
        # Create a temporary 1D observation space representing a single frame
        _single_frame_gym_space = gymnasium_spaces.Box(
            low=observation_space.low[0],    # Low bounds of the first frame
            high=observation_space.high[0],  # High bounds of the first frame
            shape=(self.env_raw_obs_dim_per_frame,),
            dtype=observation_space.dtype
        )
        # Instantiate the per-frame extractor temporarily to get its output dimension
        _temp_jsb_extractor = JSBSimFeatureExtractor(_single_frame_gym_space)
        self.features_per_frame_after_jsb = _temp_jsb_extractor.features_dim # F_jsb
        del _temp_jsb_extractor # Clean up

        # 2. Determine output dimension of the LMAFeaturesExtractor
        # The LMAFeaturesExtractor will take a sequence of features_per_frame_after_jsb.
        # Its input sequence length will be env_num_stacked_frames.
        _lma_input_seq_len_L = self.env_num_stacked_frames # This is `seq_len` for LMAFeaturesExtractor
        
        # The LMAFeaturesExtractor expects a flat 1D observation space as input.
        # Total features fed to LMA (flat): L_env * F_jsb
        _lma_total_input_features_flat = _lma_input_seq_len_L * self.features_per_frame_after_jsb
        
        _dummy_lma_flat_obs_space = gymnasium_spaces.Box(
            low=-np.inf, high=np.inf, shape=(_lma_total_input_features_flat,), dtype=observation_space.dtype
        )

        # Instantiate LMAFeaturesExtractor temporarily to get its final output dimension
        # Hyperparameters for LMA's internal configuration:
        # `target_l_new` and `d_new` for LMA are relative to its own inputs.
        # A common choice is to reduce sequence length and embedding dimension.
        _lma_target_l_new = _lma_input_seq_len_L // 2  # Example: Halve the sequence length
        _lma_target_d_new = lma_embed_dim_d0 // 2      # Example: Halve the D0 dimension to get D'

        _temp_lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space,
            embed_dim=lma_embed_dim_d0,             # D0 for LMA
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_new,         # Target L' for LMA
            d_new=_lma_target_d_new,                # D' for LMA
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=_lma_input_seq_len_L,           # L for LMA (which is env_num_stacked_frames)
            dropout=lma_dropout,
            bias=lma_bias
        )
        final_features_dim = _temp_lma_extractor.features_dim
        del _temp_lma_extractor # Clean up

        # Call super().__init__() with the original 2D observation_space
        # and the calculated final_features_dim.
        super().__init__(observation_space, features_dim=final_features_dim)

        # Now, properly initialize and store the nn.Module attributes
        self.jsb_feature_extractor = JSBSimFeatureExtractor(_single_frame_gym_space)
        
        self.lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space, # Use the same dummy space for consistency
            embed_dim=lma_embed_dim_d0,
            num_heads_stacking=lma_num_heads_stacking,
            target_l_new=_lma_target_l_new,
            d_new=_lma_target_d_new,
            num_heads_latent=lma_num_heads_latent,
            ff_latent_hidden=lma_ff_latent_hidden,
            num_lma_layers=lma_num_layers,
            seq_len=_lma_input_seq_len_L,
            dropout=lma_dropout,
            bias=lma_bias
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes the stacked observations through the two-stage feature extraction.

        Args:
            observations (torch.Tensor): Input tensor of shape
                (Batch, env_num_stacked_frames, env_raw_obs_dim_per_frame).

        Returns:
            torch.Tensor: Extracted features of shape (Batch, final_features_dim).
        """
        batch_size = observations.shape[0]
        
        # Reshape for per-frame JSBSimFeatureExtractor:
        # (B, L_env, F_raw) -> (B * L_env, F_raw)
        obs_reshaped_for_jsb = observations.reshape(
            batch_size * self.env_num_stacked_frames,
            self.env_raw_obs_dim_per_frame
        )
        
        # Extract features per frame: (B * L_env, F_raw) -> (B * L_env, F_jsb)
        extracted_features_jsb = self.jsb_feature_extractor(obs_reshaped_for_jsb)
        
        # Reshape for LMAFeaturesExtractor (which expects a flat input):
        # (B * L_env, F_jsb) -> (B, L_env * F_jsb)
        processed_obs_for_lma_flat = extracted_features_jsb.reshape(
            batch_size,
            self.env_num_stacked_frames * self.features_per_frame_after_jsb
        )
        
        # LMA processes the sequence of jsb_features: (B, L_env * F_jsb) -> (B, final_features_dim)
        final_features = self.lma_extractor(processed_obs_for_lma_flat)
        
        return final_features