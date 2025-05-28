import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces as gymnasium_spaces # NEW IMPORT for SB3 compatibility

try:
    from features import JSBSimFeatureExtractor 
except ImportError:
    from .features import JSBSimFeatureExtractor 


#===============================================
# LMA Helper Function (Unchanged)
#===============================================
def find_closest_divisor(total_value, target_divisor, max_delta=100):
    if not isinstance(total_value, int) or total_value <= 0: raise ValueError(f"total_value ({total_value}) must be positive integer.")
    if not isinstance(target_divisor, int) or target_divisor <= 0: target_divisor = max(1, target_divisor)
    if not isinstance(max_delta, int) or max_delta < 0: raise ValueError(f"max_delta ({max_delta}) must be non-negative.")
    if total_value == 0: return 1
    if target_divisor > 0 and total_value % target_divisor == 0: return target_divisor
    search_start = max(1, target_divisor)
    for delta in range(1, max_delta + 1):
        candidate_minus = search_start - delta
        if candidate_minus > 0 and total_value % candidate_minus == 0: return candidate_minus
        candidate_plus = search_start + delta
        if candidate_plus > 0 and total_value % candidate_plus == 0: return candidate_plus
    print(f"Warning: No divisor found near {target_divisor} for {total_value}. Searching all divisors.")
    best_divisor = 1
    min_diff = abs(target_divisor - 1)
    for i in range(2, int(math.sqrt(total_value)) + 1):
        if total_value % i == 0:
            div1 = i; div2 = total_value // i
            diff1 = abs(target_divisor - div1); diff2 = abs(target_divisor - div2)
            if diff1 < min_diff: min_diff = diff1; best_divisor = div1
            if diff2 < min_diff: min_diff = diff2; best_divisor = div2
    diff_total = abs(target_divisor - total_value)
    if diff_total < min_diff: best_divisor = total_value
    print(f"Using {best_divisor} as fallback divisor.")
    return best_divisor

#===============================================
# LMA Feature Extractor Implementation (Unchanged LMA Core)
#===============================================

@dataclass
class LMAConfigRL:
    seq_len: int; embed_dim: int; num_heads_stacking: int
    target_l_new: int; d_new: int; num_heads_latent: int
    L_new: int = field(init=False); C_new: int = field(init=False)
    def __post_init__(self):
        if self.seq_len <= 0 or self.embed_dim <= 0 or self.num_heads_stacking <= 0 or \
           self.target_l_new <= 0 or self.d_new <= 0 or self.num_heads_latent <= 0:
            raise ValueError("LMAConfigRL inputs must be positive.")
        if self.embed_dim % self.num_heads_stacking != 0:
            raise ValueError(f"LMA embed_dim ({self.embed_dim}) not divisible by num_heads_stacking ({self.num_heads_stacking})")
        if self.d_new % self.num_heads_latent != 0:
            raise ValueError(f"LMA d_new ({self.d_new}) not divisible by num_heads_latent ({self.num_heads_latent})")
        total_features = self.seq_len * self.embed_dim
        if total_features == 0: raise ValueError("LMA total features cannot be zero.")
        try:
            self.L_new = find_closest_divisor(total_features, self.target_l_new)
            if self.L_new != self.target_l_new: print(f"LMAConfigRL ADJUSTMENT: L_new {self.target_l_new} -> {self.L_new}")
            if self.L_new <= 0: raise ValueError("Calculated L_new is not positive.")
            if total_features % self.L_new != 0: raise RuntimeError(f"Internal Error: total_features ({total_features}) not divisible by final L_new ({self.L_new})")
            self.C_new = total_features // self.L_new
            if self.C_new <= 0: raise ValueError("Calculated C_new is not positive.")
        except ValueError as e: raise ValueError(f"LMA Config Error calculating L_new/C_new: {e}") from e

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__(); self.weight = nn.Parameter(torch.ones(ndim)); self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input): return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LMA_InitialTransform_RL(nn.Module):
    def __init__(self, features_per_step: int, lma_config: LMAConfigRL, dropout: float, bias: bool):
        super().__init__(); self.lma_config = lma_config
        self.input_embedding = nn.Linear(features_per_step, lma_config.embed_dim, bias=bias)
        self.input_embedding_act = nn.ReLU(); self.embedding_dropout = nn.Dropout(p=dropout)
        self.embed_layer_2 = nn.Linear(lma_config.C_new, lma_config.d_new, bias=bias)
        self.embed_layer_2_act = nn.ReLU()
    def forward(self, x): 
        B, L, _ = x.shape
        if L != self.lma_config.seq_len: raise ValueError(f"Input seq len {L} != LMA config seq_len {self.lma_config.seq_len}")
        y = self.input_embedding_act(self.input_embedding(x))
        y = y + self._positional_encoding(L, self.lma_config.embed_dim).to(y.device)
        y = self.embedding_dropout(y)
        d0 = self.lma_config.embed_dim; nh = self.lma_config.num_heads_stacking; dk = d0 // nh
        head_views = torch.split(y, dk, dim=2); x_stacked = torch.cat(head_views, dim=1)
        L_new = self.lma_config.L_new; C_new = self.lma_config.C_new
        x_flat = x_stacked.view(B, -1)
        x_rechunked = x_flat.view(B, L_new, C_new)
        z = self.embed_layer_2_act(self.embed_layer_2(x_rechunked))
        return z
    def _positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class LatentAttention_RL(nn.Module):
    def __init__(self, d_new: int, num_heads_latent: int, dropout: float, bias: bool):
        super().__init__(); assert d_new % num_heads_latent == 0
        self.d_new = d_new; self.num_heads = num_heads_latent; self.head_dim = d_new // num_heads_latent
        self.c_attn = nn.Linear(d_new, 3 * d_new, bias=bias); self.c_proj = nn.Linear(d_new, d_new, bias=bias)
        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout 
        self.flash = hasattr(F, 'scaled_dot_product_attention')
    def forward(self, z):
        B, L_new, C = z.size(); q, k, v = self.c_attn(z).split(self.d_new, dim=2)
        q = q.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_new, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash: y = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout_p if self.training else 0,is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)); att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att); y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L_new, self.d_new)
        y = self.resid_dropout(self.c_proj(y)); return y

class LatentMLP_RL(nn.Module):
    def __init__(self, d_new: int, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.c_fc = nn.Linear(d_new, ff_latent_hidden, bias=bias); self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ff_latent_hidden, d_new, bias=bias); self.dropout = nn.Dropout(dropout)
    def forward(self, x): x = self.c_fc(x); x = self.gelu(x); x = self.c_proj(x); x = self.dropout(x); return x

class LMABlock_RL(nn.Module):
    def __init__(self, lma_config: LMAConfigRL, ff_latent_hidden: int, dropout: float, bias: bool):
        super().__init__(); self.ln_1 = LayerNorm(lma_config.d_new, bias=bias)
        self.attn = LatentAttention_RL(lma_config.d_new, lma_config.num_heads_latent, dropout, bias)
        self.ln_2 = LayerNorm(lma_config.d_new, bias=bias)
        self.mlp = LatentMLP_RL(lma_config.d_new, ff_latent_hidden, dropout, bias)
    def forward(self, z): z = z + self.attn(self.ln_1(z)); z = z + self.mlp(self.ln_2(z)); return z

class LMAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__( self, observation_space, embed_dim=64, num_heads_stacking=4, target_l_new=3, d_new=32,
        num_heads_latent=4, ff_latent_hidden=64, num_lma_layers=2, seq_len=6, dropout=0.1, bias=True ):
        
        # Determine feature_dim first
        _lma_config_temp = LMAConfigRL( # Use a temporary config to get L_new, d_new
            seq_len=seq_len, embed_dim=embed_dim, num_heads_stacking=num_heads_stacking,
            target_l_new=target_l_new, d_new=d_new, num_heads_latent=num_heads_latent )
        feature_dim = _lma_config_temp.L_new * _lma_config_temp.d_new
        
        super().__init__(observation_space, features_dim=feature_dim) # Call super() early
        
        self.lma_config = _lma_config_temp # Assign the already created config

        if not isinstance(observation_space, gymnasium_spaces.Box):
            print(f"Warning: LMAFeaturesExtractor received an observation_space of type {type(observation_space)}, expected gymnasium.spaces.Box.")
        self.input_dim_total = observation_space.shape[0] 
        self.seq_len = seq_len 
        if self.input_dim_total % seq_len != 0: raise ValueError(f"Input dim ({self.input_dim_total}) not div by seq_len ({seq_len}).")
        self.features_per_step = self.input_dim_total // seq_len 
        self.initial_transform = LMA_InitialTransform_RL( self.features_per_step, self.lma_config, dropout, bias )
        self.lma_blocks = nn.ModuleList([ LMABlock_RL( self.lma_config, ff_latent_hidden, dropout, bias ) for _ in range(num_lma_layers) ])
        self.flatten = nn.Flatten()
    def forward(self, x): 
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.seq_len, self.features_per_step) 
        z = self.initial_transform(x_reshaped) 
        for block in self.lma_blocks: z = block(z)
        features = self.flatten(z); return features

class Transformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, num_heads=3, ff_hidden=128, num_layers=4, seq_len=6, dropout=0.3):
        feature_dim = embed_dim * seq_len
        super(Transformer, self).__init__(observation_space, features_dim=feature_dim) # Call super() early
        self.embed_dim = embed_dim 
        if not isinstance(observation_space, gymnasium_spaces.Box):
             print(f"Warning: Transformer received an observation_space of type {type(observation_space)}, expected gymnasium.spaces.Box.")
        self.input_dim = observation_space.shape[0]
        self.seq_len = seq_len; self.dropout_p = dropout
        if self.input_dim % seq_len != 0: raise ValueError("Input dimension must be divisible by seq_len.")
        self.input_embedding = nn.Linear(self.input_dim // seq_len, embed_dim)
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)
        self.encoder_layer = nn.TransformerEncoderLayer( d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_hidden, dropout=self.dropout_p, batch_first=True )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
    def forward(self, x):
        batch_size = x.shape[0]; features_per_seq = self.input_dim // self.seq_len
        x = x.view(batch_size, self.seq_len, features_per_seq)
        x = self.input_embedding(x)
        x = x + self._positional_encoding(self.seq_len, self.embed_dim).to(x.device)
        x = self.embedding_dropout(x); x = self.transformer(x); return self.flatten(x)
    def _positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp( torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim) )
        pe = torch.zeros(seq_len, embed_dim); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        return pe

#===============================================
# New Stacked LMA Feature Extractor
#===============================================
class StackedLMAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium_spaces.Box,
                 lma_embed_dim_d0=64,      
                 lma_num_heads_stacking=4, 
                 lma_num_heads_latent=4,   
                 lma_ff_latent_hidden=128, 
                 lma_num_layers=2,         
                 lma_dropout=0.1,
                 lma_bias=True):

        assert isinstance(observation_space, gymnasium_spaces.Box), \
            f"Expected gymnasium.spaces.Box, got {type(observation_space)}" 
        assert len(observation_space.shape) == 2, "Obs space for StackedLMA must be 2D (seq_len, raw_feats_per_frame)"
        
        _env_num_stacked_frames = observation_space.shape[0] 
        _env_raw_obs_dim_per_frame = observation_space.shape[1]

        # Create a temporary JSBSimFeatureExtractor to get its output dimension
        # This is needed to calculate final_features_dim before calling super().__init__
        _single_frame_gym_space = gymnasium_spaces.Box(
            low=observation_space.low[0], 
            high=observation_space.high[0],
            shape=(_env_raw_obs_dim_per_frame,),
            dtype=observation_space.dtype
        )
        # We don't assign this to self yet.
        _temp_jsb_extractor = JSBSimFeatureExtractor(_single_frame_gym_space)
        _features_per_frame_after_jsb = _temp_jsb_extractor.features_dim
        del _temp_jsb_extractor # No longer needed

        # Calculate parameters for LMAFeaturesExtractor
        _lma_input_seq_len_L = _env_num_stacked_frames      
        _lma_target_l_new = _lma_input_seq_len_L // 2            
        _lma_d0 = lma_embed_dim_d0                             
        _lma_target_d_new = _lma_d0 // 2                        

        _lma_total_input_features_flat = _lma_input_seq_len_L * _features_per_frame_after_jsb
        
        _dummy_lma_flat_obs_space = gymnasium_spaces.Box(
            low=-np.inf, high=np.inf, shape=(_lma_total_input_features_flat,), dtype=observation_space.dtype
        )

        # Create a temporary LMAFeaturesExtractor to get its output dimension (final_features_dim)
        # We don't assign this to self yet.
        _temp_lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space,
            embed_dim=_lma_d0, 
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
        final_features_dim = _temp_lma_extractor.features_dim
        del _temp_lma_extractor # No longer needed

        # NOW CALL SUPER().__INIT__()
        super().__init__(observation_space, features_dim=final_features_dim)

        # Store actual dimensions for forward pass
        self.env_num_stacked_frames = _env_num_stacked_frames
        self.env_raw_obs_dim_per_frame = _env_raw_obs_dim_per_frame
        self.features_per_frame_after_jsb = _features_per_frame_after_jsb

        # Now, properly initialize and assign the nn.Module attributes
        self.jsb_feature_extractor = JSBSimFeatureExtractor(_single_frame_gym_space)
        
        self.lma_extractor = LMAFeaturesExtractor(
            observation_space=_dummy_lma_flat_obs_space, # Use the same dummy space
            embed_dim=_lma_d0, 
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
        batch_size = observations.shape[0]
        
        obs_reshaped_for_jsb = observations.reshape(
            batch_size * self.env_num_stacked_frames, 
            self.env_raw_obs_dim_per_frame
        )
        
        extracted_features_jsb = self.jsb_feature_extractor(obs_reshaped_for_jsb)
        
        processed_obs_for_lma_flat = extracted_features_jsb.reshape(
            batch_size,
            self.env_num_stacked_frames * self.features_per_frame_after_jsb
        )
        
        final_features = self.lma_extractor(processed_obs_for_lma_flat)
        
        return final_features