"""
Llama 3 Model Architecture Implementation
Full implementation with RoPE, GQA, SwiGLU, RMSNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Llama 3 Model Configuration"""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4  # GQA için
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.1
    
    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, head_dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len, device):
        """Cache cos/sin values"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
    
    def forward(self, q, k, seq_len):
        """Apply rotary embeddings"""
        self._update_cache(seq_len, q.device)
        
        cos = self._cos_cached[:, :, :seq_len, ...]
        sin = self._sin_cached[:, :, :seq_len, ...]
        
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rot, k_rot
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rotary_emb(q, k, seq_len)
        
        # Repeat KV for GQA
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class SwiGLU(nn.Module):
    """SwiGLU Activation Function"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Llama Transformer Decoder Layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = SwiGLU(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LlamaModel(nn.Module):
    """Llama 3 Language Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("LLAMA 3 MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Small config for testing
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        max_position_embeddings=2048
    )
    
    model = LlamaModel(config)
    
    print(f"\nModel Configuration:")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads (GQA): {config.num_key_value_heads}")
    print(f"  Intermediate size: {config.intermediate_size:,}")
    print(f"  Max sequence length: {config.max_position_embeddings:,}")
    
    print(f"\nTotal parameters: {model.num_parameters():,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print(f"\n[OK] Model architecture works!")