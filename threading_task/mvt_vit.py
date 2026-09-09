"""ViT used by the original ARP real-robot MVT model."""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange


class _Attention(nn.Module):
    """Paper implementation: eight 64-D heads even though model width is 128."""
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale = heads, dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attend, self.dropout = nn.Softmax(dim=-1), nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)]
        out = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)) @ v
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = _Attention(dim, heads, dim // 2, dropout)
        self.ff = _FeedForward(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x) + x
        return self.ff(x) + x


class MVTVisualTransformer(nn.Module):
    def __init__(self, num_patches: int, dim: int = 128, depth: int = 8,
                 heads: int = 8, mlp_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_Block(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls_token.expand(len(patches), -1, -1)
        x = torch.cat((cls, patches), dim=1)
        x = self.dropout(x + self.pos_embedding[:, :x.shape[1]])
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
