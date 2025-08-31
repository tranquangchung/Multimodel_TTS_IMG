import torch
import math
import matplotlib.pyplot as plt
from typing import Optional


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2).float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)

    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def precompute_freqs_cis_2d(
    grid_size: int,
    n_elem: int,
    base: int = 10000,
    cls_token_num: int = 120,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)

    freqs_grid = torch.cat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)

    freqs_cis = torch.polar(torch.ones_like(freqs_grid), freqs_grid)
    cache_grid = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    cache = cache_grid.flatten(0, 1)
    cls_padding = torch.zeros(cls_token_num, n_elem // 2, 2, device=cache.device)
    full_cache = torch.cat([cls_padding, cache], dim=0)
    return full_cache.to(dtype=dtype)


def visualize_rope_1d(seq_len=128, n_elem=64, rope_scaling=None):
    rope = precompute_freqs_cis(seq_len=seq_len, n_elem=n_elem, rope_scaling=rope_scaling, dtype=torch.float32)
    real = rope[..., 0]
    imag = rope[..., 1]
    phase = torch.atan2(imag, real)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(real.T, aspect='auto', cmap='coolwarm')
    axs[0].set_title("1D RoPE - Real Part")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Freq Index")

    axs[1].imshow(imag.T, aspect='auto', cmap='coolwarm')
    axs[1].set_title("1D RoPE - Imag Part")
    axs[1].set_xlabel("Position")

    axs[2].imshow(phase.T, aspect='auto', cmap='twilight')
    axs[2].set_title("1D RoPE - Phase")
    axs[2].set_xlabel("Position")

    plt.tight_layout()
    plt.savefig("rope_1d_visualization.png")


def visualize_rope_2d(grid_size=16, n_elem=64, cls_token_num=120):
    rope = precompute_freqs_cis_2d(grid_size=grid_size, n_elem=n_elem, cls_token_num=cls_token_num, dtype=torch.float32)
    real = rope[..., 0]
    imag = rope[..., 1]
    phase = torch.atan2(imag, real)

    valid_real = real[cls_token_num:]  # exclude CLS
    valid_imag = imag[cls_token_num:]
    valid_phase = phase[cls_token_num:]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(valid_real.T, aspect='auto', cmap='coolwarm')
    axs[0].set_title("2D RoPE - Real Part (Flattened Grid)")
    axs[0].set_xlabel("Patch Index")
    axs[0].set_ylabel("Freq Index")

    axs[1].imshow(valid_imag.T, aspect='auto', cmap='coolwarm')
    axs[1].set_title("2D RoPE - Imag Part (Flattened Grid)")
    axs[1].set_xlabel("Patch Index")

    axs[2].imshow(valid_phase.T, aspect='auto', cmap='twilight')
    axs[2].set_title("2D RoPE - Phase")
    axs[2].set_xlabel("Patch Index")

    plt.tight_layout()
    plt.savefig("rope_2d_visualization.png")


if __name__ == "__main__":
    # Visualize both 1D and 2D RoPE
    visualize_rope_1d(seq_len=128, n_elem=64)

    visualize_rope_2d(grid_size=16, n_elem=64, cls_token_num=120)
