import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(tensor: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: x * (1 + scale) + shift, broadcasting over the token dim."""
    return tensor * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Chop an image into non-overlapping patches and project each to a token vector."""

    def __init__(self, image_channel: int, patch_size: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Conv2d(image_channel, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.proj(tensor)  # (B, hidden, H/patch, W/patch)
        tensor = tensor.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        return tensor


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
        )
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class LabelEmbedder(nn.Module):
    """
    Embed class labels, with a dedicated null token for classifier-free guidance.

    During training, a fraction of labels are randomly replaced with the null token
    so that the same network learns both the conditional and unconditional score.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        # The extra index (num_classes) is the null / unconditional token.
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop_mask, self.num_classes, labels)

    def forward(self, labels: torch.Tensor, train: bool) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


class Attention(nn.Module):
    """Standard multi-head self-attention over the patch tokens."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        B, N, C = tensor.shape
        qkv = self.qkv(tensor).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        query, key, value = qkv.unbind(0)
        out = F.scaled_dot_product_attention(query, key, value)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    """A DiT transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Produces the 6 modulation vectors (shift/scale/gate for attention and MLP).
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(condition).chunk(6, dim=1)
        )
        tensor = tensor + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(tensor), shift_msa, scale_msa)
        )
        tensor = tensor + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(tensor), shift_mlp, scale_mlp)
        )
        return tensor


class FinalLayer(nn.Module):
    """Final adaLN layer projecting tokens back to flattened patch pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(condition).chunk(2, dim=1)
        tensor = modulate(self.norm_final(tensor), shift, scale)
        return self.linear(tensor)


class DiT(nn.Module):
    """
    Pixel-space Diffusion Transformer (class-conditional, adaLN-Zero), DiT paper style.

    The network predicts the noise (epsilon) added to an image at a given timestep.
    """

    def __init__(
        self,
        image_size: int = 32,
        image_channel: int = 3,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}.")

        self.image_size = image_size
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.grid_size = image_size // patch_size

        self.patch_embed = PatchEmbed(image_channel, patch_size, hidden_size)
        self.timestep_embed = TimestepEmbedder(hidden_size)
        self.label_embed = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.grid_size * self.grid_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, image_channel)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)

        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize the patch projection like a linear layer.
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        nn.init.normal_(self.label_embed.embedding_table.weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)

        # adaLN-Zero: start every block as an identity by zeroing the modulation output.
        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, tensor: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch**2 * C) -> (B, C, H, W)."""
        B = tensor.shape[0]
        patch = self.patch_size
        grid = self.grid_size
        channel = self.image_channel
        tensor = tensor.reshape(B, grid, grid, patch, patch, channel)
        tensor = torch.einsum("bhwpqc->bchpwq", tensor)
        return tensor.reshape(B, channel, grid * patch, grid * patch)

    def forward(self, tensor: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: Noised images, shape (B, C, H, W).
        :param timestep: Diffusion timesteps, shape (B,).
        :param labels: Class labels, shape (B,). Use index num_classes for unconditional.
        :return: Predicted noise, shape (B, C, H, W).
        """
        tensor = self.patch_embed(tensor) + self.pos_embed
        condition = self.timestep_embed(timestep) + self.label_embed(labels, self.training)
        for block in self.blocks:
            tensor = block(tensor, condition)
        tensor = self.final_layer(tensor, condition)
        return self.unpatchify(tensor)
