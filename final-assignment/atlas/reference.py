"""Reference segmenter for the generative track's mask Dice.

Inference-only copy of the ATLAS model (de Jong et al., 2026; https://github.com/rlpddejong/ATLAS-model,
MIT licence): an encoder-only mask transformer (EoMT) on a DINOv3 ViT-B/16 backbone, trained on ATLAS-120k
with the same 30 train IDs as this repository. Needs the `transformers` package.

Weights: a plain state dict made with `python -m tools.export_reference_weights` from the
`ATLAS_vitb_*.ckpt` checkpoint on https://huggingface.co/rlpddejong/ATLAS-finetuned-weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.classes import NUM_CLASSES

CROP = 256  # the model was trained on 256x256 crops

# DINOv3 ViT-B/16 with 4 register tokens; built from this config, so no download is needed.
DINOV3_VITB16 = dict(
    patch_size=16, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12,
    num_register_tokens=4, image_size=CROP, layerscale_value=1.0, use_gated_mlp=False,
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True,
)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a (B, C, H, W) tensor (same parameters as timm's LayerNorm2d)."""

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ScaleBlock(nn.Module):
    """2x upsampling: transposed conv -> GELU -> depthwise conv -> LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim, eps=1e-6)  # timm default

    def forward(self, x):
        return self.norm(self.conv2(self.act(self.conv1(x))))


class Encoder(nn.Module):
    """DINOv3 backbone with the attribute names the ATLAS code expects."""

    def __init__(self):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        config = DINOv3ViTConfig(**DINOV3_VITB16, attn_implementation="sdpa")  # sdpa accepts boolean masks
        backbone = DINOv3ViTModel(config)
        backbone.patch_embed = backbone.embeddings          # cls + register + patch tokens
        backbone.patch_embed.patch_size = (config.patch_size, config.patch_size)
        backbone.patch_embed.grid_size = (CROP // config.patch_size, CROP // config.patch_size)
        backbone.embed_dim = config.hidden_size
        backbone.num_prefix_tokens = config.num_register_tokens + 1
        # transformers 4.x stores the blocks in `layer`, 5.x in `model.layer`
        encoder = backbone.model if hasattr(backbone, "model") else backbone
        backbone.blocks = encoder.layer
        del backbone.patch_embed.mask_token, backbone.embeddings, encoder.layer
        self.backbone = backbone
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))  # loaded from the checkpoint
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))


class ATLAS(nn.Module):
    """The ATLAS network. `forward` returns the final mask logits (B, Q, h, w) and class logits (B, Q, C + 1).

    The procedure and temporal heads are kept so the checkpoint loads strictly, but are not used here.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_q=100, num_blocks=4, num_procedures=14, phase_num_tokens=4):
        super().__init__()
        self.encoder = Encoder()
        dim = self.encoder.backbone.embed_dim
        self.num_q, self.num_blocks = num_q, num_blocks
        self.num_proc_q, self.num_temporal_tokens = 1, phase_num_tokens
        self.total_q = num_q + self.num_proc_q + phase_num_tokens
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, dim)
        self.query_updater = nn.Linear(dim, dim)
        self.proc_q = nn.Embedding(self.num_proc_q, dim)
        self.proc_head = nn.Linear(dim, num_procedures)
        self.temporal_q = nn.Embedding(phase_num_tokens, dim)
        self.temporal_time_mlp = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.temporal_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 256))
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.mask_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        num_upscale = max(1, int(math.log2(self.encoder.backbone.patch_embed.patch_size[0])) - 2)
        self.upscale = nn.Sequential(*[ScaleBlock(dim) for _ in range(num_upscale)])

    def _predict(self, x):
        backbone = self.encoder.backbone
        q = x[:, self.num_proc_q:self.num_proc_q + self.num_q]
        patches = x[:, self.total_q + backbone.num_prefix_tokens:]
        patches = patches.transpose(1, 2).reshape(x.shape[0], -1, *backbone.patch_embed.grid_size)
        mask_logits = torch.einsum("bqc,bchw->bqhw", self.mask_head(q), self.upscale(patches))
        return mask_logits, self.class_head(q)

    def _attn_mask(self, x, mask_logits, prob):
        """Mask queries' attention to patches outside their predicted mask.

        As in the original evaluation, each query keeps its mask with probability `prob`. The random
        choice uses a fixed seed and is the same for every image, so scores are reproducible.
        """
        backbone = self.encoder.backbone
        mask = torch.ones(x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        q0, q1 = self.num_proc_q, self.num_proc_q + self.num_q
        p0 = self.total_q + backbone.num_prefix_tokens
        inside = F.interpolate(mask_logits, backbone.patch_embed.grid_size, mode="bilinear") > 0
        mask[:, q0:q1, p0:] = inside.flatten(2)
        if prob < 1:
            generator = torch.Generator().manual_seed(0)
            unmasked = (torch.rand(self.num_q, generator=generator) > prob).to(x.device)
            mask[:, q0:q1, p0:][:, unmasked] = True
        return mask

    def forward(self, x):
        backbone = self.encoder.backbone
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std
        rope = backbone.rope_embeddings(x)
        x = backbone.patch_embed(x)
        n_blocks, mask = len(backbone.blocks), None
        for i, block in enumerate(backbone.blocks):
            if i == n_blocks - self.num_blocks:
                b = x.shape[0]
                # [procedure query, mask queries, temporal queries (time index 0)] + image tokens
                temporal = self.temporal_q.weight[None] + self.temporal_time_mlp(torch.zeros(b, 1, device=x.device))[:, None]
                queries = [self.proc_q.weight[None].expand(b, -1, -1), self.q.weight[None].expand(b, -1, -1), temporal]
                x = torch.cat([*queries, x], dim=1)
            if i >= n_blocks - self.num_blocks:
                mask_logits, _ = self._predict(backbone.norm(x))
                mask = self._attn_mask(x, mask_logits, float(self.attn_mask_probs[i - n_blocks + self.num_blocks]))
            heads_mask = mask[:, None].expand(-1, block.attention.num_heads, -1, -1) if mask is not None else None
            x = x + block.layer_scale1(block.attention(block.norm1(x), heads_mask, rope)[0])
            x = x + block.layer_scale2(block.mlp(block.norm2(x)))
        return self._predict(backbone.norm(x))


class ReferenceSegmenter(nn.Module):
    """Segments frames with ATLAS. Input: (N, 3, H, W) RGB in [0, 1]. Output: (N, 30, H, W) class scores.

    Like the original evaluation, the frame is scaled so its short side is 256, split into
    overlapping 256x256 windows, and the window predictions are averaged.
    """

    def __init__(self, weights: str, device):
        super().__init__()
        self.net = ATLAS()
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, images):
        n, _, h, w = images.shape
        scale = CROP / min(h, w)
        sh, sw = round(h * scale), round(w * scale)
        x = F.interpolate(images, size=(sh, sw), mode="bilinear", antialias=True, align_corners=False)

        # windows along the long side, evenly overlapping
        long_side = max(sh, sw)
        n_win = math.ceil(long_side / CROP)
        step = (long_side - CROP) / (n_win - 1) if n_win > 1 else 0
        starts = [round(j * step) for j in range(n_win)]
        scores = torch.zeros(n, NUM_CLASSES, sh, sw, device=images.device)
        counts = torch.zeros(1, 1, sh, sw, device=images.device)
        for s in starts:
            window = (slice(None), slice(None), slice(s, s + CROP), slice(None)) if sh > sw else \
                     (slice(None), slice(None), slice(None), slice(s, s + CROP))
            mask_logits, class_logits = self.net(x[window])
            mask_logits = F.interpolate(mask_logits, size=(CROP, CROP), mode="bilinear")
            scores[window] += torch.einsum("bqhw,bqc->bchw", mask_logits.sigmoid(), class_logits.softmax(-1)[..., :-1])
            counts[window] += 1
        return F.interpolate(scores / counts, size=(h, w), mode="bilinear", align_corners=False)
