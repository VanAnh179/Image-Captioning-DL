"""Frozen CLIP ViT-B/32 region features, cached with image IDs in the same layout as the ResNet-50 cache.

What differs from src/data/region_features.py (ResNet-50): the encoder, the image preprocessing (the one shipped with the
CLIP weights), the channel count (768 instead of 2048) and the cache file name. The region layout (7 x 7 = 49 regions,
region = row * 7 + column), the ID bookkeeping, the pixel-hash check against the manifest and RegionFeatureStore are shared.
"""

from functools import partial

import torch
from PIL import Image, ImageOps
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModel

from src.data.region_features import GRID, NUM_REGIONS, pixel_sha256

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"  # commit of the Hugging Face repo whose weights and preprocessor were used
CLIP_FEATURE_DIM = 768
CLIP_CACHE_PREFIX = "clip_vit_b32_7x7"
FORMAT_VERSION = 1


def load_clip_processor(model_name=CLIP_MODEL, revision=CLIP_REVISION):
    """The image processor stored with the weights: bicubic resize of the short side to 224, centre crop 224, CLIP mean/std."""
    return CLIPImageProcessor.from_pretrained(model_name, revision=revision)


def load_clip_image(path, processor):
    """Return (float32 [3, 224, 224] after the CLIP preprocessing, pixel hash of the untransposed RGB image).

    The hash is defined like the ResNet loader's so it can be compared with the manifest. The EXIF transpose is the same
    step model B applies before resizing (no KTVIC image carries an orientation tag, so it never changes a pixel).
    """
    with Image.open(path) as source:
        digest = pixel_sha256(source.convert("RGB"))
        rgb = ImageOps.exif_transpose(source).convert("RGB")
    return processor(images=rgb, return_tensors="pt")["pixel_values"][0], digest


def make_clip_image_loader(processor):
    """Picklable loader for extract_split / DataLoader workers."""
    return partial(load_clip_image, processor=processor)


class FrozenCLIPVisionGrid(nn.Module):
    """CLIP vision tower, frozen: preprocessed [B, 3, 224, 224] -> [B, 49, 768] patch tokens of the last layer.

    The class token is dropped; the remaining 49 tokens are the 7 x 7 grid in row-major order. Tokens are the encoder's
    last_hidden_state (before post_layernorm and the visual projection); RegionProjection normalises them afterwards.
    """

    def __init__(self, model_name=CLIP_MODEL, revision=CLIP_REVISION, vision_model=None):
        super().__init__()
        self.model = vision_model or CLIPVisionModel.from_pretrained(model_name, revision=revision)
        config = self.model.config
        self.feature_dim = config.hidden_size
        if (config.image_size // config.patch_size) ** 2 != NUM_REGIONS or config.image_size // config.patch_size != GRID:
            raise ValueError(f"{config.image_size}/{config.patch_size} does not give a {GRID}x{GRID} grid")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # frozen encoder: always inference mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values):
        # cuDNN's TF32 patch convolution makes GPU features depend on the batch size (0.02 off on values up to 12);
        # full fp32 keeps them within 1e-4 of the CPU result whatever the batch.
        with torch.backends.cudnn.flags(allow_tf32=False):
            tokens = self.model(pixel_values=pixel_values).last_hidden_state  # [B, 1 + 49, 768], index 0 = class token
        return tokens[:, 1:].contiguous()
