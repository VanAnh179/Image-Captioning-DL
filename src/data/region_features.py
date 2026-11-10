"""Frozen ResNet-50 7x7 region features, cached with image IDs so both decoders read the same data."""

import hashlib
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
GRID = 7
NUM_REGIONS = GRID * GRID  # region index = row * 7 + column
FEATURE_DIM = 2048
WEIGHTS_NAME = "IMAGENET1K_V1"
FORMAT_VERSION = 1


def feature_path(directory, split):
    return Path(directory) / f"resnet50_7x7_{split}.pt"


def pixel_sha256(rgb):
    """Same definition as scripts/prepare_ktvic_dataset.py, so it can be compared with the manifest."""
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def load_image(path, size=IMAGE_SIZE):
    """Return (float32 [3, size, size] in [0, 1], pixel hash of the untransposed RGB image).

    Same steps as make_caption_batch: EXIF transpose, RGB, bilinear resize.
    """
    with Image.open(path) as source:
        digest = pixel_sha256(source.convert("RGB"))
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return pixels.view(size, size, 3).permute(2, 0, 1).float().div_(255), digest


class FrozenResNet50Grid(nn.Module):
    """ResNet-50 up to layer4, frozen: [B, 3, 224, 224] in [0, 1] -> [B, 49, 2048]."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights[WEIGHTS_NAME] if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # BatchNorm statistics must never update
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        grid = self.body((images - self.mean) / self.std)  # [B, 2048, 7, 7]
        return grid.flatten(2).transpose(1, 2).contiguous()  # [B, 49, 2048]


class _ImageRows(Dataset):
    def __init__(self, rows, image_root):
        self.rows, self.image_root = rows, Path(image_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_image(self.image_root / self.rows[index]["image_path"])


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract_split(rows, image_root, encoder, device, batch_size=32, workers=0, dtype=torch.float16):
    """Encode rows in manifest order. Raises if any image's pixel hash differs from the manifest."""
    loader = DataLoader(_ImageRows(rows, image_root), batch_size=batch_size, num_workers=workers)
    chunks, position = [], 0
    for images, digests in loader:
        for offset, digest in enumerate(digests):
            row = rows[position + offset]
            if digest != row["pixel_sha256"]:
                raise ValueError(
                    f"Ảnh ID {row['image_id']} ({row['image_path']}) có hash pixel khác manifest; "
                    "file ảnh đã bị thay đổi, sai thư mục ảnh, hoặc Pillow không phải 12.2.0."
                )
        chunks.append(encoder(images.to(device)).to(dtype).cpu())
        position += len(digests)
    return torch.cat(chunks)


class RegionFeatureStore:
    """Read-only lookup of cached region features by image ID. Used identically by both decoders."""

    def __init__(self, payload):
        self.image_ids = payload["image_ids"]
        self.features = payload["features"]
        self.image_paths = payload["image_paths"]
        self.pixel_sha256 = payload["pixel_sha256"]
        self.meta = payload["meta"]
        self.row_of = {int(i): n for n, i in enumerate(self.image_ids.tolist())}
        if len(self.row_of) != len(self.image_ids) or len(self.features) != len(self.image_ids):
            raise ValueError("File đặc trưng có ID trùng hoặc số tensor khác số ID")

    @classmethod
    def load(cls, directory, split):
        path = feature_path(directory, split)
        if not path.is_file():
            raise FileNotFoundError(
                f"Chưa có đặc trưng {split}: {path}. Chạy scripts/extract_region_features.py trước.")
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def __len__(self):
        return len(self.row_of)

    def __contains__(self, image_id):
        return int(image_id) in self.row_of

    def get_batch(self, image_ids, dtype=torch.float32):
        """[len(image_ids), 49, 2048] in the order requested."""
        ids = [int(i) for i in (image_ids.tolist() if torch.is_tensor(image_ids) else image_ids)]
        missing = [i for i in ids if i not in self.row_of]
        if missing:
            raise KeyError(f"Không có đặc trưng cho ảnh ID {missing[:10]} (tổng {len(missing)}) trong tập này")
        return self.features[[self.row_of[i] for i in ids]].to(dtype)

    def trace(self, image_id):
        """Where a cached tensor came from: relative image path and pixel hash."""
        row = self.row_of[int(image_id)]
        return {"image_id": int(image_id), "image_path": self.image_paths[row],
                "pixel_sha256": self.pixel_sha256[row]}

    def nearest_id(self, tensor):
        """Reverse lookup: (ID, squared L2 distance) of the cached tensor closest to a [49, 2048] tensor."""
        query = tensor.reshape(1, -1).float()
        best_distance, best_row = float("inf"), -1
        for start in range(0, len(self.features), 256):
            block = self.features[start:start + 256].reshape(-1, query.shape[1]).float()
            value, index = (block - query).pow(2).sum(1).min(0)
            if value.item() < best_distance:
                best_distance, best_row = value.item(), start + index.item()
        return int(self.image_ids[best_row]), best_distance
