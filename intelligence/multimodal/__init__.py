"""Multimodal — vision and audio processing for multimodal LLMs.

Provides from-scratch implementations of:

    - ``Image``: Image data structure with HWC (Height × Width × Channels) layout
    - ``ImageEncoder``: Patch embedding + transformer for vision (ViT-style)
    - ``Audio``: Audio waveform representation and spectrogram extraction
    - ``AudioEncoder``: 1-D CNN feature extractor for audio
    - ``MultimodalEncoder``: Fuses visual and text features via cross-attention

All implementations are numpy-based and do not require PIL, OpenCV,
librosa, or torch at import time.

Example::

    img = Image.from_array(np.random.randn(224, 224, 3))
    encoder = ImageEncoder(patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072)
    features = encoder.forward(img)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from intelligence.transformers.attention import MultiHeadAttention

# ---------------------------------------------------------------------------
# Helper functions (numpy-only, no torch needed)
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + x**3 / 3.0)))


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

@dataclass
class Image:
    """Image data structure with HWC (Height × Width × Channels) layout.

    Stores pixel data as a numpy array and provides common preprocessing
    operations.

    Attributes:
        data:       numpy array of shape (H, W, C) or (H, W).
        mode:       Image mode ("rgb", "grayscale", "rgba").
        width:      Image width in pixels.
        height:     Image height in pixels.
        channels:   Number of channels.
    """

    data: np.ndarray
    mode: str = "rgb"

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim == 2:
            self.data = self.data[:, :, np.newaxis]
            self.mode = "grayscale"
        self.height, self.width, self.channels = self.data.shape

    @classmethod
    def from_array(cls, data: np.ndarray, mode: str = "rgb") -> Image:
        """Create an Image from a numpy array."""
        return cls(data=data, mode=mode)

    @classmethod
    def random(cls, width: int, height: int, channels: int = 3) -> Image:
        """Create a random noise image."""
        data = np.random.rand(height, width, channels)
        return cls(data=data, mode="rgb" if channels == 3 else "grayscale")

    def to_grayscale(self) -> Image:
        """Convert to grayscale using luminance weighting."""
        if self.channels == 1:
            return self
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(self.data[..., :3], weights)
        return Image(data=gray, mode="grayscale")

    def normalize(
        self,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
    ) -> Image:
        """Normalize pixel values using ImageNet-style statistics."""
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        normalized = self.data.astype(np.float64).copy()
        for c in range(self.channels):
            m = mean[c] if c < len(mean) else 0.0
            s = std[c] if c < len(std) else 1.0
            normalized[..., c] = (normalized[..., c] - m) / s
        return Image(data=normalized, mode=self.mode)

    def resize(self, new_width: int, new_height: int) -> Image:
        """Bilinear interpolation resize."""
        h, w = self.height, self.width
        y_coords = np.linspace(0, h - 1, new_height)
        x_coords = np.linspace(0, w - 1, new_width)

        result = np.zeros((new_height, new_width, self.channels))
        for c in range(self.channels):
            channel = self.data[..., c]
            for i, y in enumerate(y_coords):
                y0, y1 = int(np.floor(y)), min(int(np.ceil(y)), h - 1)
                for j, x in enumerate(x_coords):
                    x0, x1 = int(np.floor(x)), min(int(np.ceil(x)), w - 1)
                    if y0 == y1 and x0 == x1:
                        result[i, j, c] = channel[y0, x0]
                    else:
                        wx = x - x0 if x1 != x0 else 0
                        wy = y - y0 if y1 != y0 else 0
                        result[i, j, c] = (
                            channel[y0, x0] * (1 - wx) * (1 - wy)
                            + channel[y0, x1] * wx * (1 - wy)
                            + channel[y1, x0] * (1 - wx) * wy
                            + channel[y1, x1] * wx * wy
                        )
        return Image(data=result, mode=self.mode)

    def to_patches(self, patch_size: int) -> np.ndarray:
        """Extract non-overlapping patches.

        Returns:
            Array of shape (num_patches, patch_size * patch_size * channels).
        """
        h, w, c = self.data.shape
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = self.data[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches.append(patch.flatten())
        return np.array(patches)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, mode={self.mode}, channels={self.channels})"


# ---------------------------------------------------------------------------
# Vision Transformer Encoder
# ---------------------------------------------------------------------------

class ImageEncoder:
    """Vision Transformer (ViT) encoder — patch embedding + transformer.

    Implements the patch embedding and a simplified transformer encoder
    for processing images as sequences of patches.

    Args:
        patch_size:    Size of each image patch (e.g., 16 for 16x16 patches).
        dim:           Embedding dimension (model dimension).
        depth:         Number of transformer encoder layers.
        heads:         Number of attention heads.
        mlp_dim:       FFN hidden dimension.
        dropout:       Dropout rate.
        image_size:    Expected input image size (for position embedding).

    Attributes:
        patch_dim:   Flattened patch dimension (patch_size^2 * channels).
        num_patches: Number of patches per image.
    """

    def __init__(
        self,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        image_size: int = 224,
    ) -> None:
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.image_size = image_size
        self.patch_dim = patch_size * patch_size * 3  # assumes RGB
        self.num_patches = (image_size // patch_size) ** 2

        rng = np.random.default_rng(42)

        # Patch embedding projection
        limit = np.sqrt(6.0 / (self.patch_dim + dim))
        self.patch_embed = rng.uniform(-limit, limit, (self.patch_dim, dim))

        # CLS token
        self.cls_token = rng.standard_normal(dim) * 0.02

        # Positional embeddings (learnable)
        self.pos_embed = rng.standard_normal((self.num_patches + 1, dim)) * 0.02

        # Transformer encoder layers
        self.layers: list[dict[str, Any]] = []
        for i in range(depth):
            self.layers.append({
                "attention": MultiHeadAttention(
                    d_model=dim, num_heads=heads,
                    dropout_rate=dropout, seed=42 + i,
                ),
                "ffn_w1": rng.standard_normal((dim, mlp_dim)) * 0.02,
                "ffn_b1": np.zeros(mlp_dim),
                "ffn_w2": rng.standard_normal((mlp_dim, dim)) * 0.02,
                "ffn_b2": np.zeros(dim),
            })

    def patch_embed_forward(self, image: Image) -> np.ndarray:
        """Embed image patches into the transformer's token space.

        Args:
            image: Input Image object.

        Returns:
            Tensor of shape (num_patches + 1, dim) — includes CLS token + position.
        """
        patches = image.to_patches(self.patch_size)  # (num_patches, patch_dim)
        embedded = np.matmul(patches, self.patch_embed)  # (num_patches, dim)
        cls = self.cls_token[np.newaxis, :]  # (1, dim)
        embedded = np.concatenate([cls, embedded], axis=0)  # (num_patches+1, dim)
        embedded = embedded + self.pos_embed
        return embedded

    def forward(self, image: Image) -> np.ndarray:
        """Full forward pass: patch embedding -> transformer layers.

        Args:
            image: Input Image object.

        Returns:
            Encoded features of shape (num_patches + 1, dim).
        """
        x = self.patch_embed_forward(image)

        for layer in self.layers:
            attn = layer["attention"]
            attn.training = False
            # Self-attention sublayer
            attn_out, _ = attn(x, x, x)
            x = _layer_norm(x + attn_out)
            # FFN sublayer
            ffn_out = _gelu(np.matmul(x, layer["ffn_w1"]) + layer["ffn_b1"])
            ffn_out = np.matmul(ffn_out, layer["ffn_w2"]) + layer["ffn_b2"]
            x = _layer_norm(x + ffn_out)

        return x

    def get_cls_token(self, image: Image) -> np.ndarray:
        """Extract the CLS token embedding (global image representation)."""
        features = self.forward(image)
        return features[0]  # CLS token


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

@dataclass
class Audio:
    """Audio waveform representation.

    Args:
        samples:     1-D numpy array of audio samples (float32, -1 to 1).
        sample_rate: Sample rate in Hz (default: 16000).

    Attributes:
        samples:      Raw waveform data.
        sample_rate.  Sample rate in Hz.
        duration:     Duration in seconds.
        num_samples.  Total number of samples.
    """

    samples: np.ndarray
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float32)

    @classmethod
    def from_list(cls, samples: list[float], sample_rate: int = 16000) -> Audio:
        """Create an Audio object from a list of samples."""
        return cls(samples=np.array(samples, dtype=np.float32), sample_rate=sample_rate)

    @classmethod
    def random(cls, duration: float, sample_rate: int = 16000) -> Audio:
        """Generate random noise audio."""
        n = int(duration * sample_rate)
        samples = np.random.uniform(-1, 1, n).astype(np.float32)
        return cls(samples=samples, sample_rate=sample_rate)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def resample(self, new_sample_rate: int) -> Audio:
        """Simple nearest-neighbor resampling."""
        if new_sample_rate == self.sample_rate:
            return self
        ratio = new_sample_rate / self.sample_rate
        new_length = int(len(self.samples) * ratio)
        indices = (np.arange(new_length) / ratio).astype(int)
        indices = np.clip(indices, 0, len(self.samples) - 1)
        return Audio(samples=self.samples[indices], sample_rate=new_sample_rate)

    def spectrogram(
        self,
        window_size: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
    ) -> np.ndarray:
        """Compute a magnitude spectrogram using STFT.

        Args:
            window_size: STFT window size (samples).
            hop_length:  Hop length between frames.
            n_fft:       FFT size (zero-padded if > window_size).

        Returns:
            Spectrogram of shape (n_frames, n_fft // 2 + 1).
        """
        window = np.hanning(window_size)
        frames = []
        for i in range(0, len(self.samples) - window_size + 1, hop_length):
            frame = self.samples[i : i + window_size] * window
            if n_fft > window_size:
                frame = np.pad(frame, (0, n_fft - window_size))
            spectrum = np.abs(np.fft.rfft(frame))
            frames.append(spectrum)

        return np.array(frames) if frames else np.zeros((1, n_fft // 2 + 1))

    def mel_spectrogram(
        self,
        n_mels: int = 80,
        window_size: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """Compute a mel-spectrogram.

        Args:
            n_mels:      Number of mel bands.
            sample_rate: Audio sample rate (defaults to self.sample_rate).

        Returns:
            Mel-spectrogram of shape (n_frames, n_mels).
        """
        sr = sample_rate or self.sample_rate
        spec = self.spectrogram(window_size, hop_length, n_fft)

        n_freq = n_fft // 2 + 1

        # Convert Hz to mel
        def hz_to_mel(hz: float) -> float:
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel: float) -> float:
            return 700 * (10 ** (mel / 2595) - 1)

        mel_low = hz_to_mel(0)
        mel_high = hz_to_mel(sr / 2)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = [mel_to_hz(m) for m in mel_points]
        bin_points = [int(hz * (n_fft + 1) / sr) for hz in hz_points]

        # Triangular filters
        filters = np.zeros((n_mels, n_freq))
        for m in range(n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            for i in range(left, center):
                if i < n_freq:
                    filters[m, i] = (i - left) / (center - left)
            for i in range(center, right):
                if i < n_freq:
                    filters[m, i] = (right - i) / (right - center)

        return np.matmul(spec, filters.T)


# ---------------------------------------------------------------------------
# Audio CNN Encoder
# ---------------------------------------------------------------------------

class AudioEncoder:
    """1-D CNN feature extractor for audio.

    Similar to wav2vec 2.0's feature encoder — applies stacked
    1-D convolutions with stride to extract temporal features from
    raw audio waveforms.

    Args:
        input_dim:    Input dimension (1 for mono audio).
        hidden_dim:   Hidden dimension per layer.
        num_layers:   Number of CNN layers.
        kernel_size:  Convolution kernel size.
        stride:       Stride (controls downsampling).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 7,
        kernel_size: int = 1024,
        stride: int = 3,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride

        rng = np.random.default_rng(42)

        # CNN layers: each is a 1-D convolution
        self.layers: list[dict[str, np.ndarray]] = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = hidden_dim
            w = rng.standard_normal((d_out, d_in, kernel_size)) * 0.02
            b = np.zeros(d_out)
            self.layers.append({"weight": w, "bias": b})

    def _conv1d(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        stride: int,
        padding: int = 0,
    ) -> np.ndarray:
        """1-D convolution (simplified, no dilation)."""
        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding)), mode="constant")

        in_channels, seq_len = x.shape
        out_channels = weight.shape[0]
        kernel = weight.shape[2]

        output_len = (seq_len - kernel) // stride + 1
        output = np.zeros((out_channels, output_len))

        for i in range(output_len):
            start = i * stride
            for o in range(out_channels):
                for c in range(in_channels):
                    output[o, i] += np.sum(
                        x[c, start : start + kernel] * weight[o, c]
                    )
                output[o, i] += bias[o]

        return output

    def forward(self, audio: Audio) -> np.ndarray:
        """Extract features from an audio waveform.

        Args:
            audio: Audio object.

        Returns:
            Feature array of shape (seq_len, hidden_dim).
        """
        x = audio.samples[np.newaxis, :]  # (1, seq_len)

        for layer in self.layers:
            x = self._conv1d(x, layer["weight"], layer["bias"], self.stride)
            # ReLU activation
            x = np.clip(x, 0, None)

        # Transpose to (seq_len, hidden_dim)
        return x.T


# ---------------------------------------------------------------------------
# Multimodal Encoder — fuses vision + text
# ---------------------------------------------------------------------------

class MultimodalEncoder:
    """Multimodal encoder that fuses visual and text features.

    Uses cross-attention to let text features attend to visual features
    (or vice versa). Based on the architecture of CLIP/Llava.

    Args:
        vision_dim:  Dimension of vision features (from ImageEncoder).
        text_dim:    Dimension of text features.
        fuse_dim:    Output dimension after fusion.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        fuse_dim: int = 768,
    ) -> None:
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fuse_dim = fuse_dim

        rng = np.random.default_rng(42)

        # Projection layers to fuse_dim
        limit = np.sqrt(6.0 / (vision_dim + fuse_dim))
        self.vision_proj = rng.uniform(-limit, limit, (vision_dim, fuse_dim))
        limit = np.sqrt(6.0 / (text_dim + fuse_dim))
        self.text_proj = rng.uniform(-limit, limit, (text_dim, fuse_dim))

        # Cross-attention: text attends to vision
        self.cross_attn = MultiHeadAttention(
            d_model=fuse_dim, num_heads=8, seed=42,
        )

    def forward(
        self,
        vision_features: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """Fuse vision and text features into a joint representation.

        Args:
            vision_features: (num_patches, vision_dim) from ImageEncoder.
            text_features:   (seq_len, text_dim) from text encoder.

        Returns:
            Fused representation of shape (seq_len, fuse_dim).
        """
        # Project to common dimension
        v_proj = np.matmul(vision_features, self.vision_proj)
        t_proj = np.matmul(text_features, self.text_proj)

        # Cross-attention: text queries attend to vision keys/values
        attended, _ = self.cross_attn(t_proj, v_proj, v_proj)

        # Residual connection
        return _layer_norm(t_proj + attended)
