"""Native MLX BigVGAN-v2 vocoder and bandwidth-extension runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from lara_ltx.errors import LaraError
from lara_ltx.models.loading import iter_component_weight_batches
from lara_ltx.models.vocoder import (
    ACTIVATION_FILTER_SIZE,
    BWE_CHANNELS,
    BWE_UPSAMPLE_KERNELS,
    BWE_UPSAMPLE_RATES,
    CONV_KERNEL_SIZE,
    MEL_BINS_PER_CHANNEL,
    PRIMARY_UPSAMPLE_KERNELS,
    PRIMARY_UPSAMPLE_RATES,
    PRIMARY_VOCODER_CHANNELS,
    RESIDUAL_DILATIONS,
    RESIDUAL_KERNEL_SIZES,
    STEREO_CHANNEL_COUNT,
    STFT_FILTER_LENGTH,
    STFT_FREQUENCY_BINS,
    VOCODER_INPUT_CHANNELS,
    WAVEFORM_OUTPUT_CHANNELS,
    validate_vocoder_mapping,
    vocoder_target_shapes,
)

CHANNEL_FIRST_TO_LAST = (0, 2, 1)
CHANNEL_LAST_TO_FIRST = (0, 2, 1)
PYTORCH_CONV1D_TO_MLX = (0, 2, 1)
PYTORCH_CONV_TRANSPOSE1D_TO_MLX = (1, 2, 0)
SNAKE_EPSILON = 1e-9
MEL_LOG_FLOOR = 1e-5
DEFAULT_INPUT_SAMPLING_RATE = 16_000
DEFAULT_OUTPUT_SAMPLING_RATE = 48_000
DEFAULT_BWE_HOP_LENGTH = 80
HANN_RESAMPLE_ROLLOFF = 0.99
HANN_RESAMPLE_LOWPASS_WIDTH = 6


class Conv1d(nn.Module):
    """PyTorch-layout 1D convolution evaluated with FP32 accumulation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, value: mx.array, *, padding: int = 0, dilation: int = 1, stride: int = 1) -> mx.array:
        channel_last = mx.transpose(value.astype(mx.float32), CHANNEL_FIRST_TO_LAST)
        weight = mx.transpose(self.weight.astype(mx.float32), PYTORCH_CONV1D_TO_MLX)
        output = mx.conv1d(channel_last, weight, stride=stride, padding=padding, dilation=dilation)
        if self.bias is not None:
            output = output + self.bias.astype(mx.float32)[None, None, :]
        return mx.transpose(output, CHANNEL_LAST_TO_FIRST)


class ConvTranspose1d(nn.Module):
    """PyTorch-layout 1D transposed convolution evaluated in FP32."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((in_channels, out_channels, kernel_size))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, value: mx.array, *, stride: int, padding: int) -> mx.array:
        channel_last = mx.transpose(value.astype(mx.float32), CHANNEL_FIRST_TO_LAST)
        weight = mx.transpose(self.weight.astype(mx.float32), PYTORCH_CONV_TRANSPOSE1D_TO_MLX)
        output = mx.conv_transpose1d(channel_last, weight, stride=stride, padding=padding)
        output = output + self.bias.astype(mx.float32)[None, None, :]
        return mx.transpose(output, CHANNEL_LAST_TO_FIRST)


class FilterWeights(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.filter = mx.zeros((1, 1, kernel_size))


def _depthwise_conv1d(value: mx.array, weight: mx.array, *, stride: int = 1) -> mx.array:
    channels = value.shape[1]
    channel_last = mx.transpose(value.astype(mx.float32), CHANNEL_FIRST_TO_LAST)
    kernel = mx.broadcast_to(weight.astype(mx.float32), (channels, 1, weight.shape[-1]))
    kernel = mx.transpose(kernel, PYTORCH_CONV1D_TO_MLX)
    return mx.transpose(mx.conv1d(channel_last, kernel, stride=stride, groups=channels), CHANNEL_LAST_TO_FIRST)


def _depthwise_conv_transpose1d(value: mx.array, weight: mx.array, *, stride: int) -> mx.array:
    channels = value.shape[1]
    channel_last = mx.transpose(value.astype(mx.float32), CHANNEL_FIRST_TO_LAST)
    source = mx.broadcast_to(weight.astype(mx.float32), (channels, 1, weight.shape[-1]))
    kernel = mx.transpose(source, PYTORCH_CONV1D_TO_MLX)
    return mx.transpose(
        mx.conv_transpose1d(channel_last, kernel, stride=stride, groups=channels),
        CHANNEL_LAST_TO_FIRST,
    )


class UpSample1d(nn.Module):
    def __init__(self, ratio: int, *, kernel_size: int = ACTIVATION_FILTER_SIZE) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, value: mx.array) -> mx.array:
        value = mx.pad(value, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")
        value = self.ratio * _depthwise_conv_transpose1d(value, self.filter, stride=self.ratio)
        return value[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, ratio: int, *, kernel_size: int = ACTIVATION_FILTER_SIZE) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, value: mx.array) -> mx.array:
        value = mx.pad(value, ((0, 0), (0, 0), (self.pad_left, self.pad_right)), mode="edge")
        return _depthwise_conv1d(value, self.filter, stride=self.ratio)


class DownSample1d(nn.Module):
    def __init__(self, ratio: int, *, kernel_size: int = ACTIVATION_FILTER_SIZE) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, kernel_size=kernel_size)

    def __call__(self, value: mx.array) -> mx.array:
        return self.lowpass(value)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, value: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha.astype(mx.float32))[None, :, None]
        beta = mx.exp(self.beta.astype(mx.float32))[None, :, None]
        value = value.astype(mx.float32)
        return value + mx.square(mx.sin(value * alpha)) / (beta + SNAKE_EPSILON)


class Activation1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(2)
        self.downsample = DownSample1d(2)

    def __call__(self, value: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(value)))


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.dilations = dilations
        self.convs1 = [Conv1d(channels, channels, kernel_size) for _ in dilations]
        self.convs2 = [Conv1d(channels, channels, kernel_size) for _ in dilations]
        self.acts1 = [Activation1d(channels) for _ in dilations]
        self.acts2 = [Activation1d(channels) for _ in dilations]
        self.kernel_size = kernel_size

    def __call__(self, value: mx.array) -> mx.array:
        for index, dilation in enumerate(self.dilations):
            hidden = self.acts1[index](value)
            hidden = self.convs1[index](
                hidden,
                padding=_same_padding(self.kernel_size, dilation),
                dilation=dilation,
            )
            hidden = self.acts2[index](hidden)
            hidden = self.convs2[index](hidden, padding=_same_padding(self.kernel_size))
            value = value + hidden
        return value


@dataclass(frozen=True)
class VocoderConfig:
    initial_channels: int
    upsample_rates: tuple[int, ...]
    upsample_kernels: tuple[int, ...]
    apply_final_activation: bool = True

    def __post_init__(self) -> None:
        if (
            self.initial_channels <= 0
            or not self.upsample_rates
            or len(self.upsample_rates) != len(self.upsample_kernels)
            or any(value <= 0 for value in (*self.upsample_rates, *self.upsample_kernels))
        ):
            raise LaraError("LARA-TENSOR-024", details={"shape": "invalid_vocoder_configuration"})


PRIMARY_VOCODER_CONFIG = VocoderConfig(
    initial_channels=PRIMARY_VOCODER_CHANNELS,
    upsample_rates=PRIMARY_UPSAMPLE_RATES,
    upsample_kernels=PRIMARY_UPSAMPLE_KERNELS,
)
BWE_VOCODER_CONFIG = VocoderConfig(
    initial_channels=BWE_CHANNELS,
    upsample_rates=BWE_UPSAMPLE_RATES,
    upsample_kernels=BWE_UPSAMPLE_KERNELS,
    apply_final_activation=False,
)


class Vocoder(nn.Module):
    """Checkpoint-compatible stereo BigVGAN-v2 generator."""

    def __init__(self, config: VocoderConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_pre = Conv1d(VOCODER_INPUT_CHANNELS, config.initial_channels, CONV_KERNEL_SIZE)
        self.ups = []
        self.resblocks = []
        for stage, kernel_size in enumerate(config.upsample_kernels):
            input_channels = config.initial_channels // (2**stage)
            output_channels = config.initial_channels // (2 ** (stage + 1))
            self.ups.append(ConvTranspose1d(input_channels, output_channels, kernel_size))
            for residual_kernel in RESIDUAL_KERNEL_SIZES:
                self.resblocks.append(AMPBlock1(output_channels, residual_kernel, RESIDUAL_DILATIONS))
        final_channels = config.initial_channels // (2 ** len(config.upsample_rates))
        self.act_post = Activation1d(final_channels)
        self.conv_post = Conv1d(final_channels, WAVEFORM_OUTPUT_CHANNELS, CONV_KERNEL_SIZE, bias=False)

    def __call__(self, mel_spectrogram: mx.array) -> mx.array:
        if (
            mel_spectrogram.ndim != 4
            or mel_spectrogram.shape[1] != STEREO_CHANNEL_COUNT
            or mel_spectrogram.shape[3] != MEL_BINS_PER_CHANNEL
        ):
            raise LaraError("LARA-TENSOR-025", details={"shape": tuple(mel_spectrogram.shape)})
        batch, channels, frames, mel_bins = mel_spectrogram.shape
        hidden = mx.transpose(mel_spectrogram, (0, 1, 3, 2)).reshape(batch, channels * mel_bins, frames)
        hidden = self.conv_pre(hidden, padding=_same_padding(CONV_KERNEL_SIZE))
        for stage, (rate, kernel_size) in enumerate(
            zip(self.config.upsample_rates, self.config.upsample_kernels, strict=True)
        ):
            hidden = self.ups[stage](hidden, stride=rate, padding=(kernel_size - rate) // 2)
            start = stage * len(RESIDUAL_KERNEL_SIZES)
            outputs = [self.resblocks[index](hidden) for index in range(start, start + len(RESIDUAL_KERNEL_SIZES))]
            hidden = mx.mean(mx.stack(outputs), axis=0)
        hidden = self.conv_post(self.act_post(hidden), padding=_same_padding(CONV_KERNEL_SIZE))
        if self.config.apply_final_activation:
            hidden = mx.clip(hidden, -1.0, 1.0)
        return hidden


def _generator_rules(
    mapping: dict[str, object], prefix: str
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[int, ...]]]:
    rules = validate_vocoder_mapping(mapping)
    shapes = vocoder_target_shapes()
    source_prefix = f"{prefix}."
    selected = tuple(
        {**rule, "target_key": str(rule["target_key"]).removeprefix(source_prefix)}
        for rule in rules
        if str(rule["target_key"]).startswith(source_prefix)
    )
    target_shapes = {
        key.removeprefix(source_prefix): shape for key, shape in shapes.items() if key.startswith(source_prefix)
    }
    return selected, target_shapes


def load_primary_vocoder(*, checkpoint: Path, mapping: dict[str, object]) -> Vocoder:
    """Strict-load the primary 16 kHz generator from the reviewed full mapping."""

    rules, target_shapes = _generator_rules(mapping, "vocoder")
    model = Vocoder(PRIMARY_VOCODER_CONFIG)
    for batch in iter_component_weight_batches((checkpoint,), rules, expected_target_shapes=target_shapes):
        model.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    return model


def load_bwe_vocoder(*, checkpoint: Path, mapping: dict[str, object]) -> Vocoder:
    """Strict-load the 48 kHz bandwidth-extension residual generator."""

    rules, target_shapes = _generator_rules(mapping, "bwe_generator")
    model = Vocoder(BWE_VOCODER_CONFIG)
    for batch in iter_component_weight_batches((checkpoint,), rules, expected_target_shapes=target_shapes):
        model.load_weights(batch)
        mx.eval(*[value for _, value in batch])
    return model


class STFTWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = STFT_FREQUENCY_BINS * 2
        self.forward_basis = mx.zeros((channels, 1, STFT_FILTER_LENGTH))
        self.inverse_basis = mx.zeros((channels, 1, STFT_FILTER_LENGTH))


class MelSTFT(nn.Module):
    """Checkpoint-backed causal log-mel transform used by BWE."""

    def __init__(self, *, hop_length: int = DEFAULT_BWE_HOP_LENGTH, win_length: int = STFT_FILTER_LENGTH) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft_fn = STFTWeights()
        self.mel_basis = mx.zeros((MEL_BINS_PER_CHANNEL, STFT_FREQUENCY_BINS))

    def __call__(self, waveform: mx.array) -> mx.array:
        if waveform.ndim != 3 or waveform.shape[1] != STEREO_CHANNEL_COUNT:
            raise LaraError("LARA-TENSOR-026", details={"shape": tuple(waveform.shape)})
        batch, channels, samples = waveform.shape
        flattened = waveform.astype(mx.float32).reshape(batch * channels, 1, samples)
        left_padding = max(0, self.win_length - self.hop_length)
        flattened = mx.pad(flattened, ((0, 0), (0, 0), (left_padding, 0)))
        basis = mx.transpose(self.stft_fn.forward_basis.astype(mx.float32), PYTORCH_CONV1D_TO_MLX)
        spectrum = mx.conv1d(mx.transpose(flattened, CHANNEL_FIRST_TO_LAST), basis, stride=self.hop_length)
        spectrum = mx.transpose(spectrum, CHANNEL_LAST_TO_FIRST)
        real = spectrum[:, :STFT_FREQUENCY_BINS]
        imaginary = spectrum[:, STFT_FREQUENCY_BINS:]
        magnitude = mx.sqrt(mx.square(real) + mx.square(imaginary))
        mel = mx.einsum("mf,bft->bmt", self.mel_basis.astype(mx.float32), magnitude)
        log_mel = mx.log(mx.maximum(mel, MEL_LOG_FLOOR))
        return log_mel.reshape(batch, channels, MEL_BINS_PER_CHANNEL, log_mel.shape[-1])


def _hann_resample_filter(ratio: int) -> mx.array:
    width = math.ceil(HANN_RESAMPLE_LOWPASS_WIDTH / HANN_RESAMPLE_ROLLOFF)
    kernel_size = 2 * width * ratio + 1
    time_axis = (mx.arange(kernel_size, dtype=mx.float32) / ratio - width) * HANN_RESAMPLE_ROLLOFF
    clamped = mx.clip(time_axis, -HANN_RESAMPLE_LOWPASS_WIDTH, HANN_RESAMPLE_LOWPASS_WIDTH)
    window = mx.square(mx.cos(clamped * math.pi / HANN_RESAMPLE_LOWPASS_WIDTH / 2))
    pi_time = math.pi * time_axis
    sinc = mx.where(time_axis == 0, 1.0, mx.sin(pi_time) / pi_time)
    # The CUDA reference constructs the module in BF16 before its FP32 forward.
    return (sinc * window * HANN_RESAMPLE_ROLLOFF / ratio).reshape(1, 1, -1).astype(mx.bfloat16)


class HannUpSample1d(nn.Module):
    def __init__(self, ratio: int) -> None:
        super().__init__()
        width = math.ceil(HANN_RESAMPLE_LOWPASS_WIDTH / HANN_RESAMPLE_ROLLOFF)
        self.ratio = ratio
        self.pad = width
        self.kernel_size = 2 * width * ratio + 1
        self.pad_left = 2 * width * ratio
        self.pad_right = self.kernel_size - ratio
        self.filter = _hann_resample_filter(ratio)

    def __call__(self, waveform: mx.array) -> mx.array:
        waveform = mx.pad(waveform, ((0, 0), (0, 0), (self.pad, self.pad)), mode="edge")
        waveform = self.ratio * _depthwise_conv_transpose1d(waveform, self.filter, stride=self.ratio)
        return waveform[..., self.pad_left : -self.pad_right]


class VocoderWithBWE(nn.Module):
    """Two-stage 16 kHz vocoder plus causal 48 kHz bandwidth extension."""

    def __init__(
        self,
        *,
        input_sampling_rate: int = DEFAULT_INPUT_SAMPLING_RATE,
        output_sampling_rate: int = DEFAULT_OUTPUT_SAMPLING_RATE,
        hop_length: int = DEFAULT_BWE_HOP_LENGTH,
    ) -> None:
        super().__init__()
        if output_sampling_rate % input_sampling_rate:
            raise LaraError("LARA-TENSOR-024", details={"shape": "non_integral_audio_resample_ratio"})
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.vocoder = Vocoder(PRIMARY_VOCODER_CONFIG)
        self.bwe_generator = Vocoder(BWE_VOCODER_CONFIG)
        self.mel_stft = MelSTFT(hop_length=hop_length)
        self.resampler = HannUpSample1d(output_sampling_rate // input_sampling_rate)

    def __call__(self, mel_spectrogram: mx.array) -> mx.array:
        input_dtype = mel_spectrogram.dtype
        waveform = self.vocoder(mel_spectrogram.astype(mx.float32))
        low_rate_length = waveform.shape[-1]
        output_length = low_rate_length * self.output_sampling_rate // self.input_sampling_rate
        remainder = low_rate_length % self.hop_length
        if remainder:
            waveform = mx.pad(waveform, ((0, 0), (0, 0), (0, self.hop_length - remainder)))
        mel = self.mel_stft(waveform)
        residual = self.bwe_generator(mx.transpose(mel, (0, 1, 3, 2)))
        skip = self.resampler(waveform)
        if residual.shape != skip.shape:
            raise LaraError(
                "LARA-TENSOR-027",
                details={"residual_shape": tuple(residual.shape), "skip_shape": tuple(skip.shape)},
            )
        return mx.clip(residual + skip, -1.0, 1.0)[..., :output_length].astype(input_dtype)


def load_vocoder_with_bwe(*, checkpoint: Path, mapping: dict[str, object]) -> VocoderWithBWE:
    """Strict-load the full two-stage waveform decoder and exact STFT buffers."""

    rules = validate_vocoder_mapping(mapping)
    model = VocoderWithBWE()
    for batch in iter_component_weight_batches(
        (checkpoint,),
        rules,
        expected_target_shapes=vocoder_target_shapes(),
    ):
        # The deterministic Hann resampler is derived from the pinned sample-rate
        # configuration and intentionally is not persisted in the checkpoint.
        model.load_weights(batch, strict=False)
        mx.eval(*[value for _, value in batch])
    return model
