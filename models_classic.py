"""Independently implemented time-series classification backbones.

No imports from the previous project or pretrained time-series weights.  These
are explicitly documented classification adaptations, not bit-for-bit copies
of the forecasting repositories.  All tensors are [batch, quarter, channel].
The caller supplies train-fold-standardized features, missingness/age channels,
and optionally a boolean valid_time [batch, quarter] mask.  There is no RevIN or
per-window input centering/scaling, so absolute operating levels are preserved.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


REFERENCES = {
    "DLinear": "https://arxiv.org/abs/2205.13504",
    "LightTS": "https://arxiv.org/abs/2207.01186",
    "TimesNet": "https://openreview.net/forum?id=ju_Uqw384Oq",
    "Autoformer": "https://arxiv.org/abs/2106.13008",
    "FEDformer": "https://proceedings.mlr.press/v162/zhou22g.html",
    "ETSformer": "https://arxiv.org/abs/2202.01381",
    "Informer": "https://arxiv.org/abs/2012.07436",
    "Reformer": "https://openreview.net/forum?id=rkgNKkHtvB",
    "Pyraformer": "https://openreview.net/forum?id=0EXmFzUn5I",
}

MODEL_NOTES = {
    "common": (
        "New implementations trained from scratch; encoder features are adapted "
        "to classification using masked sequence flattening and a linear dim-wide "
        "readout. The caller owns the common classification head. Forecast-only "
        "decoders are omitted. No per-window RevIN or input mean/std scaling is "
        "used. LayerNorm/BatchNorm inside the original families are retained where "
        "specified. No claim of reproducing the original published hyperparameters. "
        "Invalid/pre-IPO input values are zeroed and excluded from final readouts. "
        "Within-window encoders are bidirectional: every supplied observation must "
        "already be available at the prediction cutoff."
    ),
    "DLinear": (
        "Moving-average seasonal/trend decomposition (kernel 5), two shared "
        "time-axis linear maps, summation, linear classification feature readout. "
        "Shared rather than channel-specific linear maps; depth is not applicable."
    ),
    "LightTS": (
        "Zhang et al., Less Is More, arXiv:2207.01186 (2022): continuous and "
        "interval sampling, information-exchange MLPs, channel mixing, and linear "
        "highway. NOT Campos et al.'s SIGMOD 2023 ensemble-distillation LightTS. "
        "Chunk size 5 for the 20-quarter experiment keeps the two sampling paths "
        "distinct. Right padding is applied only if needed. Depth is not applicable."
    ),
    "TimesNet": (
        "FFT-selected top-3 periods, 1D-to-2D folding, shared multi-kernel Inception "
        "convolutions, amplitude-weighted aggregation and residual blocks. Periods "
        "are selected per sample rather than averaged across companies in a batch, "
        "so evaluation predictions do not depend on test-batch composition. Kernels "
        "1/3/5 are used for the short sequence. No forecasting input normalization."
    ),
    "Autoformer": (
        "FFT query/key cross-correlation, top-log(T) delay selection, circular "
        "time-delay value aggregation, and progressive moving-average decomposition "
        "(kernel 5). Delay selection is per sample in train and eval, instead of the "
        "reference training batch-average shortcut. Encoder-only classification "
        "follows the public classification adaptation: decomposition removes the "
        "trend at each layer and there is no forecasting trend decoder. No claim "
        "is made that this encoder preserves every input level component. Ordinary "
        "LayerNorm is used at the output."
    ),
    "FEDformer": (
        "Fourier variant: fixed randomly selected non-Nyquist Fourier modes, "
        "learned complex per-head channel transforms, inverse FFT and progressive "
        "mixture-of-moving-average decomposition (kernels 3/5/9). Up to 8 modes. "
        "Selected modes are written back at their original frequency indices. "
        "Encoder-only classification follows the public classification adaptation, "
        "discarding extracted trends without a forecasting trend decoder; this "
        "does not guarantee preservation of every input level component. Wavelet "
        "variant and decoder cross-attention are not used. Complex parameters "
        "are stored as real/imag pairs for optimizers."
    ),
    "ETSformer": (
        "Frequency attention (top-3 non-DC/non-Nyquist harmonics), learned "
        "exponential smoothing of projected first differences for growth, and "
        "seasonally adjusted level smoothing with auxiliary growth. Residual, "
        "level, growth and seasonal encoder features feed the classification "
        "readout. The exact ES recurrence is evaluated sequentially for T=20 "
        "rather than by FFT convolution. Forecast damping/extrapolation and "
        "forecasting augmentations are omitted."
    ),
    "Informer": (
        "ProbSparse attention selects top queries using sampled key scores; "
        "unselected queries use mean value context. Sparsity factor 3, convolution "
        "+ masked BatchNorm + ELU + max-pool distillation between encoder layers. "
        "BatchNorm training statistics use valid history tokens only; eval uses "
        "stored running statistics and is independent of test-batch composition. Key "
        "samples are freshly drawn during training and fixed buffers in eval for "
        "reproducibility. Encoder-only classification, no generative decoder."
    ),
    "Reformer": (
        "Shared-QK angular LSH with 4 hashes, bucket size 4, sorted same/preceding "
        "chunk neighborhoods, duplicate-pair correction and log-normalizer "
        "aggregation across hashes. Additive reversible residual coupling and "
        "chunked feed-forward layers are retained. Hash rotations are fixed "
        "buffers. At T=20 the sparse LSH graph is evaluated with a dense masked "
        "kernel, and ordinary autograd stores activations: no O(T log T) runtime "
        "or memory-saving reversible-backprop claim is made."
    ),
    "Pyraformer": (
        "Convolutional coarse-scale construction, up to four scales with factor-2 "
        "downsampling, and pyramidal attention restricted to parent/child and "
        "neighboring same-scale nodes. Finest tokens gather all ancestor features "
        "before readout. Dense masked attention implements the sparse graph for "
        "short T; no TVM sparse-kernel speed claim. Forecast decoder omitted."
    ),
}


def _heads(dim: int, preferred: int = 4) -> int:
    return next(h for h in range(min(dim, preferred), 0, -1) if dim % h == 0)


def _zero_invalid(x: Tensor, valid: Tensor) -> Tensor:
    return x * valid.unsqueeze(-1).to(x.dtype)


def _masked_softmax(logits: Tensor, allowed: Tensor) -> Tensor:
    """Finite, zero result for an empty attention row (including all-padding)."""
    scores = logits.float().masked_fill(~allowed, -1e30)
    weights = scores.softmax(-1) * allowed.to(scores.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.to(logits.dtype)


class _Base(nn.Module):
    def __init__(self, input_dim: int, seq_len: int):
        super().__init__()
        self.input_dim, self.seq_len = input_dim, seq_len

    def inputs(self, x: Tensor, valid_time: Optional[Tensor]):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}"
            )
        if valid_time is None:
            valid_time = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        elif valid_time.shape != x.shape[:2]:
            raise ValueError("valid_time must have shape [B,T]")
        valid = valid_time.to(device=x.device, dtype=torch.bool)
        # where, unlike multiplication alone, also suppresses NaNs in padding.
        return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)), valid


class _Readout(nn.Module):
    def __init__(self, length: int, channels: int, dim: int, dropout: float,
                 activation: bool = True):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.map = nn.Linear(length * channels, dim)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        if self.activation:
            x = F.gelu(x)
        x = self.dropout(_zero_invalid(x, valid)).flatten(1)
        return self.map(x)


class _Embedding(nn.Module):
    def __init__(self, channels: int, dim: int, length: int, dropout: float,
                 position: bool = True):
        super().__init__()
        self.value = nn.Conv1d(channels, dim, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(length, dim)
        if position:
            t = torch.arange(length).float().unsqueeze(1)
            scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
            pe[:, 0::2] = torch.sin(t * scale)
            pe[:, 1::2] = torch.cos(t * scale[:dim // 2])
        self.register_buffer("position", pe)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        h = self.value(x.transpose(1, 2)).transpose(1, 2)
        return _zero_invalid(self.dropout(h + self.position), valid)


class _MovingAverage(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError("Moving-average kernel must be odd")
        self.kernel = kernel

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        pad = self.kernel // 2
        values = F.pad(_zero_invalid(x, valid).transpose(1, 2), (pad, pad), mode="replicate")
        weights = F.pad(valid[:, None, :].to(x.dtype), (pad, pad), mode="replicate")
        numerator = F.avg_pool1d(values, self.kernel, stride=1)
        denominator = F.avg_pool1d(weights, self.kernel, stride=1)
        return (numerator / denominator.clamp_min(1e-6)).transpose(1, 2)


class _Decomposition(nn.Module):
    def __init__(self, kernels=(5,)):
        super().__init__()
        self.averages = nn.ModuleList(_MovingAverage(k) for k in kernels)
        self.gate = nn.Linear(1, len(kernels)) if len(kernels) > 1 else None

    def forward(self, x: Tensor, valid: Tensor):
        trends = torch.stack([average(x, valid) for average in self.averages], -1)
        if self.gate is None:
            trend = trends[..., 0]
        else:
            weights = self.gate(x.unsqueeze(-1)).softmax(-1)
            trend = (trends * weights).sum(-1)
        return _zero_invalid(x - trend, valid), _zero_invalid(trend, valid)


class _FFN(nn.Module):
    def __init__(self, dim: int, dropout: float, multiplier: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim * multiplier), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * multiplier, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.network(x)


class DLinear(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.decompose = _Decomposition()
        self.seasonal = nn.Linear(seq_len, seq_len)
        self.trend = nn.Linear(seq_len, seq_len)
        nn.init.constant_(self.seasonal.weight, 1 / seq_len)
        nn.init.constant_(self.trend.weight, 1 / seq_len)
        self.readout = _Readout(seq_len, input_dim, dim, dropout, activation=False)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        seasonal, trend = self.decompose(x, valid)
        h = self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        return self.readout(h.transpose(1, 2), valid)


class _InformationExchange(nn.Module):
    """Mix features at each node, then exchange information across nodes."""
    def __init__(self, features: int, hidden: int, output: int, nodes: int):
        super().__init__()
        bottleneck = max(1, hidden // 4)
        self.feature = nn.Sequential(nn.Linear(features, hidden), nn.LeakyReLU(),
                                     nn.Linear(hidden, bottleneck))
        self.node = nn.Linear(nodes, nodes)
        nn.init.eye_(self.node.weight)
        nn.init.zeros_(self.node.bias)
        self.output = nn.Linear(bottleneck, output)

    def forward(self, x):
        # [B,features,nodes] -> [B,output,nodes]
        z = self.feature(x.transpose(1, 2)).transpose(1, 2)
        return self.output((z + self.node(z)).transpose(1, 2)).transpose(1, 2)


class LightTS(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.chunk = min(5, seq_len)
        self.chunks = math.ceil(seq_len / self.chunk)
        self.padded_length = self.chunk * self.chunks
        branch_dim = max(4, dim // 4)
        self.continuous = _InformationExchange(self.chunk, branch_dim, branch_dim, self.chunks)
        self.interval = _InformationExchange(self.chunk, branch_dim, branch_dim, self.chunks)
        self.reduce_continuous = nn.Linear(self.chunks, 1)
        self.reduce_interval = nn.Linear(self.chunks, 1)
        self.fuse = _InformationExchange(2 * branch_dim, 2 * branch_dim, seq_len, input_dim)
        self.highway = nn.Linear(self.padded_length, seq_len)
        self.readout = _Readout(seq_len, input_dim, dim, dropout, activation=False)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        b, _, c = x.shape
        padded = F.pad(x, (0, 0, 0, self.padded_length - self.seq_len))
        contiguous = padded.reshape(b, self.chunks, self.chunk, c).permute(0, 3, 2, 1)
        interleaved = padded.reshape(b, self.chunk, self.chunks, c).permute(0, 3, 1, 2)
        a = self.reduce_continuous(self.continuous(contiguous.reshape(b*c, self.chunk, self.chunks))).squeeze(-1)
        z = self.reduce_interval(self.interval(interleaved.reshape(b*c, self.chunk, self.chunks))).squeeze(-1)
        joint = torch.cat((a, z), -1).reshape(b, c, -1).transpose(1, 2)
        h = self.fuse(joint) + self.highway(padded.transpose(1, 2)).transpose(1, 2)
        return self.readout(h, valid)


class _Inception2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.kernels = nn.ModuleList(nn.Conv2d(in_dim, out_dim, k, padding=k // 2)
                                     for k in (1, 3, 5))

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.kernels], -1).mean(-1)


class _TimesBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(_Inception2D(dim, 2 * dim), nn.GELU(),
                                  _Inception2D(2 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        magnitude = torch.fft.rfft(_zero_invalid(x, valid).float(), dim=1).abs().mean(-1)
        if magnitude.shape[1] <= 1:
            return x
        # Each company's period dictionary comes from that company only.
        amplitudes, indices = magnitude[:, 1:].topk(min(self.top_k, magnitude.shape[1]-1), dim=-1)
        indices = indices + 1
        periods = torch.div(t, indices, rounding_mode="floor").clamp_min(1)
        weights = amplitudes.softmax(-1).to(x.dtype)
        combined = torch.zeros_like(x)
        for period in torch.unique(periods).tolist():
            period = int(period)
            n = math.ceil(t / period) * period
            grid = F.pad(x, (0, 0, 0, n-t)).reshape(b, n // period, period, d).permute(0, 3, 1, 2)
            folded = self.conv(grid).permute(0, 2, 3, 1).reshape(b, n, d)[:, :t]
            weight = (weights * (periods == period)).sum(-1)
            combined = combined + folded * weight[:, None, None]
        return _zero_invalid(x + self.dropout(combined), valid)


class TimesNet(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout)
        self.blocks = nn.ModuleList(_TimesBlock(dim, dropout) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        h = self.embed(x, valid)
        for block, norm in zip(self.blocks, self.norms):
            h = _zero_invalid(norm(block(h, valid)), valid)
        return self.readout(h, valid)


class _AutoCorrelation(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.heads = _heads(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q = q.permute(0, 2, 3, 1) * valid[:, None, None, :]
        k = k.permute(0, 2, 3, 1) * valid[:, None, None, :]
        v = v.permute(0, 2, 3, 1) * valid[:, None, None, :]
        corr = torch.fft.irfft(torch.fft.rfft(q.float(), dim=-1) *
                               torch.fft.rfft(k.float(), dim=-1).conj(), n=t, dim=-1)
        strength = corr.mean((1, 2))
        count = min(t, max(1, int(math.log(max(t, 2)))))
        scores, delays = strength.topk(count, dim=-1)
        weights = scores.softmax(-1).to(v.dtype)
        total = torch.zeros_like(v)
        effective = torch.zeros((b, 1, 1, t), dtype=v.dtype, device=x.device)
        positions = torch.arange(t, device=x.device)[None, :]
        for j in range(count):
            at = (positions + delays[:, j:j+1]) % t
            gathered = v.gather(-1, at[:, None, None, :].expand_as(v))
            present = valid.gather(1, at)[:, None, None, :].to(v.dtype)
            weight = weights[:, j, None, None, None]
            total = total + gathered * weight
            effective = effective + present * weight
        h = (total / effective.clamp_min(1e-6)).permute(0, 3, 1, 2).reshape(b, t, d)
        return _zero_invalid(self.out(self.dropout(h)), valid)


class _DecomposedLayer(nn.Module):
    def __init__(self, operator: nn.Module, dim: int, dropout: float, kernels=(5,)):
        super().__init__()
        self.operator = operator
        self.decompose1 = _Decomposition(kernels)
        self.decompose2 = _Decomposition(kernels)
        self.ff = _FFN(dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        x, _ = self.decompose1(x + self.dropout(self.operator(x, valid)), valid)
        x, _ = self.decompose2(x + self.ff(x), valid)
        return x


class Autoformer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout, position=False)
        self.layers = nn.ModuleList(_DecomposedLayer(_AutoCorrelation(dim, dropout), dim, dropout)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        h = self.embed(x, valid)
        for layer in self.layers:
            h = layer(h, valid)
        return self.readout(self.norm(h), valid)


class _FourierBlock(nn.Module):
    def __init__(self, dim: int, length: int, modes: int = 8):
        super().__init__()
        self.heads = _heads(dim)
        self.head_dim = dim // self.heads
        available = max(1, length // 2)  # Exclude the even-length Nyquist bin.
        indices = torch.randperm(available)[:min(modes, available)].sort().values
        self.register_buffer("modes", indices)
        self.project = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        shape = (self.heads, self.head_dim, self.head_dim, len(indices), 2)
        self.weight = nn.Parameter(torch.randn(shape) / math.sqrt(2 * self.head_dim))

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        h = _zero_invalid(self.project(x), valid).reshape(b, t, self.heads, self.head_dim)
        spectrum = torch.fft.rfft(h.float().permute(0, 2, 3, 1), dim=-1)
        weight = torch.view_as_complex(self.weight.float().contiguous())
        selected = spectrum.index_select(-1, self.modes)
        changed = torch.einsum("bhim,hiom->bhom", selected, weight)
        output = torch.zeros_like(spectrum).index_copy(-1, self.modes, changed)
        h = torch.fft.irfft(output, n=t, dim=-1).permute(0, 3, 1, 2).reshape(b, t, d)
        return _zero_invalid(self.out(h.to(x.dtype)), valid)


class FEDformer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout, position=False)
        self.layers = nn.ModuleList(_DecomposedLayer(_FourierBlock(dim, seq_len), dim,
                                                    dropout, kernels=(3, 5, 9))
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        h = self.embed(x, valid)
        for layer in self.layers:
            h = layer(h, valid)
        return self.readout(self.norm(h), valid)


class _ExponentialSmoothing(nn.Module):
    """The ES convolution expressed as its exact first-order recurrence."""
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.logit_alpha = nn.Parameter(torch.zeros(heads, 1))
        self.initial = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, auxiliary: Optional[Tensor] = None):
        b, t, d = x.shape
        values = self.dropout(x).reshape(b, t, self.heads, self.head_dim)
        aux = self.dropout(auxiliary).reshape_as(values) if auxiliary is not None else None
        alpha = self.logit_alpha.sigmoid().to(x.dtype)
        state = self.initial.to(x.dtype).unsqueeze(0).expand(b, -1, -1)
        sequence = []
        for j in range(t):
            update = alpha * state + (1 - alpha) * values[:, j]
            if aux is not None:
                update = update + alpha * aux[:, j]
            state = torch.where(valid[:, j, None, None], update, state)
            sequence.append(state * valid[:, j, None, None])
        return torch.stack(sequence, 1).reshape(b, t, d)


class _FrequencySeason(nn.Module):
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: Tensor, valid: Tensor):
        t = x.shape[1]
        spectrum = torch.fft.rfft(_zero_invalid(x, valid).float(), dim=1)
        stop = spectrum.shape[1] - (t % 2 == 0)
        if stop <= 1:
            return torch.zeros_like(x)
        available = spectrum[:, 1:stop]
        idx = available.abs().topk(min(self.top_k, available.shape[1]), dim=1).indices + 1
        kept = torch.zeros_like(spectrum).scatter(1, idx, spectrum.gather(1, idx))
        return _zero_invalid(torch.fft.irfft(kept, n=t, dim=1).to(x.dtype), valid)


class _ETSLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.season = _FrequencySeason()
        self.growth_in = nn.Linear(dim, dim)
        self.growth_out = nn.Linear(dim, dim)
        self.growth_start = nn.Parameter(torch.zeros(dim))
        self.growth_smoothing = _ExponentialSmoothing(dim, _heads(dim), dropout)
        self.level_smoothing = _ExponentialSmoothing(dim, dim, dropout)
        self.growth_to_level = nn.Linear(dim, dim)
        self.season_to_level = nn.Linear(dim, dim)
        self.ff = _FFN(dim, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, level: Tensor, valid: Tensor):
        seasonal = self.dropout(self.season(x, valid))
        residual = _zero_invalid(x - seasonal, valid)
        projected = self.growth_in(residual)
        previous = torch.cat((self.growth_start[None, None, :].expand(x.shape[0], 1, -1),
                              projected[:, :-1]), 1)
        previous_valid = F.pad(valid[:, :-1], (1, 0), value=False)
        previous = torch.where(previous_valid.unsqueeze(-1), previous,
                               self.growth_start[None, None, :])
        growth = self.growth_out(self.growth_smoothing(projected - previous, valid))
        growth = _zero_invalid(self.dropout(growth), valid)
        residual = self.norm1(residual - growth)
        residual = _zero_invalid(self.norm2(residual + self.ff(residual)), valid)
        lagged_growth = F.pad(growth[:, :-1], (0, 0, 1, 0))
        level = self.level_smoothing(level - self.season_to_level(seasonal), valid,
                                     auxiliary=self.growth_to_level(lagged_growth))
        return residual, level, growth, seasonal


class ETSformer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout, position=False)
        self.level_in = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList(_ETSLayer(dim, dropout) for _ in range(depth))
        self.components = nn.Linear(4 * dim, dim)
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        residual = self.embed(x, valid)
        level = _zero_invalid(self.level_in(x), valid)
        growth_sum, season_sum = torch.zeros_like(residual), torch.zeros_like(residual)
        for layer in self.layers:
            residual, level, growth, seasonal = layer(residual, level, valid)
            growth_sum, season_sum = growth_sum + growth, season_sum + seasonal
        h = self.components(torch.cat((residual, level, growth_sum, season_sum), -1))
        return self.readout(h, valid)


class _ProbSparse(nn.Module):
    def __init__(self, dim: int, length: int, dropout: float, factor: int = 3):
        super().__init__()
        self.heads = _heads(dim)
        self.samples = min(length, max(1, factor * math.ceil(math.log(max(length, 2)))))
        self.register_buffer("eval_indices", torch.randint(length, (length, self.samples)))
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        sampled_at = torch.randint(t, (t, self.samples), device=x.device) if self.training else self.eval_indices
        sampled = k[:, :, sampled_at, :]
        # Selection statistics use FP32 so masked sentinels are also AMP-safe.
        scores = (q.unsqueeze(-2) * sampled).sum(-1).float()
        sampled_valid = valid[:, sampled_at][:, None]
        maximum = scores.masked_fill(~sampled_valid, -1e30).max(-1).values
        total = (scores * sampled_valid).sum(-1) / valid.sum(-1).clamp_min(1)[:, None, None]
        importance = (maximum - total).masked_fill(~valid[:, None], -1e30)
        selected = importance.topk(min(t, self.samples), dim=-1).indices
        chosen_q = q.gather(2, selected.unsqueeze(-1).expand(-1, -1, -1, q.shape[-1]))
        logits = chosen_q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        weights = _masked_softmax(logits, valid[:, None, None, :])
        update = self.dropout(weights) @ v
        mean = (v * valid[:, None, :, None]).sum(2) / valid.sum(-1).clamp_min(1)[:, None, None]
        context = mean.unsqueeze(2).expand(-1, -1, t, -1).clone()
        context = context.scatter(2, selected.unsqueeze(-1).expand_as(update), update.to(context.dtype))
        h = context.transpose(1, 2).reshape(b, t, d)
        return _zero_invalid(self.out(h), valid)


class _InformerLayer(nn.Module):
    def __init__(self, dim: int, length: int, dropout: float):
        super().__init__()
        self.attention = _ProbSparse(dim, length, dropout)
        self.ff = _FFN(dim, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        x = _zero_invalid(self.norm1(x + self.dropout(self.attention(x, valid))), valid)
        return _zero_invalid(self.norm2(x + self.ff(x)), valid)


class _MaskedBatchNorm(nn.Module):
    """Channel BatchNorm whose statistics exclude invalid history tokens."""
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, x: Tensor, valid: Tensor):
        # x: [B,C,T]. Keep accumulation and variance in FP32 under autocast.
        xf = x.float()
        keep = valid[:, None, :]
        if self.training:
            count = valid.sum()
            if count.item() == 0:
                return torch.zeros_like(x)
            observed = torch.where(keep, xf, torch.zeros_like(xf))
            mean = observed.sum((0, 2)) / count
            centered = torch.where(keep, xf - mean[None, :, None], torch.zeros_like(xf))
            variance = centered.square().sum((0, 2)) / count
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                # Match nn.BatchNorm's unbiased running variance when possible.
                running_variance = variance.detach() * count / (count - 1).clamp_min(1)
                self.running_mean.lerp_(mean.detach().to(self.running_mean.dtype), self.momentum)
                self.running_var.lerp_(running_variance.to(self.running_var.dtype), self.momentum)
        else:
            mean, variance = self.running_mean.float(), self.running_var.float()
        z = (xf - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        z = z * self.weight.float()[None, :, None] + self.bias.float()[None, :, None]
        return torch.where(keep, z, torch.zeros_like(z)).to(x.dtype)


class _Distill(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm = _MaskedBatchNorm(dim)

    def forward(self, x: Tensor, valid: Tensor):
        h = F.elu(self.norm(self.conv(_zero_invalid(x, valid).transpose(1, 2)), valid))
        h = h.masked_fill(~valid[:, None], torch.finfo(h.dtype).min)
        h = F.max_pool1d(h, 3, stride=2, padding=1).transpose(1, 2)
        next_valid = F.max_pool1d(valid[:, None].float(), 3, stride=2, padding=1)[:, 0].bool()
        return torch.where(next_valid.unsqueeze(-1), h, torch.zeros_like(h)), next_valid


class Informer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout)
        lengths = [seq_len]
        for _ in range(depth - 1):
            lengths.append((lengths[-1] + 1) // 2)
        self.layers = nn.ModuleList(_InformerLayer(dim, length, dropout) for length in lengths)
        self.distill = nn.ModuleList(_Distill(dim) for _ in range(depth - 1))
        self.readout = _Readout(lengths[-1], dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        h = self.embed(x, valid)
        for i, layer in enumerate(self.layers):
            h = layer(h, valid)
            if i < len(self.distill):
                h, valid = self.distill[i](h, valid)
        return self.readout(h, valid)


class _LSHAttention(nn.Module):
    def __init__(self, dim: int, length: int, dropout: float, hashes: int = 4, bucket_size: int = 4):
        super().__init__()
        self.heads, self.hashes, self.bucket_size = _heads(dim), hashes, bucket_size
        self.head_dim = dim // self.heads
        buckets = max(2, math.ceil(length / bucket_size))
        self.buckets = buckets + buckets % 2
        self.register_buffer("rotations", torch.randn(self.heads, self.head_dim, hashes, self.buckets // 2))
        self.qk = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        qk = self.qk(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        keys = F.normalize(qk, dim=-1, eps=1e-6)
        values = self.value(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        # Hashing is nondifferentiable; attention scores remain differentiable.
        projected = torch.einsum("bhtd,hdrk->bhrtk", keys.detach(), self.rotations.to(keys.dtype))
        hashes = torch.cat((projected, -projected), -1).argmax(-1)
        token = torch.arange(t, device=x.device)
        hashes = torch.where(valid[:, None, None, :], hashes, self.buckets)
        order = (hashes * t + token).argsort(-1)
        ranks = order.argsort(-1)
        chunks = ranks // self.bucket_size
        own_or_previous = ((chunks.unsqueeze(-1) == chunks.unsqueeze(-2)) |
                           (chunks.unsqueeze(-1) == chunks.unsqueeze(-2) + 1))
        same_bucket = hashes.unsqueeze(-1) == hashes.unsqueeze(-2)
        allowed = own_or_previous & same_bucket & valid[:, None, None, None, :]
        allowed = allowed & valid[:, None, None, :, None]
        identity = torch.eye(t, device=x.device, dtype=torch.bool)
        has_other = (allowed & ~identity).any(-1, keepdim=True)
        allowed = allowed & (~identity | ~has_other)
        multiplicity = allowed.sum(2).clamp_min(1).float()
        logits = (qk @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits.unsqueeze(2) - multiplicity.log().unsqueeze(2)
        weights = _masked_softmax(logits, allowed)
        round_output = torch.einsum("bhrij,bhjd->bhrid", self.dropout(weights), values)
        normalizer = logits.float().masked_fill(~allowed, -1e30).logsumexp(-1)
        round_weight = normalizer.softmax(2).to(values.dtype)
        h = (round_output * round_weight.unsqueeze(-1)).sum(2).transpose(1, 2).reshape(b, t, d)
        return _zero_invalid(self.out(h), valid)


class _ReversibleCoupling(nn.Module):
    def __init__(self, dim: int, length: int, dropout: float):
        super().__init__()
        self.norm_attention, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = _LSHAttention(dim, length, dropout)
        self.ff = _FFN(dim, dropout)

    def _f(self, x, valid):
        return self.attention(self.norm_attention(x), valid)

    def _g(self, x, valid):
        # Chunking preserves the positionwise FFN and limits intermediate size.
        parts = [self.ff(self.norm_ff(part)) for part in x.split(8, dim=1)]
        return _zero_invalid(torch.cat(parts, 1), valid)

    def forward(self, first, second, valid):
        first = first + self._f(second, valid)
        second = second + self._g(first, valid)
        return _zero_invalid(first, valid), _zero_invalid(second, valid)

    def inverse(self, first, second, valid):
        if self.training:
            raise RuntimeError("inverse is for eval verification; training dropout needs RNG replay")
        second = second - self._g(first, valid)
        first = first - self._f(second, valid)
        return _zero_invalid(first, valid), _zero_invalid(second, valid)


class Reformer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout)
        self.layers = nn.ModuleList(_ReversibleCoupling(dim, seq_len, dropout) for _ in range(depth))
        self.merge = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        first = second = self.embed(x, valid)
        for layer in self.layers:
            first, second = layer(first, second, valid)
        h = self.norm(self.merge(torch.cat((first, second), -1)))
        return self.readout(h, valid)


class _PyramidLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.heads = _heads(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.project = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = _FFN(dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, graph: Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        logits = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        allowed = graph[None, None] & valid[:, None, None, :]
        weights = _masked_softmax(logits, allowed)
        h = (self.dropout(weights) @ v).transpose(1, 2).reshape(b, t, d)
        x = _zero_invalid(self.norm1(x + self.dropout(self.project(h))), valid)
        return _zero_invalid(self.norm2(x + self.ff(x)), valid)


class Pyraformer(_Base):
    def __init__(self, input_dim, seq_len=20, dim=64, depth=2, dropout=0.1):
        super().__init__(input_dim, seq_len)
        self.embed = _Embedding(input_dim, dim, seq_len, dropout)
        lengths = [seq_len]
        while len(lengths) < 4 and lengths[-1] > 1:
            lengths.append(math.ceil(lengths[-1] / 2))
        self.lengths = tuple(lengths)
        narrow = max(4, dim // 4)
        self.compress = nn.Linear(dim, narrow)
        self.coarsen = nn.ModuleList(nn.Conv1d(narrow, narrow, 2, stride=2)
                                     for _ in lengths[1:])
        self.expand = nn.ModuleList(nn.Linear(narrow, dim) for _ in lengths[1:])
        self.coarse_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in lengths[1:])
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)
        total = sum(lengths)
        graph = torch.zeros(total, total, dtype=torch.bool)
        for level, (offset, length) in enumerate(zip(offsets, lengths)):
            for node in range(length):
                graph[offset+node, offset+max(0, node-1):offset+min(length, node+2)] = True
                if level + 1 < len(lengths):
                    parent = offsets[level+1] + node // 2
                    graph[offset+node, parent] = True
                    graph[parent, offset+node] = True
        ancestors = torch.stack([torch.arange(seq_len) // (2 ** level) + offset
                                 for level, offset in enumerate(offsets)], -1)
        self.register_buffer("graph", graph)
        self.register_buffer("ancestors", ancestors)
        self.layers = nn.ModuleList(_PyramidLayer(dim, dropout) for _ in range(depth))
        self.gather_scales = nn.Linear(len(lengths) * dim, dim)
        self.readout = _Readout(seq_len, dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = self.inputs(x, valid_time)
        fine = self.embed(x, valid)
        representations, masks = [fine], [valid]
        coarse = _zero_invalid(self.compress(fine), valid).transpose(1, 2)
        mask = valid
        for conv, expand, norm in zip(self.coarsen, self.expand, self.coarse_norm):
            padding = coarse.shape[-1] % 2
            coarse = F.gelu(conv(F.pad(coarse, (0, padding))))
            mask = F.max_pool1d(F.pad(mask[:, None].float(), (0, padding)), 2, 2)[:, 0].bool()
            coarse = coarse * mask[:, None]
            representations.append(_zero_invalid(norm(expand(coarse.transpose(1, 2))), mask))
            masks.append(mask)
        h, pyramid_valid = torch.cat(representations, 1), torch.cat(masks, 1)
        for layer in self.layers:
            h = layer(h, pyramid_valid, self.graph)
        gathered = h[:, self.ancestors, :].flatten(2)
        return self.readout(self.gather_scales(gathered), valid)


_MODELS = {cls.__name__.lower(): cls for cls in
           (DLinear, LightTS, TimesNet, Autoformer, FEDformer, ETSformer, Informer, Reformer, Pyraformer)}


def build_backbone(name: str, input_dim: int, seq_len: int = 20,
                   dim: int = 64, depth: int = 2, dropout: float = 0.1):
    """Return a fresh classic backbone; return None for other model families."""
    cls = _MODELS.get(name.replace("_", "").replace("-", "").lower())
    if cls is None:
        return None
    if min(input_dim, seq_len, dim, depth) < 1 or not 0 <= dropout < 1:
        raise ValueError("Dimensions/depth must be positive and dropout in [0,1)")
    return cls(input_dim=input_dim, seq_len=seq_len, dim=dim, depth=depth, dropout=dropout)


__all__ = ["build_backbone", "MODEL_NOTES", "REFERENCES"]
