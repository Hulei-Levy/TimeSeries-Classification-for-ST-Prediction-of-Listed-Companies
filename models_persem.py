"""PerSeM and its own-history / native-frequency factorial ablations.

Independent rewrite of the user's ThisWork.py and MyEmbedding.py mechanisms;
no import from the old project.  All eight variants retain the same 20-quarter
absolute-level branch. Frozen document vectors are shared only as input data;
the single Linear(768,64) text projection is learned within each fold/model.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


VARIANTS = ("PerSeM", "PerSeM_A", "PerSeM_B", "PerSeM_C", "PerSeM_AB",
            "PerSeM_AC", "PerSeM_BC", "PerSeM_ABC")

MODEL_NOTES = {
    "core": (
        "Independent rewrite of ThisWork.py: circular Conv1d(k=3) token embedding, "
        "quarter one-hot projection and sinusoidal positions; FFT period folding, "
        "multi-kernel 2D Inception, amplitude aggregation, inner residual, temporal "
        "self-attention/LayerNorm, and the original outer shared-LayerNorm residual. "
        "Default dim=64, depth=2, d_ff=128, top_k=2, heads=4, kernels=1/3/5. Period "
        "selection is PER SAMPLE with DC excluded, rather than the old batch mean. "
        "Short sequences and all-missing masks are explicitly handled. No window "
        "centering/RevIN, extra pretraining or extra statistics branch."
    ),
    "classification": (
        "The original flattened time-position classification head is adapted to "
        "flatten(20*dim)->dim followed by LayerNorm->one logit, matching the common "
        "64-dimensional classifier interface while retaining quarter positions. "
        "Every variant keeps this same full-five-year absolute branch. A/B/C add "
        "64-dimensional residual evidence before the common head; they do not "
        "replace the absolute branch. Fixed width does not imply equal parameter "
        "counts across ablations. No pretrained time-series weights are loaded."
    ),
    "A": (
        "Within each modality, the latest calendar year is compared with valid "
        "year representations in the preceding 1/2/4 calendar years. These ranges "
        "always exclude the query year. A learned masked gate aggregates the "
        "three scales. Comparison features include query/reference vectors, signed "
        "and absolute differences, observed coverage and report-age gap. Modalities "
        "are compared separately before a learned modality fusion."
    ),
    "B": (
        "Within each company AND modality, cosine similarity retrieves at most "
        "two valid annual vectors from the preceding four calendar years. Learned "
        "scores reweight the selected comparisons. One candidate uses only one; "
        "no query-year observations or external companies enter the reference bank. "
        "Content similarity excludes absolute report age; age/coverage are supplied "
        "only after retrieval. Empty query/history gives an EXACT zero increment."
    ),
    "history_independence": (
        "A/B first extract native financial quarters, governance Q4 and MD&A Q2/Q4 "
        "reports, then split RAW inputs into five years BEFORE any convolution, "
        "attention or FFT. Each modality's year encoder is shared across its five "
        "independently encoded years. It uses within-year positions and quarter "
        "phase, not absolute age/year. Current-year perturbations therefore cannot "
        "change historical bank vectors. A/B share this bank when both enabled. "
        "Train-fold median placeholders never turn an all-missing year into a valid "
        "query/reference. Missing the fixed query year does not select an older one."
    ),
    "C": (
        "Additional native-frequency branch: financial 20 quarters, governance "
        "5 annual Q4 reports and MD&A 10 Q2/Q4 reports are encoded separately. "
        "Quarter-hold duplicates are excluded before encoding. Twenty quarter "
        "queries use the native financial hidden states plus fixed time positions, "
        "independently of the held-value absolute branch. They attend to each "
        "native sequence using per-head learned penalties "
        "on report-age distance. Masked modality gates fuse the results at each "
        "quarter; flatten->dim retains 20 time positions. All observations belong "
        "to the fixed prediction cutoff; within-window attention is bidirectional. "
        "Report age is age since report-period end, not true disclosure age."
    ),
    "missing": (
        "valid_time masks pre-IPO positions. Modality availability additionally "
        "requires an actual raw observed financial/governance field or available "
        "text report. Per-feature observed masks accompany numeric values. "
        "All-empty attention, reference and modality fusions return zero. "
        "The shared absolute-branch/head bias may still define a finite prior logit."
    ),
    "layout": (
        "layout[financial/governance] each has value_indices (expanded numeric "
        "and one-hot channels), observed_indices (raw-field 0/1 masks) and "
        "age_indices (raw-field ages in unstandardized YEARS). Root keys are "
        "text_available_index, text_age_index and optional valid_index. "
        "Frozen embeddings must be [N,768]; doc_ids use -1 for missing."
    ),
}


def _zero(x: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))


def _safe_softmax(scores: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    z = scores.float().masked_fill(~allowed, -1e30)
    w = z.softmax(dim) * allowed.to(z.dtype)
    return (w / w.sum(dim, keepdim=True).clamp_min(1e-12)).to(scores.dtype)


def _mean(x: Tensor, valid: Tensor, dim: int = 1) -> Tensor:
    w = valid.to(x.dtype)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1)


def _sinusoid(positions: Tensor, dim: int) -> Tensor:
    phase = positions.float().unsqueeze(-1) * torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    out = torch.zeros(len(positions), dim)
    out[:, 0::2], out[:, 1::2] = phase.sin(), phase[:, :dim // 2].cos()
    return out


class _QuarterEmbedding(nn.Module):
    def __init__(self, inputs: int, dim: int, quarters, dropout: float):
        super().__init__()
        quarters = torch.as_tensor(quarters, dtype=torch.long)
        self.token = nn.Conv1d(inputs, dim, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.quarter_map = nn.Linear(dim + 4, dim)
        self.register_buffer("quarter", F.one_hot(quarters % 4, 4).float())
        self.register_buffer("position", _sinusoid(quarters, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        h = self.token(_zero(x, valid).transpose(1, 2)).transpose(1, 2)
        phase = self.quarter[None].expand(x.shape[0], -1, -1).to(h.dtype)
        h = self.quarter_map(torch.cat((h, phase), -1))
        return _zero(self.dropout(h + self.position.to(h.dtype)), valid)


class _Inception(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(inputs, outputs, k, padding=k // 2)
                                   for k in (1, 3, 5))

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.convs], -1).mean(-1)


class _TemporalAttention(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.heads = 4
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        weights = self.dropout(_safe_softmax(scores, allowed))
        out = (weights @ v).transpose(1, 2).reshape(b, t, d)
        return _zero(self.output(out), valid)


class _PeriodAttentionBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(_Inception(dim, 2 * dim), nn.GELU(), _Inception(2 * dim, dim))
        self.attention = _TemporalAttention(dim, dropout)
        self.attention_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, valid: Tensor):
        b, t, d = x.shape
        magnitude = torch.fft.rfft(_zero(x, valid).float(), dim=1).abs().mean(-1)
        if t == 1:
            periods = torch.ones((b, 1), dtype=torch.long, device=x.device)
            weights = x.new_ones((b, 1))
        else:
            amplitude, bins = magnitude[:, 1:].topk(min(self.top_k, magnitude.shape[1] - 1), dim=-1)
            periods = torch.div(t, bins + 1, rounding_mode="floor").clamp_min(1)
            weights = amplitude.softmax(-1).to(x.dtype)
        mixed = torch.zeros_like(x)
        for period in torch.unique(periods).tolist():
            p = int(period)
            padded = math.ceil(t / p) * p
            grid = F.pad(x, (0, 0, 0, padded - t)).reshape(b, padded // p, p, d).permute(0, 3, 1, 2)
            h = self.conv(grid).permute(0, 2, 3, 1).reshape(b, padded, d)[:, :t]
            weight = (weights * (periods == p)).sum(-1)
            mixed = mixed + h * weight[:, None, None]
        inner = _zero(x + mixed, valid)
        return _zero(self.attention_norm(inner + self.attention(inner, valid)), valid)


class _PerSeMEncoder(nn.Module):
    def __init__(self, inputs: int, dim: int, quarters, depth: int, dropout: float):
        super().__init__()
        self.embedding = _QuarterEmbedding(inputs, dim, quarters, dropout)
        self.blocks = nn.ModuleList(_PeriodAttentionBlock(dim, dropout) for _ in range(depth))
        # This norm is shared across layers, matching the original outer loop.
        self.outer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor):
        h = self.embedding(x, valid)
        for block in self.blocks:
            h = _zero(h + self.outer_norm(block(h, valid)), valid)
        return _zero(self.dropout(F.gelu(h)), valid)


class _ModalityInput(nn.Module):
    def __init__(self, description: dict, numeric_dim: int, expected_raw: int):
        super().__init__()
        for name in ("value_indices", "observed_indices", "age_indices"):
            ix = torch.as_tensor(description[name], dtype=torch.long)
            if ix.ndim != 1 or len(ix) == 0 or (ix < 0).any() or (ix >= numeric_dim).any():
                raise ValueError(f"Invalid {name} in modality layout")
            self.register_buffer(name, ix)
        if len(self.observed_indices) != expected_raw or len(self.age_indices) != expected_raw:
            raise ValueError(f"Expected {expected_raw} raw observed masks and ages")
        self.content_dim = len(self.value_indices) + expected_raw
        self.full_dim = self.content_dim + expected_raw

    def forward(self, numeric: Tensor, ipo_valid: Tensor):
        values = numeric.index_select(-1, self.value_indices)
        observed = numeric.index_select(-1, self.observed_indices) > 0.5
        observed = observed & ipo_valid.unsqueeze(-1)
        ages = numeric.index_select(-1, self.age_indices)
        ages = torch.where(observed, ages, torch.zeros_like(ages))
        available = observed.any(-1)
        content = torch.cat((values, observed.to(values.dtype)), -1)
        content = _zero(content, available)
        full = _zero(torch.cat((content, ages), -1), available)
        report_age = ages.sum(-1) / observed.sum(-1).clamp_min(1)
        coverage = observed.float().mean(-1)
        return {"content": content, "full": full, "valid": available,
                "age": report_age, "coverage": coverage}


class _YearBank(nn.Module):
    """Split raw native inputs BEFORE embedding, circular conv, FFT or attention."""
    def __init__(self, content_dim: int, dim: int, quarters_per_year, depth: int, dropout: float):
        super().__init__()
        self.steps = len(quarters_per_year)
        self.encoder = _PerSeMEncoder(content_dim, dim, quarters_per_year, depth, dropout)

    def forward(self, view: dict):
        content, mask = view["content"], view["valid"]
        b, t, c = content.shape
        if t != 5 * self.steps:
            raise ValueError("Each history bank requires exactly five calendar years")
        split = content.reshape(b * 5, self.steps, c)
        split_mask = mask.reshape(b * 5, self.steps)
        sequence = self.encoder(split, split_mask)
        vectors = _mean(sequence, split_mask).reshape(b, 5, -1)
        valid = split_mask.any(-1).reshape(b, 5)
        age = _mean(view["age"].reshape(b * 5, self.steps, 1), split_mask).reshape(b, 5)
        coverage = view["coverage"].reshape(b, 5, self.steps).mean(-1)
        return {"vectors": _zero(vectors, valid), "valid": valid,
                "age": age, "coverage": coverage}


def _comparison_features(query, reference, query_coverage, reference_coverage, age_gap, fraction):
    q = query[:, None, :].expand_as(reference)
    meta = torch.stack((query_coverage[:, None].expand_as(reference_coverage),
                        reference_coverage, age_gap, fraction), -1).to(q.dtype)
    return torch.cat((q, reference, q - reference, (q - reference).abs(), meta), -1)


class _MultiScaleReference(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        width = 4 * dim + 4
        self.compare = nn.Sequential(nn.Linear(width, dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(width, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, bank: dict):
        vectors, valid = bank["vectors"], bank["valid"]
        refs, masks, coverages, gaps, fractions = [], [], [], [], []
        for years in (1, 2, 4):
            history_valid = valid[:, 4-years:4]
            refs.append(_mean(vectors[:, 4-years:4], history_valid))
            masks.append(history_valid.any(-1) & valid[:, 4])
            coverages.append(_mean(bank["coverage"][:, 4-years:4, None], history_valid).squeeze(-1))
            history_age = _mean(bank["age"][:, 4-years:4, None], history_valid).squeeze(-1)
            gaps.append(history_age - bank["age"][:, 4])
            fractions.append(history_valid.float().mean(-1))
        reference = torch.stack(refs, 1)
        available = torch.stack(masks, 1)
        features = _comparison_features(vectors[:, 4], reference, bank["coverage"][:, 4],
                                        torch.stack(coverages, 1), torch.stack(gaps, 1),
                                        torch.stack(fractions, 1))
        weights = _safe_softmax(self.gate(features).squeeze(-1), available)
        increments = _zero(self.compare(features), available)
        delta = (increments * weights.unsqueeze(-1)).sum(1)
        return {"delta": _zero(delta, available.any(-1)), "valid": available.any(-1),
                "weights": weights, "reference_valid": available}


class _RetrievedReference(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        width = 4 * dim + 4
        self.compare = nn.Sequential(nn.Linear(width, dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(width, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.similarity_scale = nn.Parameter(torch.zeros(()))

    def forward(self, bank: dict):
        vectors, valid = bank["vectors"], bank["valid"]
        query, history = vectors[:, 4], vectors[:, :4]
        similarity = (F.normalize(query.float(), dim=-1)[:, None] *
                      F.normalize(history.float(), dim=-1)).sum(-1)
        candidate_valid = valid[:, :4] & valid[:, 4:5]
        scores = similarity.masked_fill(~candidate_valid, -1e30)
        selected = scores.topk(2, dim=-1).indices
        reference = history.gather(1, selected.unsqueeze(-1).expand(-1, -1, vectors.shape[-1]))
        selected_valid = candidate_valid.gather(1, selected)
        coverage = bank["coverage"][:, :4].gather(1, selected)
        gap = bank["age"][:, :4].gather(1, selected) - bank["age"][:, 4:5]
        fraction = valid[:, :4].float().mean(-1, keepdim=True).expand_as(coverage)
        features = _comparison_features(query, reference, bank["coverage"][:, 4], coverage, gap, fraction)
        logits = self.gate(features).squeeze(-1).float() + F.softplus(self.similarity_scale) * similarity.gather(1, selected)
        weights = _safe_softmax(logits, selected_valid).to(features.dtype)
        increments = _zero(self.compare(features), selected_valid)
        delta = (increments * weights.unsqueeze(-1)).sum(1)
        return {"delta": _zero(delta, selected_valid.any(-1)), "valid": selected_valid.any(-1),
                "weights": weights, "selected_indices": selected.masked_fill(~selected_valid, -1),
                "selected_valid": selected_valid}


class _ModalityFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.modality_bias = nn.Parameter(torch.zeros(3))
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, values: Tensor, available: Tensor):
        # [B,(T),3,D] and [B,(T),3], identical operation for history and C.
        scores = self.gate(values).squeeze(-1) + self.modality_bias
        weights = _safe_softmax(scores, available)
        fused = (values * weights.unsqueeze(-1)).sum(-2)
        return _zero(self.output(fused), available.any(-1)), weights


class _TimeCrossAttention(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.heads = 4
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.output = nn.Linear(dim, dim)
        self.log_age_rate = nn.Parameter(torch.zeros(self.heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, source, query_valid, source_valid, query_age, source_age):
        b, t, d = query.shape
        n = source.shape[1]
        q = self.q(query).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = self.k(source).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        v = self.v(source).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)).float() / math.sqrt(d // self.heads)
        gap = (query_age[None, :, None] - source_age[:, None, :]).abs()
        logits = logits - F.softplus(self.log_age_rate)[None, :, None, None] * gap[:, None]
        allowed = query_valid[:, None, :, None] & source_valid[:, None, None, :]
        weights = self.dropout(_safe_softmax(logits, allowed).to(v.dtype))
        h = (weights @ v).transpose(1, 2).reshape(b, t, d)
        available = query_valid & source_valid.any(-1, keepdim=True)
        return _zero(self.output(h), available), available


class PerSeMClassifier(nn.Module):
    """Eight compatible classifiers; variants select residual A/B/C evidence."""
    def __init__(self, numeric_dim: int, layout: dict, embeddings, variant: str = "PerSeM",
                 dim: int = 64, depth: int = 2, dropout: float = 0.1, text_dim: int = 64):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if dim < 4 or dim % 4 or depth < 1 or text_dim != 64:
            raise ValueError("dim must be divisible by 4, depth >=1 and text_dim must be 64")
        if numeric_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid numeric_dim or dropout")
        self.variant, self.numeric_dim, self.dim = variant, numeric_dim, dim
        self.active = frozenset(variant.split("_", 1)[1] if "_" in variant else "")
        vectors = torch.as_tensor(embeddings, dtype=torch.float32).detach()
        if vectors.ndim != 2 or vectors.shape[1] != 768:
            raise ValueError("Frozen embeddings must be [N,768]")
        if vectors.shape[0] == 0:
            vectors = torch.zeros(1, 768)
        self.register_buffer("text_vectors", vectors, persistent=False)
        self.text_projection = nn.Linear(768, text_dim)
        self.absolute_encoder = _PerSeMEncoder(numeric_dim + text_dim, dim, range(20), depth, dropout)
        self.absolute_readout = nn.Linear(20 * dim, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.numeric_views = nn.ModuleDict({
            "financial": _ModalityInput(layout["financial"], numeric_dim, 30),
            "governance": _ModalityInput(layout["governance"], numeric_dim, 31),
        })
        self.modalities = ("financial", "governance", "text")
        self.text_available_index = int(layout["text_available_index"])
        self.text_age_index = int(layout["text_age_index"])
        for ix in (self.text_available_index, self.text_age_index):
            if not 0 <= ix < numeric_dim:
                raise ValueError("Text metadata index outside numeric input")
        self.register_buffer("financial_nodes", torch.arange(20))
        self.register_buffer("governance_nodes", torch.arange(3, 20, 4))
        self.register_buffer("text_nodes", torch.arange(1, 20, 2))
        self.register_buffer("query_report_age", (19 - torch.arange(20)).float() / 4 + 0.5)
        content_dims = {"financial": self.numeric_views["financial"].content_dim,
                        "governance": self.numeric_views["governance"].content_dim,
                        "text": text_dim + 1}
        full_dims = {"financial": self.numeric_views["financial"].full_dim,
                     "governance": self.numeric_views["governance"].full_dim,
                     "text": text_dim + 2}
        year_quarters = {"financial": (0, 1, 2, 3), "governance": (3,), "text": (1, 3)}
        if self.active & {"A", "B"}:
            self.year_encoders = nn.ModuleDict({m: _YearBank(content_dims[m], dim, year_quarters[m], depth, dropout)
                                                for m in self.modalities})
        if "A" in self.active:
            self.scale_comparisons = nn.ModuleDict({m: _MultiScaleReference(dim, dropout) for m in self.modalities})
            self.A_fusion = _ModalityFusion(dim)
        if "B" in self.active:
            self.retrieval_comparisons = nn.ModuleDict({m: _RetrievedReference(dim, dropout) for m in self.modalities})
            self.B_fusion = _ModalityFusion(dim)
        if "C" in self.active:
            self.native_encoders = nn.ModuleDict({m: _PerSeMEncoder(full_dims[m], dim, getattr(self, m + "_nodes"), depth, dropout)
                                                  for m in self.modalities})
            self.cross_attention = nn.ModuleDict({m: _TimeCrossAttention(dim, dropout) for m in self.modalities})
            self.C_fusion = _ModalityFusion(dim)
            self.C_readout = nn.Linear(20 * dim, dim, bias=False)
            self.C_dropout = nn.Dropout(dropout)
            self.register_buffer("C_time_query", _sinusoid(torch.arange(20), dim))

    def _prepare(self, numeric, doc_ids, valid_time):
        if numeric.ndim != 3 or numeric.shape[1:] != (20, self.numeric_dim):
            raise ValueError(f"numeric must be [B,20,{self.numeric_dim}]")
        if doc_ids.shape != numeric.shape[:2] or valid_time.shape != numeric.shape[:2]:
            raise ValueError("doc_ids and valid_time must be [B,20]")
        valid = valid_time.to(device=numeric.device, dtype=torch.bool)
        numeric = _zero(numeric, valid)
        doc_ids = doc_ids.to(device=numeric.device, dtype=torch.long)
        available = (doc_ids >= 0) & valid & (numeric[..., self.text_available_index] > 0.5)
        safe_ids = torch.where(available, doc_ids, torch.zeros_like(doc_ids))
        text = _zero(self.text_projection(self.text_vectors[safe_ids]), available)
        age = torch.where(available, numeric[..., self.text_age_index], torch.zeros_like(numeric[..., self.text_age_index]))
        return numeric, text, valid, available, age

    def _native_views(self, numeric, text, valid, text_available, text_age):
        views = {m: self.numeric_views[m](numeric, valid) for m in ("financial", "governance")}
        text_content = torch.cat((text, text_available.unsqueeze(-1).to(text.dtype)), -1)
        views["text"] = {"content": _zero(text_content, text_available),
                         "full": _zero(torch.cat((text_content, text_age.unsqueeze(-1)), -1), text_available),
                         "valid": text_available, "age": text_age,
                         "coverage": text_available.float()}
        return {m: {key: value.index_select(1, getattr(self, m + "_nodes"))
                    for key, value in view.items()} for m, view in views.items()}

    def native_views(self, numeric, doc_ids, valid_time):
        """Diagnostic access to genuine 20/5/10 native sequences before encoding."""
        return self._native_views(*self._prepare(numeric, doc_ids, valid_time))

    def encode_history_banks(self, numeric, doc_ids, valid_time):
        """Independent raw-year banks for auditing; requires A or B to be enabled."""
        if not self.active & {"A", "B"}:
            raise RuntimeError("This variant has no historical comparison branch")
        views = self.native_views(numeric, doc_ids, valid_time)
        return {m: self.year_encoders[m](views[m]) for m in self.modalities}

    def compare_history_banks(self, banks):
        result = {}
        for letter, attribute in (("A", "scale_comparisons"), ("B", "retrieval_comparisons")):
            if letter not in self.active:
                continue
            comparators = getattr(self, attribute)
            per_modal = {m: comparators[m](banks[m]) for m in self.modalities}
            values = torch.stack([per_modal[m]["delta"] for m in self.modalities], 1)
            available = torch.stack([per_modal[m]["valid"] for m in self.modalities], 1)
            delta, gates = getattr(self, letter + "_fusion")(values, available)
            result[letter] = {"delta": delta, "modality_weights": gates, "modalities": per_modal}
        return result

    def forward_features(self, numeric, doc_ids, valid_time):
        numeric, text, valid, available, text_age = self._prepare(numeric, doc_ids, valid_time)
        absolute_sequence = self.absolute_encoder(torch.cat((numeric, text), -1), valid)
        absolute = self.absolute_readout(absolute_sequence.flatten(1))
        output = {"absolute": absolute}
        if self.active:
            views = self._native_views(numeric, text, valid, available, text_age)
        if self.active & {"A", "B"}:
            banks = {m: self.year_encoders[m](views[m]) for m in self.modalities}
            comparisons = self.compare_history_banks(banks)
            for letter, result in comparisons.items():
                output[letter] = result["delta"]
        if "C" in self.active:
            messages, masks = [], []
            native_hidden = {m: self.native_encoders[m](views[m]["full"], views[m]["valid"])
                             for m in self.modalities}
            # The C branch never reads held quarterly governance/text copies
            # through its queries. A time query remains usable when finance is
            # missing, provided the firm is already listed at this quarter.
            queries = _zero(native_hidden["financial"] + self.C_time_query, valid)
            for m in self.modalities:
                view = views[m]
                message, mask = self.cross_attention[m](queries, native_hidden[m], valid, view["valid"],
                                                        self.query_report_age, view["age"])
                messages.append(message)
                masks.append(mask)
            fused, _ = self.C_fusion(torch.stack(messages, 2), torch.stack(masks, 2))
            output["C"] = self.C_readout(self.C_dropout(F.gelu(fused)).flatten(1))
        output["representation"] = sum((output[letter] for letter in "ABC" if letter in self.active), absolute)
        return output

    def forward(self, numeric, doc_ids, valid_time):
        return self.head(self.forward_features(numeric, doc_ids, valid_time)["representation"]).squeeze(-1)

    def auxiliary_loss(self):
        return 0.0


__all__ = ["PerSeMClassifier", "VARIANTS", "MODEL_NOTES"]
