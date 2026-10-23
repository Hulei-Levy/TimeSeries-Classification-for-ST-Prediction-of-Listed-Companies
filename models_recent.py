"""Small, from-scratch classification adaptations of five forecasting architectures.

This file is newly implemented from the cited method descriptions. It imports
neither the old project nor upstream model packages, and loads no checkpoints.
These are architecture comparisons, NOT reproductions of foundation-model
pretraining, zero-shot forecasting, or the papers' original parameter budgets.

Common adaptation: fold-training-set scaling is supplied by the data pipeline;
forecast-specific instance normalization/RevIN is disabled to preserve absolute
financial levels. Forecast/quantile/reconstruction heads and pretraining losses
are replaced by the caller's binary classification head. No per-model mean/std
summary shortcut is added. Patch size 4 is chosen for the 20-quarter input.

Sources checked 2026-10-22:
TiRex: https://arxiv.org/abs/2505.23719 ; https://github.com/NX-AI/tirex
xLSTM-Mixer: https://arxiv.org/abs/2410.16928 ;
    https://github.com/mauricekraus/xlstm-mixer
TimeFilter: https://arxiv.org/abs/2501.13041 ;
    https://github.com/TROUBADOUR000/TimeFilter
Time-MoE: https://arxiv.org/abs/2409.16040 ;
    https://github.com/Time-MoE/Time-MoE
SEMPO: https://arxiv.org/abs/2510.19710 ; https://github.com/mala-lab/SEMPO

Input: [batch, time, feature], plus optional valid_time [batch,time]. Missingness
and report-age channels are already included in feature. Output: [batch,dim].
TimeFilter/TimeMoE expose unweighted `last_aux_loss`; recommended multiplier .01.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NOTES = {
    'TiRex': '4-step residual patches; headwise exponential-gated sLSTM and RMS/SwiGLU residual blocks; channel-independent backbone; no CPM pretraining, quantile loss, or future-patch generation.',
    'XLSTMMixer': 'NLinear time mixing, learned initial memory token, sLSTM recurrence over feature tokens, shared latent-dimension reversed view, learned view reconciliation; no RevIN.',
    'TimeFilter': 'Per-channel 4-step patches; multi-head kNN graph; noisy top-p routing over spatial, temporal, and cross-time cross-channel edges; graph residual layers.',
    'TimeMoE': 'Multivariate gated input embedding, causal RoPE attention, RMSNorm, sparse top-2 of 4 temporal SwiGLU experts plus gated shared expert; no forecast heads/pretraining.',
    'SEMPO': 'Channel-independent high/low-energy spectral decomposition, 4 independently masked spectral views, patching, routed prompt K/V in encoder and decoder; all parameters trained jointly for classification, no frozen pretrained stage.',
}


def _prepare(x, valid_time, expected_len):
    if x.ndim != 3 or x.shape[1] != expected_len:
        raise ValueError(f'Expected [B,{expected_len},C], got {tuple(x.shape)}')
    if valid_time is None:
        valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    else:
        valid = valid_time.to(device=x.device, dtype=torch.bool)
        if valid.shape != x.shape[:2]:
            raise ValueError('valid_time must have shape [B,T]')
    # Ignore arbitrary padding values, including nonfinite values, exactly.
    return torch.where(valid[..., None], x, torch.zeros_like(x)), valid


def _masked_mean(x, valid):
    w = valid.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp_min(1)


def _patch_channels(x, valid, patch):
    b, t, c = x.shape
    pad = (-t) % patch
    xp = F.pad(x.transpose(1, 2), (0, pad)).unfold(-1, patch, patch)
    vp = F.pad(valid, (0, pad), value=False).unfold(-1, patch, patch)
    return xp, vp


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.):
        super().__init__()
        hidden = hidden or dim * 4
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, -1)
        return self.dropout(self.down(F.silu(gate) * value))


class ScalarMemory(nn.Module):
    """sLSTM equations with scalar c/n/m states and block-diagonal recurrence.

    Both input and forget gates are exponential, stabilized in log space. Each
    recurrent head mixes its own hidden coordinates. TiRex uses headwise input
    maps; xLSTM-Mixer uses dense input maps as in its xLSTM block description.
    """
    def __init__(self, dim, heads=4, headwise_input=False):
        super().__init__()
        if dim % heads:
            raise ValueError('sLSTM dimension must be divisible by heads')
        self.dim, self.heads, self.width = dim, heads, dim // heads
        self.headwise_input = headwise_input
        if headwise_input:
            self.input_weight = nn.Parameter(torch.empty(heads, self.width, 4 * self.width))
            nn.init.xavier_uniform_(self.input_weight)
        else:
            self.input_map = nn.Linear(dim, 4 * dim, bias=False)
        self.recurrent = nn.Parameter(torch.empty(heads, self.width, 4 * self.width))
        nn.init.xavier_uniform_(self.recurrent)
        self.bias = nn.Parameter(torch.zeros(heads, 4, self.width))
        with torch.no_grad():
            self.bias[:, 1].fill_(1.)  # initial retention bias
        self.out_weight = nn.Parameter(torch.ones(dim))
        self.out_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, valid=None):
        b, t, _ = x.shape
        dtype = x.dtype
        # A scalar recurrence in fp32 avoids AMP under/overflow in c/n states.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            if self.headwise_input:
                projected = torch.einsum('bthd,hdk->bthk', xf.reshape(b,t,self.heads,self.width), self.input_weight.float())
                projected = projected.reshape(b,t,self.heads,4,self.width)
            else:
                projected = F.linear(xf, self.input_map.weight.float()).reshape(b,t,4,self.heads,self.width).transpose(2,3)
            h = xf.new_zeros(b, self.heads, self.width)
            c, n, m = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
            values = []
            for step in range(t):
                rec = torch.einsum('bhd,hdk->bhk', h, self.recurrent.float()).reshape(b,self.heads,4,self.width)
                zi, lf, li, lo = (projected[:,step] + rec + self.bias.float()).unbind(2)
                next_m = torch.maximum(lf + m, li)
                fg, ig = torch.exp(lf + m - next_m), torch.exp(li - next_m)
                next_c = fg * c + ig * zi.tanh()
                next_n = fg * n + ig
                next_h = lo.sigmoid() * (next_c / next_n.clamp_min(1e-8))
                if valid is not None:
                    keep = valid[:,step,None,None]
                    h, c, n, m = [torch.where(keep, a, old) for a, old in zip((next_h,next_c,next_n,next_m),(h,c,n,m))]
                    values.append(torch.where(keep, h, torch.zeros_like(h)))
                else:
                    h, c, n, m = next_h,next_c,next_n,next_m
                    values.append(h)
            y = torch.stack(values, 1)
            y = F.layer_norm(y, (self.width,)).reshape(b,t,self.dim)
            y = y * self.out_weight.float() + self.out_bias.float()
        return y.to(dtype)


class ScalarBlock(nn.Module):
    def __init__(self, dim, dropout, heads=4, headwise_input=False, rms=True):
        super().__init__()
        norm = RMSNorm if rms else nn.LayerNorm
        self.norm1, self.norm2 = norm(dim), norm(dim)
        self.memory = ScalarMemory(dim, heads, headwise_input)
        self.ff = SwiGLU(dim, dim * 2, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid=None):
        x = x + self.drop(self.memory(self.norm1(x), valid))
        x = x + self.ff(self.norm2(x))
        return x if valid is None else x * valid[...,None].to(x.dtype)


class ChannelReadout(nn.Module):
    """Keeps feature identity when reducing channel-independent representations."""
    def __init__(self, channels, dim, dropout):
        super().__init__()
        self.projection = nn.Linear(channels * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.projection(self.dropout(x.flatten(1)))


class TiRex(nn.Module):
    def __init__(self, input_dim, seq_len, dim, depth, dropout):
        super().__init__()
        self.seq_len, self.patch = seq_len, min(4, seq_len)
        p = 2 * self.patch
        self.patch_skip = nn.Linear(p, dim)
        self.patch_mlp = nn.Sequential(nn.Linear(p, dim), nn.SiLU(), nn.Linear(dim, dim), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ScalarBlock(dim, dropout, headwise_input=True) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.readout = ChannelReadout(input_dim, dim, dropout)

    def forward(self, x, valid_time=None):
        x, valid = _prepare(x, valid_time, self.seq_len)
        b, _, c = x.shape
        patches, pm = _patch_channels(x, valid, self.patch)
        pvalid = pm.any(-1)[:,None,:].expand(b,c,-1).reshape(b*c,-1)
        values = torch.cat((patches, pm[:,None].expand_as(patches).to(x.dtype)), -1).reshape(b*c,-1,2*self.patch)
        h = self.patch_skip(values) + self.patch_mlp(values)
        for block in self.blocks:
            h = block(h, pvalid)
        h = _masked_mean(self.norm(h), pvalid).reshape(b,c,-1)
        return self.readout(h)


class XLSTMMixer(nn.Module):
    def __init__(self, input_dim, seq_len, dim, depth, dropout):
        super().__init__()
        self.seq_len = seq_len
        # Initial NLinear output length is a latent 20-step refinement grid.
        self.time_mix = nn.Linear(seq_len, seq_len)
        self.up = nn.Linear(seq_len, dim)
        self.memory_token = nn.Parameter(torch.randn(1,1,dim) * .02)
        self.blocks = nn.ModuleList([ScalarBlock(dim, dropout, heads=8, rms=False) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.view_mix = nn.Linear(dim * 2, dim)
        self.readout = ChannelReadout(input_dim, dim, dropout)

    def _run(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x, valid_time=None):
        x, valid = _prepare(x, valid_time, self.seq_len)
        last_index = (valid.long() * torch.arange(1,self.seq_len+1,device=x.device)).amax(1).clamp_min(1)-1
        last = x[torch.arange(x.shape[0],device=x.device),last_index][:,None,:]
        centered = torch.where(valid[...,None], x-last, torch.zeros_like(x))
        z = self.time_mix(centered.transpose(1,2)) + last.transpose(1,2)
        z = self.up(z)
        z = torch.cat((self.memory_token.expand(x.shape[0],-1,-1), z), 1)
        # The published implementation reverses LATENT coordinates, not tokens.
        a, b = self._run(z), self._run(z.flip(-1))
        return self.readout(self.view_mix(torch.cat((a[:,1:],b[:,1:]), -1)))


class GraphFilterBlock(nn.Module):
    def __init__(self, channels, patches, dim, dropout, heads=4, alpha=.2, top_p=.8):
        super().__init__()
        self.heads, self.alpha, self.top_p = heads, alpha, top_p
        n = channels * patches
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q = nn.Linear(dim//heads, dim//heads)
        self.k = nn.Linear(dim//heads, dim//heads)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.router = nn.Linear(n, 3, bias=False)
        self.noise = nn.Linear(n, 3, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*2,dim))
        self.drop = nn.Dropout(dropout)
        idx = torch.arange(n)
        same_channel = (idx[:,None]//patches == idx[None,:]//patches)
        same_patch = (idx[:,None]%patches == idx[None,:]%patches)
        regions = torch.stack((same_channel, same_patch & ~same_channel, ~same_patch & ~same_channel))
        self.register_buffer('regions', regions, persistent=False)
        self.last_aux_loss = None

    def forward(self, z, valid):
        b,n,d = z.shape
        h = self.norm1(z).reshape(b,n,self.heads,d//self.heads).transpose(1,2)
        raw = F.gelu(self.q(h) @ self.k(h).transpose(-1,-2))
        # Top-k neighbors are selected among valid observed-history patches.
        scores = raw.masked_fill(~valid[:,None,None,:], -torch.inf)
        k = max(1, min(n, math.ceil(n * self.alpha)))
        ix = scores.topk(k,dim=-1).indices
        keep = torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,ix,True)
        keep = keep & valid[:,None,None,:] & valid[:,None,:,None]
        adj = torch.where(keep, raw, torch.zeros_like(raw))
        logits = self.router(adj)
        if self.training:
            logits = logits + torch.randn_like(logits) * (F.softplus(self.noise(adj)) + .01)
        prob = logits.float().softmax(-1)
        sorted_p, order = prob.sort(-1,descending=True)
        selected_sorted = sorted_p.cumsum(-1) - sorted_p < self.top_p
        selected = torch.zeros_like(selected_sorted).scatter_(-1,order,selected_sorted)
        route = (prob * selected).to(adj.dtype)
        # Weighted filtering retains gradients to the router (Eq.10).
        region_weight = sum(route[...,j,None] * self.regions[j] for j in range(3))
        edge_valid = keep & (region_weight > 0)
        eye = torch.eye(n,device=z.device,dtype=torch.bool)[None,None]
        edge_valid = edge_valid | (eye & valid[:,None,:,None])
        weights = (adj * region_weight).masked_fill(~edge_valid, -torch.inf)
        weights = torch.nan_to_num(weights.float().softmax(-1)).to(z.dtype)
        v = self.value(z).reshape(b,n,self.heads,d//self.heads).transpose(1,2)
        message = (weights @ v).transpose(1,2).reshape(b,n,d)
        z = z + self.drop(self.out(message))
        z = z + self.drop(self.ff(self.norm2(z)))
        valid_float = valid[:,None,:,None].to(prob.dtype)
        total = valid_float.sum().clamp_min(1) * self.heads
        importance = (prob * valid_float).sum((0,1,2)) / total
        balance = importance.var(unbiased=False) / importance.mean().square().clamp_min(1e-8)
        entropy = (-(prob * prob.clamp_min(1e-8).log()).sum(-1) * valid_float[...,0]).sum() / total
        self.last_aux_loss = balance + .1 * entropy
        return z * valid[...,None].to(z.dtype)


class TimeFilter(nn.Module):
    def __init__(self, input_dim, seq_len, dim, depth, dropout):
        super().__init__()
        self.seq_len, self.patch = seq_len, min(4,seq_len)
        self.npatch = math.ceil(seq_len/self.patch)
        self.embed = nn.Linear(self.patch,dim)
        self.position = nn.Parameter(torch.randn(1,input_dim,self.npatch,dim) * .02)
        self.blocks = nn.ModuleList([GraphFilterBlock(input_dim,self.npatch,dim,dropout) for _ in range(depth)])
        self.readout = ChannelReadout(input_dim,dim,dropout)
        self.last_aux_loss = None

    def forward(self, x, valid_time=None):
        x, valid = _prepare(x,valid_time,self.seq_len)
        b,_,c = x.shape
        patches, pm = _patch_channels(x,valid,self.patch)
        pv = pm.any(-1)
        nv = pv[:,None].expand(b,c,-1).reshape(b,-1)
        z = (self.embed(patches) + self.position).reshape(b,-1,self.position.shape[-1])
        losses = []
        for block in self.blocks:
            z = block(z,nv)
            losses.append(block.last_aux_loss)
        self.last_aux_loss = torch.stack(losses).mean()
        z = z.reshape(b*c,self.npatch,-1)
        return self.readout(_masked_mean(z,nv.reshape(b*c,self.npatch)).reshape(b,c,-1))


class RotaryAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        if dim % (2*heads):
            raise ValueError('RoPE needs an even head dimension')
        self.heads, self.width, self.dropout = heads, dim//heads, dropout
        self.qkv = nn.Linear(dim,3*dim)
        self.out = nn.Linear(dim,dim,bias=False)
        self.register_buffer('inv_freq',1/(10000**(torch.arange(0,self.width,2).float()/self.width)),persistent=False)

    def _rotate(self, x):
        t = x.shape[-2]
        angle = torch.outer(torch.arange(t,device=x.device).float(),self.inv_freq)
        cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        even, odd = x[...,0::2], x[...,1::2]
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),-1).flatten(-2)

    def forward(self, x, valid):
        b,t,d = x.shape
        q,k,v = self.qkv(x).reshape(b,t,3,self.heads,self.width).permute(2,0,3,1,4).unbind(0)
        q,k = self._rotate(q),self._rotate(k)
        causal = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        allowed = causal[None,None] & valid[:,None,None,:]
        h = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.dropout if self.training else 0.)
        return self.out(h.transpose(1,2).reshape(b,t,d)) * valid[...,None].to(x.dtype)


class SparseTemporalExperts(nn.Module):
    def __init__(self, dim, experts=4, top_k=2, dropout=0.):
        super().__init__()
        self.expert_count, self.top_k = experts,top_k
        self.gate = nn.Linear(dim,experts,bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim,dim*2,dropout) for _ in range(experts)])
        self.shared = SwiGLU(dim,dim*4,dropout)
        self.shared_gate = nn.Linear(dim,1,bias=False)
        self.last_aux_loss = None

    def forward(self, x, valid):
        shape = x.shape
        flat = x.reshape(-1,shape[-1])
        probability = self.gate(flat).float().softmax(-1)
        weight, selected = probability.topk(self.top_k,-1)
        y = torch.zeros_like(flat)
        live = valid.reshape(-1)
        for idx, expert in enumerate(self.experts):
            token, rank = torch.where((selected == idx) & live[:,None])
            if token.numel():
                value = expert(flat[token]) * weight[token,rank,None].to(x.dtype)
                y = y.index_add(0,token,value)
        y = y + self.shared_gate(flat).sigmoid() * self.shared(flat)
        count = live.sum().clamp_min(1)
        usage = (F.one_hot(selected,self.expert_count).float() * live[:,None,None]).sum((0,1)) / (count*self.top_k)
        mean_probability = (probability * live[:,None]).sum(0) / count
        self.last_aux_loss = self.expert_count * (usage * mean_probability).sum()
        return y.reshape(shape) * valid[...,None].to(x.dtype)


class TimeMoEBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm1,self.norm2 = RMSNorm(dim),RMSNorm(dim)
        self.attention = RotaryAttention(dim,dropout=dropout)
        self.experts = SparseTemporalExperts(dim,dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self,x,valid):
        x = x + self.drop(self.attention(self.norm1(x),valid))
        x = x + self.experts(self.norm2(x),valid)
        return x * valid[...,None].to(x.dtype)


class TimeMoE(nn.Module):
    def __init__(self,input_dim,seq_len,dim,depth,dropout):
        super().__init__()
        self.seq_len = seq_len
        self.input_value = nn.Linear(input_dim,dim,bias=False)
        self.input_gate = nn.Linear(input_dim,dim,bias=False)
        self.blocks = nn.ModuleList([TimeMoEBlock(dim,dropout) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.last_aux_loss = None

    def forward(self,x,valid_time=None):
        x,valid = _prepare(x,valid_time,self.seq_len)
        h = self.input_value(x) * F.silu(self.input_gate(x))
        losses = []
        for block in self.blocks:
            h = block(h,valid)
            losses.append(block.experts.last_aux_loss)
        self.last_aux_loss = torch.stack(losses).mean()
        return _masked_mean(self.norm(h),valid)


class SpectralViews(nn.Module):
    """EASD with separate high/low energy masks and four learned band views.

    Hard threshold masks are used in the forward pass; sigmoid straight-through
    derivatives train the energy/band thresholds, rather than leaving them dead
    under Boolean comparisons. This estimator is an explicit adaptation.
    """
    def __init__(self, seq_len, views=4):
        super().__init__()
        self.views, self.seq_len = views, seq_len
        self.log_energy_threshold = nn.Parameter(torch.zeros(()))
        self.high_cut = nn.Parameter(torch.tensor([-1.1,0.,1.1,6.]))
        self.low_cut = nn.Parameter(torch.tensor([1.1,0.,-1.1,6.]))
        self.register_buffer('high_direction',torch.tensor([1.,-1.,1.,1.]))
        self.register_buffer('low_direction',torch.tensor([-1.,1.,-1.,1.]))

    @staticmethod
    def _hard_st(logit):
        soft = logit.sigmoid()
        return (logit >= 0).to(soft.dtype).detach() - soft.detach() + soft

    def forward(self,x):
        # x is [B,C,T]; float32 FFT is required for non-power-of-two lengths.
        with torch.autocast(device_type=x.device.type,enabled=False):
            spectrum = torch.fft.rfft(x.float(),dim=-1,norm='ortho')
            energy = spectrum.abs().square()
            ratio = energy / energy.median(-1,keepdim=True).values.clamp_min(1e-6)
            dominant = self._hard_st((ratio - self.log_energy_threshold.exp()).clamp(-40,40))
            high,low = spectrum*dominant,spectrum*(1-dominant)
            f = torch.arange(spectrum.shape[-1],device=x.device).float()
            maxf = float(spectrum.shape[-1])
            mh = self._hard_st((self.high_cut.sigmoid()[:,None]*maxf-f) * self.high_direction[:,None])
            ml = self._hard_st((self.low_cut.sigmoid()[:,None]*maxf-f) * self.low_direction[:,None])
            z = high[:,:,None,:]*mh + low[:,:,None,:]*ml
            views = torch.fft.irfft(z,n=self.seq_len,dim=-1,norm='ortho')
        return views.to(x.dtype)


class PromptRouter(nn.Module):
    def __init__(self,dim,depth,experts=8,dropout=0.):
        super().__init__()
        self.depth,self.dim = depth,dim
        self.pool = nn.Parameter(torch.randn(experts,dim)*.02)
        self.gate = nn.Linear(dim,experts)
        self.reparam = nn.Sequential(nn.Linear(dim,dim),nn.Tanh(),nn.Linear(dim,depth*2*dim),nn.Dropout(dropout))

    def forward(self,x):
        mixture = self.gate(x).softmax(-1) @ self.pool
        b,p,_ = x.shape
        return self.reparam(mixture).reshape(b,p,self.depth,2,self.dim).permute(2,3,0,1,4)


class PromptBlock(nn.Module):
    def __init__(self,dim,dropout,heads=4):
        super().__init__()
        self.heads,self.width,self.dropout = heads,dim//heads,dropout
        self.norm1,self.norm2 = RMSNorm(dim),RMSNorm(dim)
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.out = nn.Linear(dim,dim)
        self.ff = SwiGLU(dim,dim*4,dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self,x,prompt,valid):
        b,p,d = x.shape
        h = self.norm1(x)
        key = torch.cat((prompt[0],h),1)
        value = torch.cat((prompt[1],h),1)
        q = self.q(h).reshape(b,p,self.heads,self.width).transpose(1,2)
        k = self.k(key).reshape(b,2*p,self.heads,self.width).transpose(1,2)
        v = self.v(value).reshape(b,2*p,self.heads,self.width).transpose(1,2)
        allowed = torch.cat((valid,valid),-1)[:,None,None,:]
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.dropout if self.training else 0.)
        x = x + self.drop(self.out(y.transpose(1,2).reshape(b,p,d)))
        x = x + self.ff(self.norm2(x))
        return x * valid[...,None].to(x.dtype)


class SEMPO(nn.Module):
    def __init__(self,input_dim,seq_len,dim,depth,dropout):
        super().__init__()
        self.seq_len,self.patch,self.views = seq_len,min(4,seq_len),4
        self.npatch = math.ceil(seq_len/self.patch)
        self.spectral = SpectralViews(seq_len,self.views)
        self.embed = nn.Linear(self.patch,dim)
        self.position = nn.Parameter(torch.randn(1,self.npatch,dim)*.02)
        self.encoder_router = PromptRouter(dim,depth,dropout=dropout)
        self.encoder = nn.ModuleList([PromptBlock(dim,dropout) for _ in range(depth)])
        self.decoder_router = PromptRouter(dim,1,dropout=dropout)
        self.decoder = PromptBlock(dim,dropout)
        self.norm = RMSNorm(dim)
        self.readout = ChannelReadout(input_dim,dim,dropout)

    def forward(self,x,valid_time=None):
        x,valid = _prepare(x,valid_time,self.seq_len)
        b,t,c = x.shape
        views = self.spectral(x.transpose(1,2))  # B,C,views,T
        views = views * valid[:,None,None,:].to(views.dtype)
        pad = (-t)%self.patch
        chunks = F.pad(views,(0,pad)).unfold(-1,self.patch,self.patch)
        pm = F.pad(valid,(0,pad),value=False).unfold(-1,self.patch,self.patch).any(-1)
        vm = pm[:,None,None].expand(b,c,self.views,-1).reshape(b*c*self.views,-1)
        h = self.embed(chunks.reshape(b*c*self.views,self.npatch,self.patch)) + self.position
        prompts = self.encoder_router(h)
        for layer,block in enumerate(self.encoder):
            h = block(h,prompts[layer],vm)
        h = h.reshape(b*c,self.views,self.npatch,-1).mean(1)
        dm = pm[:,None].expand(b,c,-1).reshape(b*c,-1)
        h = self.decoder(h,self.decoder_router(h)[0],dm)
        h = _masked_mean(self.norm(h),dm).reshape(b,c,-1)
        return self.readout(h)


def build_backbone(name, input_dim, seq_len=20, dim=64, depth=2, dropout=.1):
    if dim < 8 or dim % 8:
        raise ValueError('Recent backbone dim must be a positive multiple of 8')
    if depth < 1 or seq_len < 1 or input_dim < 1:
        raise ValueError('depth, seq_len and input_dim must be positive')
    constructors = {'tirex':TiRex,'xlstmmixer':XLSTMMixer,'xlstmixer':XLSTMMixer,
                    'timefilter':TimeFilter,'timemoe':TimeMoE,'sempo':SEMPO}
    key = name.lower().replace('-','').replace('_','')
    if key not in constructors:
        raise KeyError(f'Unknown recent backbone: {name}')
    return constructors[key](input_dim,seq_len,dim,depth,dropout)
