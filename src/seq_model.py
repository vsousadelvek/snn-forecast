"""SNN temporal para previsao de trajetoria real.

Entrada: sequencia de centroides passados (B, K, 2) normalizados.
A dinamica do movimento fica codificada nas membranas LIF — a rede
le a sequencia timestep a timestep e regrediza posicoes futuras.
"""
import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_mem, v_th=1.0):
        ctx.save_for_backward(v_mem)
        ctx.v_th = v_th
        return (v_mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v_mem,) = ctx.saved_tensors
        x = torch.pi * (v_mem - ctx.v_th)
        return grad_out / (1.0 + x * x), None


class LIF(nn.Module):
    def __init__(self, beta=0.9, v_th=1.0):
        super().__init__()
        self.beta, self.v_th = beta, v_th

    def forward(self, x, mem):
        mem = self.beta * mem + x
        s = SpikeFn.apply(mem, self.v_th)
        return s, mem * (1.0 - s)


class SeqForecastSNN(nn.Module):
    """Le (dx,dy, x_abs,y_abs) por timestep; ultima membrana -> n_pred."""

    def __init__(self, n_pred=5, hidden=96):
        super().__init__()
        self.fc_in = nn.Linear(4, hidden)
        self.lif1 = LIF()
        self.fc_h = nn.Linear(hidden, hidden)
        self.lif2 = LIF()
        self.head = nn.Linear(hidden, 2 * n_pred)
        self.lif_out = LIF(v_th=1e9)

    def forward(self, seq):
        """seq: (B, K, 2)."""
        B, K, _ = seq.shape
        m1 = torch.zeros(B, self.fc_in.out_features, device=seq.device)
        m2 = torch.zeros(B, self.fc_h.out_features, device=seq.device)
        mO = torch.zeros(B, self.head.out_features, device=seq.device)
        acc = 0.0
        for k in range(K):
            s, m1 = self.lif1(self.fc_in(seq[:, k]), m1)
            s, m2 = self.lif2(self.fc_h(s), m2)
            _, mO = self.lif_out(self.head(s), mO)
            acc = acc + mO
        return acc / K
