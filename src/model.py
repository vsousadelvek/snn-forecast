"""SNN preditora: K event frames -> posicoes futuras do objeto.

Encoder conv-LIF sobre cada event frame; a membrana final da camada
de leitura alimenta uma head que regrediza M posicoes (x,y) futuras
normalizadas em [0,1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, beta=0.85, v_th=1.0):
        super().__init__()
        self.beta, self.v_th = beta, v_th

    def forward(self, x, mem):
        mem = self.beta * mem + x
        s = SpikeFn.apply(mem, self.v_th)
        return s, mem * (1.0 - s)


class ForecastSNN(nn.Module):
    """(B,K,2,64,64) -> (B, 2*n_pred) posicoes futuras."""

    def __init__(self, n_pred=5, steps_per_frame=2):
        super().__init__()
        self.steps_per_frame = steps_per_frame
        self.conv1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)   # 32
        self.lif1 = LIF()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 16
        self.lif2 = LIF(v_th=0.8)
        self.conv3 = nn.Conv2d(32, 48, 3, stride=2, padding=1)  # 8
        self.lif3 = LIF(v_th=0.6)
        self.fc1 = nn.Linear(48 * 8 * 8, 128)
        self.lif4 = LIF(v_th=0.5)
        self.head = nn.Linear(128, 2 * n_pred)
        self.lif5 = LIF(v_th=1e9)
        nn.init.normal_(self.fc1.weight, std=0.06)

    def forward(self, clip):
        B, K = clip.shape[:2]
        m1 = torch.zeros(B, 16, 32, 32, device=clip.device)
        m2 = torch.zeros(B, 32, 16, 16, device=clip.device)
        m3 = torch.zeros(B, 48, 8, 8, device=clip.device)
        m4 = torch.zeros(B, 128, device=clip.device)
        m5 = torch.zeros(B, self.head.out_features, device=clip.device)
        acc = 0.0
        for k in range(K):
            x = clip[:, k]
            for _ in range(self.steps_per_frame):
                s, m1 = self.lif1(self.conv1(x), m1)
                s, m2 = self.lif2(self.conv2(s), m2)
                s, m3 = self.lif3(self.conv3(s), m3)
                s4, m4 = self.lif4(self.fc1(s.flatten(1)), m4)
                _, m5 = self.lif5(self.head(s4), m5)
                acc = acc + m5
        return acc / (K * self.steps_per_frame)
