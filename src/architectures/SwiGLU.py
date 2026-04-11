from torch import nn

import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        # Two parallel projections for the gated path
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # "gate" projection
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # "value" projection
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)  # down projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))