import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MollifiedTrapezoidExtension(nn.Module):
    def __init__(self, d=0.06, alpha=0.5):
        super().__init__()
        self.d = d
        self.alpha = alpha

    def smooth_relu(self, z, eps):
        # Added 1e-8 to prevent NaN in autograd at origin
        return 0.5 * (z + torch.sqrt(z**2 + eps**2 + 1e-8))

    def smooth_T_d(self, t, eps):
        """
        Algebraic formulation of your specific T_d(t) using SmoothReLU.
        T_d(t) = ( ReLU(t) - ReLU(t-d) - ReLU(t-(1-d)) + ReLU(t-1) ) / d
        """
        term1 = self.smooth_relu(t, eps)
        term2 = self.smooth_relu(t - self.d, eps)
        term3 = self.smooth_relu(t - (1.0 - self.d), eps)
        term4 = self.smooth_relu(t - 1.0, eps)
        
        return (term1 - term2 - term3 + term4) / self.d

    def forward(self, x, y):
        # eps(x,y) acts as the distance to the boundary
        eps = self.alpha * x * (1 - x) * y * (1 - y)
        
        # F(x,y) = T_d(x) - T_d(y)
        F_xy = self.smooth_T_d(x, eps) - self.smooth_T_d(y, eps)
        
        # Returns the vector field (Q_1, Q_2) where Q_2 is strictly 0
        Q_1 = F_xy
        
        return Q_1.view(-1,1)