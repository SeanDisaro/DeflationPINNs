import torch
import torch.nn as nn
import numpy as np

class HarmonicTrapezoidExtension(nn.Module):
    def __init__(self, d=0.06, num_terms=31):
        super().__init__()
        self.d = d
        
        # Only use ODD terms since even terms are 0 by symmetry
        # e.g., 1, 3, 5, 7 ... num_terms
        odds = np.arange(1, num_terms + 1, 2)
        self.n = torch.tensor(odds, dtype=torch.float32).unsqueeze(0) # Shape: (1, N)
        
        # Analytical Fourier coefficients for your T_d(t)
        n_pi = self.n * np.pi
        self.c_n = (4.0 * torch.sin(n_pi * self.d)) / (self.d * n_pi**2)

    def stable_sinh_ratio(self, n_pi, z):
        """ Computes sinh(n*pi*(1-z)) / sinh(n*pi) stably """
        exp_neg_nz = torch.exp(-n_pi * z)
        exp_neg_2n_1_z = torch.exp(-2.0 * n_pi * (1.0 - z))
        exp_neg_2n = torch.exp(-2.0 * n_pi)
        return exp_neg_nz * (1.0 - exp_neg_2n_1_z) / (1.0 - exp_neg_2n + 1e-12)

    def forward(self, x, y):
        n_pi = self.n * np.pi
        
        # --- Top & Bottom boundaries: T_d(x) ---
        sin_nx = torch.sin(x @ n_pi)
        # Sum of effects from y=0 and y=1
        y_decay = self.stable_sinh_ratio(n_pi, y) + self.stable_sinh_ratio(n_pi, 1.0 - y)
        F_y_boundaries = torch.sum(self.c_n * sin_nx * y_decay, dim=1, keepdim=True)
        
        # --- Left & Right boundaries: -T_d(y) ---
        sin_ny = torch.sin(y @ n_pi)
        # Sum of effects from x=0 and x=1
        x_decay = self.stable_sinh_ratio(n_pi, x) + self.stable_sinh_ratio(n_pi, 1.0 - x)
        F_x_boundaries = -torch.sum(self.c_n * sin_ny * x_decay, dim=1, keepdim=True)
        
        # Total F(x,y)
        Q_1 = F_y_boundaries + F_x_boundaries
        
        return Q_1.view(-1,1)