import torch
import matplotlib.pyplot as plt
import numpy as np
from src.starDomainExtrapolation.starDomain import *


def pimlLoss_w_AutoGrad_reaction(modelOut:dict[list[torch.Tensor]],x: torch.Tensor, boundaryPoints:torch.Tensor = None, modelOutBoundary:dict[torch.Tensor] = None ,
                 alpha:float = 1., beta:float = 0.1, omega = 6):
    """Computes the PINN loss for the 1dim steady state reaction diffusion problem from the paper. 

    Args:
        modelOut (dict[list[torch.Tensor]]): output of our DeflationPINN
        x (torch.Tensor): x
        boundaryPoints (torch.Tensor, optional): _description_. Defaults to None.
        modelOutBoundary (dict[list[torch.Tensor]], optional): _description_. Defaults to None.
        alpha (float, optional): PINN scaling coefficient for PDE dynamics. Defaults to 1..
        beta (float, optional): PINN scaling coefficient for boundary condition. Defaults to 0.1.

    Returns:
        torch.Tensor: PINN loss
    """
    lossPDE = torch.tensor(0.)
    u = modelOut["out"]
    batchSize = x.shape[0]
    for i in range(len(u)):
        u_x = torch.autograd.grad(u[i].view(-1,1), x, torch.ones((batchSize, 1), requires_grad = True).to("cuda"),allow_unused=True,create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x[:, 0].view(-1,1), x,torch.ones((batchSize, 1), requires_grad = True).to("cuda"), allow_unused=True, create_graph=True)[0][:,0]

        new_PINNLoss = 0.01*u_xx.view(-1,1) + 0.7*torch.tanh(u[i]) - sourceTerm(x, omega= omega)

        lossPDE = lossPDE +  torch.nanmean(new_PINNLoss **2)

    if boundaryPoints != None and modelOutBoundary != None:
        # 0 on boundary points
        uB = modelOutBoundary["out"]

        boundaryLoss = torch.tensor(0.)
        for i in range(len(uB)):
            boundaryLoss = boundaryLoss +  torch.nanmean(torch.norm(uB[i] , dim = 1) **2)
        return  alpha* lossPDE + beta * boundaryLoss
    else:
        return alpha* lossPDE
    
def sourceTerm(x:torch.Tensor, D = 0.01, kappa = 0.7, omega = 6):
    u = (torch.sin(omega*x))**3
    u_xx = 3*(omega**2) *torch.sin(omega*x) *(2*(torch.cos(omega * x)**2) - (torch.sin(omega * x) **2))
    return D*u_xx + kappa*torch.tanh(u)

