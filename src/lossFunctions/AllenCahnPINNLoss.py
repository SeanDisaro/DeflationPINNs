import torch
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from src.starDomainExtrapolation.starDomain import *
from config import MY_DEVICE


def pimlLoss_w_AutoGrad_allenCahn(modelOut:dict[list[torch.Tensor]],x: torch.Tensor, boundaryPoints:torch.Tensor = None, modelOutBoundary:dict[list[torch.Tensor]] = None ,
                omega:float = 6., alpha:float = 1., beta:float = 0.1):
    """Computes the PINN loss for the Allen Cahn problem. 

    Args:
        modelOut (dict[list[torch.Tensor]]): output of our DeflationPINN
        x (torch.Tensor): x
        boundaryPoints (torch.Tensor, optional): _description_. Defaults to None.
        modelOutBoundary (dict[list[torch.Tensor]], optional): _description_. Defaults to None.
        omega (float, optional): parameter from the allen cahn equation which regulates how many solutions. Defaults to 6.
        alpha (float, optional): PINN scaling coefficient for PDE dynamics. Defaults to 1..
        beta (float, optional): PINN scaling coefficient for boundary condition. Defaults to 0.1.

    Returns:
        torch.Tensor: PINN loss
    """
    lossPDE = torch.tensor(0., device=MY_DEVICE)
    out = modelOut["out"]

    batchSize = x.shape[0]
    for i in range(len(out)):
        out_AD = torch.autograd.grad(out[i].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0]
        
        out_AD_dxx = torch.autograd.grad(out_AD[:, 0].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out_AD2_dyy = torch.autograd.grad(out_AD[:, 1].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,1]
        

        laplace = out_AD_dxx + out_AD2_dyy
        residual = laplace.view(-1,1) + 6*out[i] - out[i]**3
        lossPDE = lossPDE + torch.mean(residual**8)

    if boundaryPoints != None and modelOutBoundary != None:
        outB = modelOutBoundary["out"]
        boundaryLoss = torch.tensor(0., device=MY_DEVICE)
        for i in range(len(outB)):
            boundaryLoss = boundaryLoss +  torch.mean(torch.norm(outB[i] , dim = 1))
        return  alpha* lossPDE + beta * boundaryLoss
    else:
        return alpha* lossPDE
    
