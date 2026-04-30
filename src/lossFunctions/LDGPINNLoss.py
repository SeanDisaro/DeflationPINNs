import torch
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from src.starDomainExtrapolation.starDomain import *
from config import MY_DEVICE


def pimlLoss_w_AutoGrad_LDG(modelOut:dict[list[torch.Tensor]],x: torch.Tensor, boundaryPoints:torch.Tensor = None, modelOutBoundary:dict[list[torch.Tensor]] = None ,
                eps:float = 0.02, alpha:float = 1., beta:float = 0.1):
    """Computes the PINN loss for the LDG problem from the paper. 

    Args:
        modelOut (dict[list[torch.Tensor]]): output of our DeflationPINN
        x (torch.Tensor): x
        boundaryPoints (torch.Tensor, optional): _description_. Defaults to None.
        modelOutBoundary (dict[list[torch.Tensor]], optional): _description_. Defaults to None.
        eps (float, optional): epsilon parameter from the LDG problem. Defaults to 0.02.
        alpha (float, optional): PINN scaling coefficient for PDE dynamics. Defaults to 1..
        beta (float, optional): PINN scaling coefficient for boundary condition. Defaults to 0.1.

    Returns:
        torch.Tensor: PINN loss
    """
    lossPDE = torch.tensor(0., device=MY_DEVICE)
    out1 = modelOut["out1"]
    out2 = modelOut["out2"]
    batchSize = x.shape[0]
    for i in range(len(out1)):
        out1_AD = torch.autograd.grad(out1[i].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0]
        
        out2_AD = torch.autograd.grad(out2[i].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0]
        
        out1_AD_dxx = torch.autograd.grad(out1_AD[:, 0].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out1_AD2_dyy = torch.autograd.grad(out1_AD[:, 1].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,1]
        
        out2_AD_dxx = torch.autograd.grad(out2_AD[:, 0].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out2_AD2_dyy = torch.autograd.grad(out2_AD[:, 1].view(-1,1), x,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,1]


        laplace_comp1 = out1_AD_dxx + out1_AD2_dyy
        laplace_comp2 = out2_AD_dxx + out2_AD2_dyy
        Q_squaredNorm = out1[i] **2 + out2[i] **2
        lossPDE_comp1 = ((2*   (1 - Q_squaredNorm) *out1[i]).view(-1)    + (eps **2)*laplace_comp1)
        lossPDE_comp2 = ((2*   (1 - Q_squaredNorm) *out2[i]).view(-1)    + (eps **2)*laplace_comp2)
        lossPDE = lossPDE +  torch.nanmean(torch.abs(lossPDE_comp1)) + torch.nanmean(torch.abs(lossPDE_comp2 ))

    if boundaryPoints != None and modelOutBoundary != None:
        out1B = modelOutBoundary["out1"]
        out2B = modelOutBoundary["out2"]
        outTrue1 = boundaryFunctionExtension(boundaryPoints, 3*eps)
        outTrue2 = torch.zeros((boundaryPoints.shape[0], 1))
        boundaryLoss = torch.tensor(0., device=MY_DEVICE)
        for i in range(len(out1B)):
            boundaryLoss = boundaryLoss +  torch.nanmean(torch.norm(out1B[i] - outTrue1 , dim = 1)) + torch.nanmean(torch.norm(out2B[i] - outTrue2, dim = 1))
        return  alpha* lossPDE + beta * boundaryLoss
    else:
        return alpha* lossPDE
    


def boundaryFunctionExtension(x: torch.Tensor, d:float)->torch.Tensor:
    """This function is where to define the extension of the trapezoidal boundary function from the paper.

    Args:
        x (torch.Tensor): Two dimensional input data vecotrs.
        d (float): parameter for the funciton.

    Returns:
        torch.Tensor: funciton value of extension.
    """    
    x1 = x[:, 0]
    x2 = x[:, 1]
    #this mask stuff is to make sure, that we do not devide by 0.
    maskZeroDenominator = (x1* (1-x1) == 0) * (x2* (1-x2) == 0)
    maskZeroDenominatorNegation = maskZeroDenominator == False
    out = trapezoidalFun(x1, d) - trapezoidalFun(x2, d)
    out = out * maskZeroDenominatorNegation

    
    return out


def trapezoidalFun(x: torch.Tensor, d:float)->torch.Tensor:
    """This is the funciton T_d(x) from the paper.

    Args:
        x (torch.Tensor): one dimensional x values.
        d (float): parameter for the function.

    Returns:
        torch.Tensor: T_d(x) as in paper.
    """
    out = torch.zeros_like(x,dtype=float)
    maskInterval_0_to_d       = (x >=0)   * (x<=d)
    maskInterval_d_to_1Minusd = (x >d)      * (x<(1-d))
    maskInterval_1Minusd_to_1 = (x >=(1-d)) * (x<=1)
    out = maskInterval_0_to_d       * (x / d)
    out = out + maskInterval_d_to_1Minusd * 1
    out = out + maskInterval_1Minusd_to_1 * ((1-x)/d)
    return out





def boundaryConditionSpherical(starDomain, angles, boundaryFunction):
    return boundaryFunction(starDomain.getCartesianCoordinates( starDomain.radiusDomainFunciton(angles) ,angles))
  
def zeroOnBoundaryExtension(starDomain, input):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return linearRadial( radius / starDomain.radiusDomainFunciton(angles)).view(-1,1)
  
def DCBoundaryExtension(starDomain,input, boundaryFunction):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return boundaryConditionSpherical(starDomain, angles, boundaryFunction).view(-1,1)  *  (1- linearRadial( radius / starDomain.radiusDomainFunciton(angles))).view(-1,1) 




def linearRadial(x):
    return (1 - x).view(-1,1)




def squaredRadial(x):
    return (1 - x*x).view(-1,1)

def cInfRadial(x):
    is_zero = (x == 0)
    
    x_safe = torch.where(is_zero, torch.ones_like(x), x)

    ln2 = torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    computed_val = 1 - 2 * torch.exp(-ln2 / x_safe)
    
    # 4. Final Merge: Return 1.0 where x was 0, otherwise return the computed value
    return torch.where(is_zero, torch.ones_like(computed_val), computed_val)

def smooth_transition(x):
    mask = (x > 0) & (x < 1)
    
    x_safe = torch.where(mask, x, torch.tensor(0.5, device=x.device, dtype=x.dtype))
    
    u = (2 * x_safe - 1) / (x_safe * (1 - x_safe))

    result = torch.sigmoid(-u)
    

    result = torch.where(x <= 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), result)
    result = torch.where(x >= 1, torch.tensor(0.0, device=x.device, dtype=x.dtype), result)
    return result



def bfunc(x):
    return boundaryFunctionExtension(torch.cat((x[0],x[1]), dim=1), d= 0.06)