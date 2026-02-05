import torch
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from src.lossFunctions.LDGPINNLoss import pimlLoss_w_AutoGrad
from src.lossFunctions.DeflationLoss import linearDeflationLoss_dictModel

def defPINNLossPIML_w_AD( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                    eps:float = 0.02, deflationLossPoints: tuple[float,float] = (10000.,1.) ,deflationLossCoeff:float = 1., alpha:float = 1., beta:float = 0.1, delta: float = 1.) -> torch.Tensor:

    loss_PDEAndBoundary = pimlLoss_w_AutoGrad(  modelOut = modelOut,x=x, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    eps = eps, alpha = alpha, beta = beta)

    deflationLossOut = linearDeflationLoss_dictModel(modelOut = modelOut, maxLoss = deflationLossPoints[0], maxDistance = deflationLossPoints[1]) #deflationLoss(modelOut=modelOut, a = deflationLossCoeff)

    return loss_PDEAndBoundary + delta*deflationLossOut