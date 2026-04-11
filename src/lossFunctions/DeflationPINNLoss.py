import torch
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from src.lossFunctions.LDGPINNLoss import pimlLoss_w_AutoGrad_LDG
from src.lossFunctions.DeflationLoss import linearDeflationLoss_dictModel
from src.lossFunctions.OneD_reactionPINNLoss import pimlLoss_w_AutoGrad_reaction

def defPINNLossPIML_w_AD_LDG( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None,modelOutBoundary:list[list[torch.Tensor]] = None ,
                    eps:float = 0.02, deflationLossPoints: tuple[float,float] = (10000.,1.), alpha:float = 1., beta:float = 0.1, delta: float = 1.) -> torch.Tensor:

    loss_PDEAndBoundary = pimlLoss_w_AutoGrad_LDG(  modelOut = modelOut,x=x, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    eps = eps, alpha = alpha, beta = beta)

    deflationLossOut = linearDeflationLoss_dictModel(modelOut = modelOut, maxLoss = deflationLossPoints[0], maxDistance = deflationLossPoints[1], keyOutModel="out2")

    return loss_PDEAndBoundary + delta*deflationLossOut




def defPINNLossPIML_w_AD_reaction( x: torch.Tensor, modelOut:list[list[torch.Tensor]], boundaryPoints:torch.Tensor = None, modelOutBoundary:list[list[torch.Tensor]] = None ,
                     deflationLossPoints: tuple[float,float] = (10000.,1.) , alpha:float = 1., beta:float = 0.1, delta: float = 1., omega = 6) -> torch.Tensor:

    loss_PDEAndBoundary = pimlLoss_w_AutoGrad_reaction(  modelOut = modelOut,x=x, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                                    alpha = alpha, beta = beta, omega = omega)

    deflationLossOut = linearDeflationLoss_dictModel(modelOut = modelOut, maxLoss = deflationLossPoints[0], maxDistance = deflationLossPoints[1], keyOutModel="out") 

    return loss_PDEAndBoundary + delta*deflationLossOut