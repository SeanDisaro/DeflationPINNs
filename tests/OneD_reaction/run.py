import torch
import deepxde as dde
from src.architectures.DeflationPINN import one_dim_DefPINN
from tests.OneD_reaction.training import train
from tests.OneD_reaction.testing import plotSolutions, plot_piml_Errors
import matplotlib.pyplot as plt
from src.starDomainExtrapolation.starDomain import *
from config import MY_DEVICE
import json

def cauchy_activation(x):
    return 1.0 / (x ** 2 + 1.0)


def run():
    """
    This funciton gets called in the main.py. Here, we define hyperparameters for a training process and start the training.
    """

    # n as in n grid
    n = 100
    # Generate linearly spaced points between 0 and 1
    points = torch.linspace(-1, 1, n)
    points = torch.cat((points, torch.linspace(-1,-0.95, 50)))
    points = torch.cat((points, torch.linspace(0.95,1, 50)))
    points = points.view(-1,1)


    points= points.to(MY_DEVICE)


    points.requires_grad = True
    
    #PINN
    omega = 6
    numSolutions = 11



    #loss
    deflationLossPoints = (1.,3.) 
    alpha = 1. 
    delta = 100. 
    epochs = 100_000
    learningRate = 1e-4

    # NN
    numBranchFeatures = 8
    trunk_layer = 1
    trunk_width = 2000
    activationFunction = cauchy_activation #torch.nn.Tanh(),
    DirichletHardConstraint = True
    skipConnection = False
    useSwiGLU=False
    fourierFeatures=False
    DirichletConstAt1 = -1.
    DirichletConstAt2= 1.
    DirichletConstValLeft = (np.sin(-1.*omega))**3
    DirichletConstValRight = (np.sin(1.*omega))**3

    #scheduler
    gamma = 0.8
    stepSize = 6000



    experimentProfile = {
        "n":n,
        "deflationLossPoints":deflationLossPoints,
        "alpha":alpha,
        "delta":delta,
        "epochs":epochs,
        "learningRate":learningRate,
        "numBranchFeatures":numBranchFeatures,
        "trunk_layer":trunk_layer,
        "trunk_width":trunk_width,
        "DirichletHardConstraint":DirichletHardConstraint,
        "omega":omega,
        "numSolutions":numSolutions
    }
    with open(f'./logs/experimentProfile_{MY_DEVICE}.json', 'w') as f:
        json.dump(experimentProfile, f)

    

    

    
    model = one_dim_DefPINN(
                    numSolutions = numSolutions,
                    numBranchFeatures = numBranchFeatures,
                    trunk_layer = trunk_layer,
                    trunk_width = trunk_width, 
                    activationFunction = activationFunction,
                    DirichletHardConstraint = DirichletHardConstraint,
                    skipConnection = skipConnection,
                    useSwiGLU=useSwiGLU,
                    fourierFeatures=fourierFeatures,
                    DirichletConstAt1 = DirichletConstAt1,
                    DirichletConstAt2= DirichletConstAt2,
                    DirichletConstValLeft = DirichletConstValLeft,
                    DirichletConstValRight = DirichletConstValRight
                    )
    
    model.to(MY_DEVICE)



    saveName = "steadyStateReactionDiffusion_" + str("zero") +"_"+ MY_DEVICE

    plotSolutions(model,  1000, saveName = saveName)



    savePIMLErrors = "steadyStateReactionDiffusion_PIML_Errors_" +   str("zero")+"_"+ MY_DEVICE
    plot_piml_Errors(model, 1000, showPlot = False, saveName= savePIMLErrors , omega= omega)

    model = train( model= model, x = points, epochs= epochs, boundaryPoints= None,
    learningRate = learningRate ,loadBestModel = True, showTrainingPlot = True, 
        alpha = alpha, delta = delta, deflationLossPoints=deflationLossPoints, schedulder_gamma=gamma, schedulder_StepSize=stepSize)


    saveName = "steadyStateReactionDiffusion_AfterTraining" +"_"+ MY_DEVICE
    plotSolutions(model,  1000, saveName= saveName)


    savePIMLErrors = "steadyStateReactionDiffusion_PIML_Errors_PIML_Errors_AfterTraining"+"_"+ MY_DEVICE
    plot_piml_Errors(model, 1000, showPlot = False, saveName= savePIMLErrors , omega= omega )

    return

