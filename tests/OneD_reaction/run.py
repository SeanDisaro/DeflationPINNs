import torch
import deepxde as dde
from src.architectures.DeflationPINN import one_dim_DefPINN
from tests.OneD_reaction.training import train
from tests.OneD_reaction.testing import plotSolutions, plot_piml_Errors
import matplotlib.pyplot as plt
from src.starDomainExtrapolation.starDomain import *




def run():
    """
    This funciton gets called in the main.py. Here, we define hyperparameters for a training process and start the training.
    """

    # n as in n grid
    n = 1000
    # Generate linearly spaced points between 0 and 1
    points = torch.linspace(-1, 1, n)
    # points = torch.cat((points, torch.linspace(-1,-0.9, 50)))
    # points = torch.cat((points, torch.linspace(0.9,1, 50)))
    points = points.view(-1,1)


    points= points.to("cuda")


    points.requires_grad = True
    numSolutions = 11
    learningRate = 1e-2




    deflationLossPoints = (1.,0.5) 

    alpha = 1. 
    delta = 100. 
    omega = 6
    epochs = 10000


    
    model = one_dim_DefPINN(
                    numSolutions = numSolutions,
                    numBranchFeatures = 20,
                    trunk_layer = 1,
                    trunk_width = 1000, 
                    activationFunction = torch.nn.Tanh(),
                    DirichletHardConstraint = True,
                    skipConnection = False,
                    useSwiGLU=True,
                    fourierFeatures=True,
                    DirichletConstAt1 = -1.,
                    DirichletConstAt2= 1.,
                    DirichletConstValLeft = (np.sin(-1.*omega))**3,
                    DirichletConstValRight = (np.sin(1.*omega))**3
                    )

    # print(model.parameters)


    saveName = "steadyStateReactionDiffusion_" + str("zero")

    plotSolutions(model,  1000, saveName = saveName)



    savePIMLErrors = "steadyStateReactionDiffusion_PIML_Errors_" +   str("zero")
    plot_piml_Errors(model, 1000, showPlot = False, saveName= savePIMLErrors , omega= omega)

    model = train( model= model, x = points, epochs= epochs, boundaryPoints= None,
    learningRate = learningRate ,loadBestModel = True, showTrainingPlot = True, 
        alpha = alpha, delta = delta, deflationLossPoints=deflationLossPoints)


    saveName = "steadyStateReactionDiffusion_AfterTraining" 
    plotSolutions(model,  1000, saveName= saveName)


    savePIMLErrors = "steadyStateReactionDiffusion_PIML_Errors_PIML_Errors_AfterTraining"
    plot_piml_Errors(model, 1000, showPlot = False, saveName= savePIMLErrors , omega= omega )

    return

