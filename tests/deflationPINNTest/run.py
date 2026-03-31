import torch
import deepxde as dde
from src.architectures.DeflationPINN import two_dim_DefPINN
from tests.deflationPINNTest.training import train
from src.lossFunctions.LDGPINNLoss import bfunc, DCBoundaryExtension
from src.harmonicTrapezoidalExtension import HarmonicTrapezoidExtension
from src.molifiedTrapezoidExtension import MollifiedTrapezoidExtension
from tests.deflationPINNTest.testing import plot_Q11_Q12, plot_nematic_director, plot_piml_Error
import matplotlib.pyplot as plt
from src.starDomainExtrapolation.starDomain import *




def run():
    """
    This funciton gets called in the main.py. Here, we define hyperparameters for a training process and start the training.
    """
    geom = dde.geometry.geometry_2d.Rectangle([0.,0.], [1.,1.])

    squareAsStarDom = HyperCuboid(2, torch.tensor([0.5,0.5]), torch.tensor([1.,1.]))
    squareAsStarDom.updateDevice("cuda")

    # n as in n x n grid
    n = 33
    safetySpace = 0.001
    # Generate linearly spaced points between 0 and 1
    x = torch.linspace(0+safetySpace, 1-safetySpace, n) 
    y = torch.linspace(0+safetySpace, 1-safetySpace, n)

    # Create a meshgrid
    yy, xx = torch.meshgrid(y, x, indexing='ij')  

    # Stack and reshape to get [n, 2] shape
    points = torch.stack([xx, yy], dim=-1).reshape(-1, 2)





    points= points.to("cuda")


    points.requires_grad = True
    numSolutions = 6
    learningRate = 1e-4 




    deflationLossPoints = (1.,0.4) 

    alpha = 0.01 
    delta = 100. 

    epochs = 10000


    
    model = two_dim_DefPINN(
                    numSolutions = numSolutions,
                    numBranchFeatures = 16, 
                    trunk_layer = 1,
                    trunk_width = 4000, 
                    activationFunction = torch.nn.Tanh(),
                    geom = geom,
                    DirichletHardConstraint = True,
                    skipConnection = False,
                    DirichletConditionFunc1 = lambda input:  DCBoundaryExtension(squareAsStarDom,[input[:,0].view(-1,1), input[:,1].view(-1,1)], bfunc), 
                    DirichletConditionFunc2 = lambda x: torch.zeros((x.shape[0],1),device="cuda:0")
                    )

	

    saveNameQ11Q12 = "Reduced2DimLDG_Results_" + str("zero")
    plot_Q11_Q12(model,  33, saveName= saveNameQ11Q12)

    saveNamenematic_director = "Reduced2DimLDG_nematic_director_" +  str("zero")
    plot_nematic_director(model, 33, saveName= saveNamenematic_director)

    savePIMLErrors = "Reduced2DimLDG_PIML_Errors_" +   str("zero")
    plot_piml_Error(model, 33, showPlot = False, saveName= savePIMLErrors )

    model= train( model= model, x = points, epochs= epochs, boundaryPoints= None,
    learningRate = learningRate ,loadBestModel = True, showTrainingPlot = True, 
        alpha = alpha,delta = delta, deflationLossPoints=deflationLossPoints)


    saveNameQ11Q12 = "Reduced2DimLDG_Results_AfterTraining" 
    plot_Q11_Q12(model,  40, saveName= saveNameQ11Q12)

    saveNamenematic_director = "Reduced2DimLDG_nematic_director_AfterTraining" 
    plot_nematic_director(model, 40, saveName= saveNamenematic_director)

    savePIMLErrors = "Reduced2DimLDG_PIML_Errors_AfterTraining"
    plot_piml_Error(model, 40, showPlot = False, saveName= savePIMLErrors )


    return

