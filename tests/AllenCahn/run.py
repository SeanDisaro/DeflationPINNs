import torch
import deepxde as dde
from src.architectures.DeflationPINN import two_dim_2_one_dim_DefPINN
from tests.AllenCahn.training import train
from tests.AllenCahn.testing import plotSolutions, plot_piml_Errors
from src.starDomainExtrapolation.starDomain import *
from config import MY_DEVICE
import json


def run():
    """
    This funciton gets called in the main.py. Here, we define hyperparameters for a training process and start the training.
    """

    # n as in n grid
    R = 1.0
    n = 33  # resolution along each axis

    # Build a square grid covering [-R, R] x [-R, R]
    coords = torch.linspace(-R, R, n)
    xx, yy = torch.meshgrid(coords, coords, indexing='xy')


    # Keep only points inside the disk and remove (0,0) because of radial construction
    r2 = xx**2 + yy**2
    mask = (r2 < R**2) & (r2 > 1e-8)
    
    points = torch.stack([xx[mask], yy[mask]], dim=-1)
    

    diskAsStarDom = Sphere(dim = 2 , center =torch.tensor( [0,0]), radius= torch.tensor([1.]))
    diskAsStarDom.updateDevice(MY_DEVICE)

    points= points.to(MY_DEVICE)


    points.requires_grad = True
    




    #loss
    deflationLossPoints = (1.,0.4)
    alpha = 100
    delta = 1 
    epochs = 40_000
    learningRate = 1e-4

    # NN
    numBranchFeatures = 32
    trunk_layer = 6
    trunk_width = 100
    activationFunction = torch.nn.Tanh()


    #scheduler
    gamma = 0.8
    stepSize = 2000

    #PINN
    omega = 6
    numSolutions = 3

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
        "omega":omega,
        "numSolutions":numSolutions
    }
    with open(f'./logs/AllenCahnexperimentProfile_{MY_DEVICE}.json', 'w') as f:
        json.dump(experimentProfile, f)

    

    

    
    model = two_dim_2_one_dim_DefPINN(
                    numSolutions = numSolutions,
                    numBranchFeatures=numBranchFeatures,
                    trunk_layer=trunk_layer,
                    trunk_width=trunk_width,
                    activationFunction = activationFunction,
                    geom = None,
                    DirichletHardConstraint = True,
                    skipConnection = False,
                    r_function= lambda x: zeroOnBoundaryExtension(diskAsStarDom, [x[:,0].view(-1,1), x[:,1].view(-1,1)]),
                    DirichletConditionFunc= None,
                    )
    
    model.to(MY_DEVICE)



    saveName = "AllenCahn_" + str("zero") +"_"+ MY_DEVICE.replace(":", "_")

    plotSolutions(model,  33, saveName = saveName)



    savePIMLErrors = "AllenCahn_PIML_Errors_" +   str("zero")+"_"+ MY_DEVICE.replace(":", "_")
    plot_piml_Errors(model, 33, showPlot = False, saveName= savePIMLErrors , omega= omega)

    model = train( model= model, x = points, epochs= epochs, boundaryPoints= None,
    learningRate = learningRate ,loadBestModel = True, showTrainingPlot = True, 
        alpha = alpha, delta = delta, deflationLossPoints=deflationLossPoints, schedulder_gamma=gamma, schedulder_StepSize=stepSize)


    saveName = "AllenCahn_AfterTraining" +"_"+ MY_DEVICE.replace(":", "_")
    plotSolutions(model,  33, saveName= saveName)


    savePIMLErrors = "AllenCahn_PIML_Errors_PIML_Errors_AfterTraining"+"_"+ MY_DEVICE.replace(":", "_")
    plot_piml_Errors(model, 33, showPlot = False, saveName= savePIMLErrors , omega= omega )

    return







def boundaryConditionSpherical(starDomain, angles, boundaryFunction):
    return boundaryFunction(starDomain.getCartesianCoordinates( starDomain.radiusDomainFunciton(angles) ,angles))
  
def zeroOnBoundaryExtension(starDomain, input):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return squaredRadial( radius / starDomain.radiusDomainFunciton(angles)).view(-1,1)
  
def DCBoundaryExtension(starDomain,input, boundaryFunction):
    radius, angles = starDomain.getSphericalCoordinates(input)
    return boundaryConditionSpherical(starDomain, angles, boundaryFunction).view(-1,1)  *  (1- squaredRadial( radius / starDomain.radiusDomainFunciton(angles))).view(-1,1) 



def squaredRadial(x):
    return (1 - x*x).view(-1,1)
