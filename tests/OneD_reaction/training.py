import torch
from torch.optim.lr_scheduler import StepLR
from src.lossFunctions.DeflationPINNLoss import defPINNLossPIML_w_AD_reaction
from src.architectures.DeflationPINN import one_dim_DefPINN
from typing import Tuple
import matplotlib.pyplot as plt
import random
from pathlib import Path, PurePath
from tqdm import tqdm
import logging
from config import *
import dill as pickle

pathSavePictures = PurePath(plotFolder, "OneD_reaction")

def train(  model: one_dim_DefPINN, x: torch.Tensor, epochs: int, boundaryPoints: torch.Tensor = None,
            learningRate:float  = 1e-4,loadBestModel:bool = False, showTrainingPlot:bool = True, modelName: str= "DeflationPINN_reaction",
            alpha:float = 1., beta:float = 0.1, delta:float = 1,omega =6,  deflationLossPoints: tuple[float,float] = (10000.,1.) , deflationCoefficient:float = 1., FrequencyReportLosses:int = 20, learningRateEpochPlotName:str = "Learning_Epoch_Plot")->one_dim_DefPINN:
    """
    This is the training funciton for the Deflation PINN model for the reaction diffusion eq. It returns the trained model and the feature list containing the solution functions, which can be used with the trained model.
    """    

    optimizer = torch.optim.AdamW(model.parameters(), lr = learningRate)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
    bestLoss = torch.inf
    
    useBoundaryLossTerm = False
    if boundaryPoints != None:
        useBoundaryLossTerm = True



    #plotting settings:
    if showTrainingPlot:
        plt.ion()
        fig, ax = plt.subplots()
        lossesForPlot = []
        xValueForPlot = [FrequencyReportLosses * i for i in range(epochs // FrequencyReportLosses)]
        ax.set_ylim(0, 10)
        ax.set_xlim(0, epochs)
 
        ax.set_title("Live Loss Plot Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        line, = ax.plot(xValueForPlot[:len(lossesForPlot)], lossesForPlot)

    #compute loss
    modelOut = model(x)
    modelOutBoundary = None
    if useBoundaryLossTerm:
        modelOutBoundary = model(boundaryPoints)
    loss = defPINNLossPIML_w_AD_reaction(  x=x, modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary ,
                     deflationLossPoints = deflationLossPoints , alpha = alpha , beta = beta, delta= delta, omega = omega)

    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()
        


        #compute loss
        modelOut = model(x)
        modelOutBoundary = None
        if useBoundaryLossTerm:
            modelOutBoundary = model(boundaryPoints)
        loss = defPINNLossPIML_w_AD_reaction(  x = x, modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary,
                                   deflationLossPoints = deflationLossPoints, alpha = alpha, beta = beta, delta = delta, omega= omega)


        #update best loss so far
        if loss.item() < bestLoss:
            bestLoss = loss.item()
            #save model; pathTrainedModels is defined in config
            if loadBestModel:
                #torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))
                with open(PurePath(pathTrainedModels, modelName +".pkl"), "wb") as f:
                    pickle.dump(model, f)


        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch%FrequencyReportLosses == 0:

            #plotting
            if showTrainingPlot:
                lossesForPlot.append(loss.item())
            
                ax.set_ylim(0, max(lossesForPlot))
                ax.set_xlim(0,epoch)
                line.set_ydata(lossesForPlot)
                line.set_xdata(xValueForPlot[:len(lossesForPlot)])

                # Redraw the plot
                fig.canvas.draw()
                fig.canvas.flush_events()


        epoch += 1

    # save learning rate/ epoch plot
    fig.savefig(PurePath(pathSavePictures, learningRateEpochPlotName + ".png"))

    if loadBestModel:
        #load best model; pathTrainedModels is defined in config
        #model = torch.load(PurePath(pathTrainedModels, modelName +".pt")).to("cuda")
        with open(PurePath(pathTrainedModels, modelName +".pkl"), "rb") as f:
            model = pickle.load(f).to("cuda:0")


    else:
        #torch.save(model, PurePath(pathTrainedModels, modelName +".pt"))
        with open(PurePath(pathTrainedModels, modelName +".pkl"), "wb") as f:
            pickle.dump(model, f)

    modelOut = model(x)
    modelOutBoundary = None
    if useBoundaryLossTerm:
        modelOutBoundary = model(boundaryPoints)
    # loss = defPINNLossPIML_w_AD_reaction(  x = x, modelOut = modelOut, boundaryPoints = boundaryPoints, modelOutBoundary = modelOutBoundary,
    #                                deflationLossPoints = deflationLossPoints, deflationLossCoeff=deflationCoefficient, alpha = alpha, beta = beta, delta = delta)


    return model