
import torch
import matplotlib.pyplot as plt
from pathlib import PurePath
from config import *
from src.lossFunctions.OneD_reactionPINNLoss import sourceTerm
import numpy as np
pathSavePictures = PurePath(plotFolder, "OneD_reaction")

def plotSolutions(model, grid_N, saveName = "reactionDiffPlot", showPlot = False, omega= 6):
    points = torch.linspace(-1.0, 1.0, grid_N, device=MY_DEVICE).view(-1,1)

    modelOut = model( points)

    u = modelOut['out']

    x_np = points.detach().cpu().numpy()
    n = len(u)
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)
    trueSol = np.sin(omega *x_np) **3
    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].plot(
                x_np,
                u[i].detach().cpu().numpy(), c = "red", 
                )
            ax[i%2, i//2].plot(
                x_np,
                trueSol, c = "green", 
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].plot(
                x_np,
                u[i].detach().cpu().numpy(),
                c = "red", 
                )
            ax[i].plot(
                x_np,
                trueSol, c = "green", 
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()



def plot_piml_Errors(model, grid_N, saveName = "reactionDiffPlot", showPlot = False, omega = 6):
    points = torch.linspace(-1.0, 1.0, grid_N, device=MY_DEVICE).view(-1,1)
    points.requires_grad = True
    modelOut = model( points)

    u = modelOut['out']
    n = len(u)
    u_xx_Arr = []
    tanh_Arr = []
    piml_errors = []
    batchSize = points.shape[0]
    for i in range(n):
        u_x = torch.autograd.grad(u[i].view(-1,1), points, torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),allow_unused=True,create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x[:, 0].view(-1,1), points,torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE), allow_unused=True, create_graph=True)[0][:,0]
        u_xx_Arr.append(0.01*u_xx.view(-1,1))
        tanh_Arr.append(0.7*torch.tanh(u[i]))
        new_PINNLoss = 0.01*u_xx.view(-1,1) + 0.7*torch.tanh(u[i]) - sourceTerm(points, omega= omega)
        print(f"Error Nr.{i+1}: {torch.mean(new_PINNLoss **2).item()}")
        piml_errors.append(new_PINNLoss)

    x_np = points.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].plot(
                x_np,                             
                piml_errors[i][:,0].detach().cpu().numpy(), c = "red", 
                )
            # ax[i%2, i//2].plot(
            #     x_np,                             
            #     u_xx_Arr[i][:,0].detach().cpu().numpy(), c = "blue", 
            #     )
            # ax[i%2, i//2].plot(
            #     x_np,                             
            #     tanh_Arr[i][:,0].detach().cpu().numpy(), c = "purple", 
            #     )
            # ax[i%2, i//2].plot(
            #     x_np,                             
            #     sourceTerm(points, omega= omega)[:,0].detach().cpu().numpy(), c = "brown", 
            #     )

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].plot(
                x_np,                             
                piml_errors[i][:,0].detach().cpu().numpy(), c = "red", 

                )
            # ax[i].plot(
            #     x_np,                             
            #     u_xx_Arr[i][:,0].detach().cpu().numpy(), c = "blue", 
            #     )
            # ax[i].plot(
            #     x_np,                             
            #     tanh_Arr[i][:,0].detach().cpu().numpy(), c = "purple", 
            #     )
            # ax[i].plot(
            #     x_np,                             
            #     sourceTerm(points, omega= omega)[:,0].detach().cpu().numpy(), c = "brown", 
            #     )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()