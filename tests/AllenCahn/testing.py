
import torch
import matplotlib.pyplot as plt
from pathlib import PurePath
from config import *
from src.lossFunctions.OneD_reactionPINNLoss import sourceTerm
import numpy as np
from scipy.stats import binned_statistic_2d

pathSavePictures = PurePath(plotFolder, "AllenCahn")

def plotSolutions(model, grid_N, saveName = "allenCahnPlot", showPlot = False):
    R = 1.0
    n = grid_N 

    coords = torch.linspace(-R, R, n)
    xx, yy = torch.meshgrid(coords, coords, indexing='xy')

    # Keep only points inside the disk
    mask = xx**2 + yy**2 < R**2
    points = torch.stack([xx[mask], yy[mask]], dim=-1)
    points = points.to(MY_DEVICE)


    modelOut = model( points)

    X, Y = points[:,0].detach().cpu().numpy() , points[:,1].detach().cpu().numpy()
    u = modelOut['out']

    n = len(u)
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)
    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                                        X, Y, u[i].detach().cpu().numpy()[:,0], statistic='mean', bins=20
                                        )
            ax[i%2, i//2].set_title(f'Plot {i}')
            im = ax[i%2, i//2].imshow(
                                stat.T,                              # transpose so x is horizontal
                                origin='lower',
                                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                                aspect='auto',
                                cmap='viridis',
                                interpolation='bilinear',
                            )
            fig.colorbar(im, ax=ax[i%2, i//2])
        

    else:
        for i in range(n):
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                                        X, Y, u[i].detach().cpu().numpy()[:,0], statistic='mean', bins=20
                                        )
            ax[i].set_title(f'Plot {i}')
            im = ax[i].imshow(
                            stat.T,                              # transpose so x is horizontal
                            origin='lower',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            aspect='auto',
                            cmap='viridis',
                            interpolation='bilinear',
                        )
            fig.colorbar(im, ax=ax[i])    

        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()



def plot_piml_Errors(model, grid_N, saveName = "allenCahnDiffPlot", showPlot = False, omega = 6):
    R = 1.0
    n = grid_N 

    coords = torch.linspace(-R, R, n)
    xx, yy = torch.meshgrid(coords, coords, indexing='xy')

    # Keep only points inside the disk
    mask = xx**2 + yy**2 < R**2
    points = torch.stack([xx[mask], yy[mask]], dim=-1)
    points = points.to(MY_DEVICE)
    points.requires_grad = True

    modelOut = model( points)

    u = modelOut['out']
    n = len(u)

    piml_errors = []
    batchSize = points.shape[0]
    for i in range(n):
        out_AD = torch.autograd.grad(u[i].view(-1,1), points,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0]
        
        out_AD_dxx = torch.autograd.grad(out_AD[:, 0].view(-1,1), points,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out_AD2_dyy = torch.autograd.grad(out_AD[:, 1].view(-1,1), points,
                                     torch.ones((batchSize, 1), requires_grad = True, device=MY_DEVICE),
                                       allow_unused=True, create_graph=True)[0][:,1]
        

        laplace = out_AD_dxx + out_AD2_dyy
        new_PINNLoss = laplace + 6*u[i] - u[i]**3
        print(f"Error Nr.{i+1}: {torch.nanmean(new_PINNLoss **2).item()}")
        piml_errors.append(new_PINNLoss)

    X, Y = points[:,0].detach().cpu().numpy() , points[:,1].detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                                        X, Y, piml_errors[i][:,0].detach().cpu().numpy(), statistic='mean', bins=20
                                        )
            ax[i%2, i//2].set_title(f'Plot {i}')
            im = ax[i%2, i//2].imshow(
                            stat.T,                              # transpose so x is horizontal
                            origin='lower',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            aspect='auto',
                            cmap='viridis',
                            interpolation='bilinear',
                        )
            fig.colorbar(im, ax=ax[i%2, i//2])

    else:
        for i in range(n):
            stat, x_edges, y_edges, _ = binned_statistic_2d(
                                        X, Y, piml_errors[i][:,0].detach().cpu().numpy(), statistic='mean', bins=20
                                        )
            ax[i].set_title(f'Plot {i}')
            im = ax[i].imshow(
                            stat.T,                              # transpose so x is horizontal
                            origin='lower',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            aspect='auto',
                            cmap='viridis',
                            interpolation='bilinear',
                        )
            fig.colorbar(im, ax=ax[i])

        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()