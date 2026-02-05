import matplotlib.pyplot as plt
from config import *
import torch
from pathlib import PurePath

pathSavePictures = PurePath(plotFolder, "deflationPINNTest")


def plot_Q11_Q12(model, grid_N, showPlot = False, saveName:str = "Reduced2DimLDG_Results" ):
    
    xs = torch.linspace(0.0, 1.0, grid_N, device="cpu")
    ys = torch.linspace(0.0, 1.0, grid_N, device="cpu")

    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    jointInputVec = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)) , dim = 1).to("cuda")

    modelOut = model( jointInputVec)

    U = modelOut['out1']
    V = modelOut['out2']


    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    n = len(U)
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].quiver(
                X, Y,                             # arrow tails
                U[i].detach().cpu().numpy(),
                V[i].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].quiver(
                X, Y,                             # arrow tails
                U[i].detach().cpu().numpy(),
                V[i].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()




def plot_nematic_director(model, grid_N, showPlot = False, saveName:str = "Reduced2DimLDG_nematic_director" ):
    xs = torch.linspace(0.0, 1.0, grid_N, device="cpu")
    ys = torch.linspace(0.0, 1.0, grid_N, device="cpu")

    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    jointInputVec = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)) , dim = 1).to("cuda")

    modelOut = model( jointInputVec)

    Q11 = modelOut['out1']
    Q12 = modelOut['out2']
    n = len(Q11)

    theta = []
    nematic_director = []
    for i in range(n):
        thetaAux = torch.arctan( Q12[i]/Q11[i] + (Q11[i] == 0).float() ) * (Q11[i] != 0).float()
        theta.append(thetaAux)
        nematic_director1 = torch.cos(thetaAux)
        nematic_director2 = torch.sin(thetaAux)
        nematic_director.append( torch.cat((nematic_director1,nematic_director2) , dim = 1) )

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].quiver(
                X, Y,                             # arrow tails
                nematic_director[i][:, 0].detach().cpu().numpy(),
                nematic_director[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].quiver(
                X, Y,                             # arrow tails
                nematic_director[i][:, 0].detach().cpu().numpy(),
                nematic_director[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()




def plot_piml_Error(model, grid_N, showPlot = False, saveName:str = "Reduced2DimLDG_PIML_Errors" ):
    xs = torch.linspace(0.0001, 0.9999, grid_N, device="cpu")
    ys = torch.linspace(0.001, 0.9999, grid_N, device="cpu")
    eps = 0.02
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    jointInputVec = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)) , dim = 1).to("cuda")
    jointInputVec.requires_grad = True
    modelOut = model( jointInputVec)

    Q11 = modelOut['out1']
    Q12 = modelOut['out2']
    n = len(Q11)

    piml_errors = []
    batchSize = jointInputVec.shape[0]
    for i in range(n):
        out1_AD = torch.autograd.grad(Q11[i].view(-1,1), jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
        
        out2_AD = torch.autograd.grad(Q12[i].view(-1,1),jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0]
        
        out1_AD_dxx = torch.autograd.grad(out1_AD[:, 0].view(-1,1), jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out1_AD2_dyy = torch.autograd.grad(out1_AD[:, 1].view(-1,1), jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,1]
        
        out2_AD_dxx = torch.autograd.grad(out2_AD[:, 0].view(-1,1), jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,0]
        
        out2_AD2_dyy = torch.autograd.grad(out2_AD[:, 1].view(-1,1), jointInputVec,
                                     torch.ones((batchSize, 1), requires_grad = True).to("cuda"),
                                       allow_unused=True, create_graph=True)[0][:,1]

        laplace_comp1 = out1_AD_dxx + out1_AD2_dyy
        laplace_comp2 = out2_AD_dxx + out2_AD2_dyy
        Q_squaredNorm = Q11[i] **2 + Q12[i] **2
        lossPDE_comp1 = (200*   (1 - Q_squaredNorm) *Q11[i]    + ((10*eps) **2)*laplace_comp1)*0.01
        lossPDE_comp2 = (200*   (1 - Q_squaredNorm) *Q12[i]    + ((10*eps) **2)*laplace_comp2)*0.01

        piml_errors.append( torch.cat((lossPDE_comp1,lossPDE_comp2) , dim = 1) )

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    fig, ax = plt.subplots(2,n//2 + n%2)
    fig.set_figheight(30)
    fig.set_figwidth(30)

    path = PurePath(pathSavePictures, saveName + ".png")
    if n>2:
        for i in range(n):
            ax[i%2, i//2].set_title(f'Plot {i}')
            ax[i%2, i//2].quiver(
                X, Y,                             # arrow tails
                piml_errors[i][:, 0].detach().cpu().numpy(),
                piml_errors[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        

    else:
        for i in range(n):
            ax[i].set_title(f'Plot {i}')
            ax[i].quiver(
                X, Y,                             # arrow tails
                piml_errors[i][:, 0].detach().cpu().numpy(),
                piml_errors[i][:, 1].detach().cpu().numpy(),      # arrow directions
                angles='xy', scale_units='xy', scale=20, width=0.0015
                )
        
    fig.savefig(path)
    if showPlot:
        fig.show()
    plt.close()



