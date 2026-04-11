from tests.LDG.run import run
from tests.OneD_reaction.run import run
from src.lossFunctions.OneD_reactionPINNLoss import sourceTerm
import torch

if __name__ == "__main__":
    run()
    # x = torch.linspace(-1.0, 1.0, 1000, device="cuda").view(-1,1)
    # x.requires_grad = True
    # omega = 6
    # u = (torch.sin(omega*x))**3
    # u_x = torch.autograd.grad(u.view(-1,1), x, torch.ones((1000, 1), requires_grad = True).to("cuda"),allow_unused=True,create_graph=True)[0]

    # u_xx = torch.autograd.grad(u_x[:, 0].view(-1,1), x,torch.ones((1000, 1), requires_grad = True).to("cuda"), allow_unused=True, create_graph=True)[0][:,0]

    # new_PINNLoss = torch.mean((0.01*u_xx.view(-1,1) + 0.7*torch.tanh(u) - sourceTerm(x, omega= omega))**2)
    # print(new_PINNLoss)
    # print(torch.min(0.01*u_xx.view(-1,1) + 0.7*torch.tanh(u) - sourceTerm(x, omega= omega)))