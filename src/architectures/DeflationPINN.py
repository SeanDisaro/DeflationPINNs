from torch import nn
import deepxde as dde
from typing import Callable, Tuple
import torch



class two_dim_DefPINN(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the out put
    is 2 and it outputs on top all the derivatives of the two outputs up to order 2 which we need for the laplacians of the components.
    Furthermore, we output the branch features, since we want to reuse those for autoregression.
    Thus the total number of outputs of this model is 11.
    """
    def __init__(   self,
                    numSolutions: int,
                    numBranchFeatures: int,
                    trunk_layer: int,
                    trunk_width: int,
                    activationFunction: Callable[[torch.Tensor], torch.Tensor],
                    geom: dde.geometry.geometry.Geometry,
                    DirichletHardConstraint: bool,
                    skipConnection: bool = False,
                    DirichletConditionFunc1: Callable[[torch.Tensor], torch.Tensor] = None,
                    DirichletConditionFunc2: Callable[[torch.Tensor], torch.Tensor] = None
                    ):
        super().__init__()
        self.activationFunction = activationFunction
        self.numBranchFeatures = numBranchFeatures
        self.geom = geom
        self.DirichletHardConstraint = DirichletHardConstraint
        self.skipConnection = skipConnection
        self.numSolutions = numSolutions
        self.DirichletConditionFunc1 = DirichletConditionFunc1
        self.DirichletConditionFunc2 = DirichletConditionFunc2
        self.trunk_layer = trunk_layer




        self.trunkNet_Lin        =  (     
                                      [nn.Linear(2,trunk_width)]
                                    + [nn.Linear(trunk_width, trunk_width) for i in range(max(trunk_layer-1,0))]
                                    + [nn.Linear(trunk_width, numBranchFeatures*2)]
                                    )
        
        # add the modules manually to ensure, that all parameters appear in model paramters
        for idx,module in enumerate(self.trunkNet_Lin):
            self.add_module(f"trunkNet_Lin_{idx}", module)


        self.branchFeatures =   nn.ParameterList([nn.Parameter(torch.randn(( numBranchFeatures*2))) for i in range( numSolutions )])

        # add the modules manually to ensure, that all parameters appear in model paramters
        #for idx,module in enumerate(self.branchFeatures):
        #    self.add_module(f"branchFeatures{idx}", module)


        self.deepONet_biases     =    nn.Parameter(torch.randn(2))

    def trunk(self, x: torch.Tensor)->torch.Tensor:
        out = torch.zeros_like(x, device= x.device) + x
        for i in range(self.trunk_layer):
            skipConn = out.clone()
            out = self.activationFunction(self.trunkNet_Lin[i](out))
            if self.skipConnection and i != 0:
                out = out + skipConn
        
        out = self.trunkNet_Lin[-1](out)

        return out
    


    

    def forward(self,  x: torch.Tensor)->dict[str, list[torch.Tensor]]:
        trunkOut = self.trunk(x)

        branchOut = self.branchFeatures


        batchSize = trunkOut.shape[0]
        branchTiledFeatures = []

        out1, out2 = [],[]

        for i in range(self.numSolutions):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            branchTiledFeatures.append(tiledBranchAux )
            totalOutAux = trunkOut*tiledBranchAux

            out1.       append( ( torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )
            out2.       append( ( torch.sum(totalOutAux[:,   self.numBranchFeatures: 2*self.numBranchFeatures], dim = 1) + self.deepONet_biases[1]).view(-1,1) )


        if self.DirichletHardConstraint:
            for idxSol in range(self.numSolutions):
                out1[idxSol] = out1[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")*20
                out2[idxSol] = out2[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")*10
                if self.DirichletConditionFunc1 != None:
                    #out1_preHardConst.append(out1[idxSol].clone())
                    boundaryExtension1 = self.DirichletConditionFunc1(x).view(-1,1)
                    # scaling = max()
                    out1[idxSol] = out1[idxSol] + boundaryExtension1
                if self.DirichletConditionFunc2 != None:
                    #out2_preHardConst.append(out1[idxSol].clone())
                    boundaryExtension2 = self.DirichletConditionFunc2(x).view(-1,1)
                    out2[idxSol] = out2[idxSol] +boundaryExtension2

        dic = {"out1": out1, "out2": out2}

        return dic