from torch import nn
import deepxde as dde
from typing import Callable, Tuple
import torch
from .SwiGLU import *


class two_dim_2_two_dim_DefPINN(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the output is 2.
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

        


        self.trunkNet_Lin        =  nn.ModuleList(
                                      [nn.Linear(2,trunk_width)]
                                    + [nn.Linear(trunk_width, trunk_width) for i in range(max(trunk_layer-1,0))]
                                    + [nn.Linear(trunk_width, numBranchFeatures*2)]
                                    )


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
        # branchTiledFeatures = []

        out1, out2 = [],[]

        for i in range(self.numSolutions):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            # branchTiledFeatures.append(tiledBranchAux )
            totalOutAux = trunkOut*tiledBranchAux

            out1.       append( ( torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )
            out2.       append( ( torch.sum(totalOutAux[:,   self.numBranchFeatures: 2*self.numBranchFeatures], dim = 1) + self.deepONet_biases[1]).view(-1,1) )


        if self.DirichletHardConstraint:
            for idxSol in range(self.numSolutions):
                out1[idxSol] = out1[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
                out2[idxSol] = out2[idxSol] * self.geom.boundary_constraint_factor(x, smoothness="Cinf")
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
    




class one_dim_DefPINN(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the output is 1. Input dimension is 1.
    """
    def __init__(   self,
                    numSolutions: int,
                    numBranchFeatures: int,
                    trunk_layer: int,
                    trunk_width: int,
                    activationFunction: Callable[[torch.Tensor], torch.Tensor],
                    DirichletHardConstraint: bool,
                    skipConnection: bool = False,
                    useSwiGLU: bool = True,
                    fourierFeatures:bool = True,
                    DirichletConstAt1: float = -1., # the first value is a coordinate and the second is the value the funciton should have at that coordinate
                    DirichletConstAt2: float = 1.,
                    DirichletConstValLeft:float = 0,
                    DirichletConstValRight:float = 0,
                    ):
        
        super().__init__()
        
        self.activationFunction = activationFunction
        self.numBranchFeatures = numBranchFeatures
        self.DirichletHardConstraint = DirichletHardConstraint
        self.fourierFeatures = fourierFeatures
        self.skipConnection = skipConnection
        self.numSolutions = numSolutions
        self.trunk_layer = trunk_layer
        self.useSwiGLU = useSwiGLU

        self.boundaryParam1 = (DirichletConstValLeft - DirichletConstValRight)/( DirichletConstAt1 - DirichletConstAt2) #(DirichletConstValLeft - DirichletConstValRight + DirichletConstAt2**2 - DirichletConstAt1**2)/( DirichletConstAt1 - DirichletConstAt2)
        self.boundaryParam2 = DirichletConstValRight - self.boundaryParam1*DirichletConstAt2 #DirichletConstValLeft - DirichletConstAt1 **2 - DirichletConstAt1*self.boundaryParam1
        self.DirichletConstAt1 = DirichletConstAt1
        self.DirichletConstAt2 = DirichletConstAt2
        mid = (self.DirichletConstAt1 + self.DirichletConstAt2)/2
        self.DirichletRegularizationConstant = ( mid- self.DirichletConstAt1)*(mid- self.DirichletConstAt2)

        
        if fourierFeatures:
            multFourierFeatures = 3

        else:
            multFourierFeatures = 1

        if useSwiGLU:
            self.swigluActivationFunctions = nn.ModuleList([SwiGLUFFN(d_model = trunk_width  , d_ff=int((2/3) * 4 * trunk_width)) for _ in  range(trunk_layer)])

        self.trunkNet_Lin        =  nn.ModuleList(
                                      [nn.Linear(1,trunk_width)]
                                    + [nn.Linear(trunk_width, trunk_width) for i in range(max(trunk_layer-1,0))]
                                    + [nn.Linear(trunk_width, multFourierFeatures * numBranchFeatures)]
                                    )
        if fourierFeatures:
            self.fourierNetSin = nn.Linear(numBranchFeatures, numBranchFeatures)
            self.fourierNetCos = nn.Linear(numBranchFeatures, numBranchFeatures)


        self.branchFeatures =   nn.ParameterList([nn.Parameter(torch.randn((multFourierFeatures * numBranchFeatures))) for i in range( numSolutions )])

  

        self.deepONet_biases     =    nn.Parameter(torch.randn(1))

    def trunk(self, x: torch.Tensor)->torch.Tensor:
        out = torch.zeros_like(x, device= x.device) + x
        for i in range(self.trunk_layer):
            skipConn = out.clone()
            if self.useSwiGLU:
                out = self.swigluActivationFunctions[i].forward(self.trunkNet_Lin[i](out))
            else:
                out = self.activationFunction(self.trunkNet_Lin[i](out))
            if self.skipConnection and i != 0:
                out = out + skipConn
        
        out = self.trunkNet_Lin[-1](out)

        return out
    

    def fourierAugumentedLayer(self,x):
        if self.fourierFeatures:
            x1 = x[:,:self.numBranchFeatures]
            frequencies = torch.linspace(1, 100,steps=self.numBranchFeatures, device=x.device, dtype=x.dtype)
            x2 = torch.sin(frequencies*x[:,self.numBranchFeatures : 2*self.numBranchFeatures])
            x3 = torch.cos(frequencies*x[:,2*self.numBranchFeatures : 3*self.numBranchFeatures])
            x2 = self.fourierNetSin(x2)
            x3 = self.fourierNetSin(x3)

            return torch.cat((x1,x2,x3), dim = 1 )
        else:
            return x
    

    def forward(self,  x: torch.Tensor)->dict[str, list[torch.Tensor]]:
        trunkOut = self.trunk(x)

        branchOut = self.branchFeatures


        batchSize = trunkOut.shape[0]
        
        vecOut = []
        out = []

        for i in range(self.numSolutions):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            totalOutAux = trunkOut*tiledBranchAux
            vecOut.append( totalOutAux)

            #out.append( ( torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )


        for i in range(self.numSolutions):
            vecOut[i] = self.fourierAugumentedLayer(vecOut[i])
            out.append( ( torch.sum(vecOut[i], dim = 1) + self.deepONet_biases[0]).view(-1,1) )


        if self.DirichletHardConstraint:
            for idxSol in range(self.numSolutions):
                #out[idxSol] = out[idxSol] *torch.tanh(10*(x- self.DirichletConstAt1))*torch.tanh(10*(x - self.DirichletConstAt2))
                out[idxSol] = out[idxSol] * 0.05*( x- self.DirichletConstAt1)*(x- self.DirichletConstAt2)/self.DirichletRegularizationConstant
                out[idxSol] = out[idxSol] + x*self.boundaryParam1 +self.boundaryParam2 

        

        dic = {"out": out}

        return dic
    


class two_dim_2_one_dim_DefPINN(nn.Module):
    """
    This model is a Deflation Diffusion DeepONet; The dimension of the output is 2.
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
                    r_function = None,
                    DirichletConditionFunc: Callable[[torch.Tensor], torch.Tensor] = None,
                    ):
        super().__init__()
        
        self.activationFunction = activationFunction
        self.numBranchFeatures = numBranchFeatures
        self.geom = geom
        self.DirichletHardConstraint = DirichletHardConstraint
        self.skipConnection = skipConnection
        self.numSolutions = numSolutions
        self.DirichletConditionFunc = DirichletConditionFunc
        self.trunk_layer = trunk_layer
        if not r_function:
            if geom:
                self.r_function = lambda x: self.geom.boundary_constraint_factor(x, smoothness="Cinf")
        else:
            self.r_function = r_function

        


        self.trunkNet_Lin        =  nn.ModuleList(
                                      [nn.Linear(2,trunk_width)]
                                    + [nn.Linear(trunk_width, trunk_width) for i in range(max(trunk_layer-1,0))]
                                    + [nn.Linear(trunk_width, numBranchFeatures)]
                                    )


        self.branchFeatures =   nn.ParameterList([nn.Parameter(torch.randn(( numBranchFeatures))) for i in range( numSolutions )])



        self.deepONet_biases     =    nn.Parameter(torch.randn(1))

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


        out= []

        for i in range(self.numSolutions):
            tiledBranchAux = torch.tile(branchOut[i], (batchSize,1))
            totalOutAux = trunkOut*tiledBranchAux

            out.       append( ( torch.sum(totalOutAux[:,                         :   self.numBranchFeatures], dim = 1) + self.deepONet_biases[0]).view(-1,1) )


        if self.DirichletHardConstraint:
            for idxSol in range(self.numSolutions):
                out[idxSol] = out[idxSol] * self.r_function(x)
                if self.DirichletConditionFunc != None:
                    boundaryExtension = self.DirichletConditionFunc(x).view(-1,1)
                    out[idxSol] = out[idxSol] + boundaryExtension


        dic = {"out": out}

        return dic