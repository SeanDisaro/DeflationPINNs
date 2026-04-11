import torch


def linearDeflationLoss_dictModel(modelOut:dict[list[torch.Tensor]], maxLoss:float=10000, maxDistance:float = 0.1, keyOutModel = "out")->torch.Tensor:
    """Linear deflation function. We compute basically sum _i,j max(maxLoss - maxLoss/ maxDistance * dist(u_i, u_j), 0  )

    Args:
        modelOut (dict[torch.Tensor]): Output of our model.
        maxLoss (float, optional): Maximal possible loss. Defaults to 10000.
        maxDistance (float, optional): After dist(u_i, u_j) gets to this value, the loss becomes 0 and we stop optimizing for deflation. Defaults to 0.1.

    Returns:
        torch.Tensor: Deflation Loss
    """
    out = modelOut[keyOutModel]
    n = len(out)
    m = maxLoss/maxDistance
    loss = torch.Tensor([0.]).to(out[0].device)
    if n == 1:
        return loss
    for i in range(n):
        for j in range(n-1-i):
            difference_ij_1 = out[i] - out[i + j+1]
            lossAux2 = torch.maximum((maxLoss - m* torch.nanmean(torch.norm(difference_ij_1 , dim = 1))) , torch.tensor(0.))
            loss = loss + lossAux2
    
    loss = 2*loss /(n* (n-1))
    return loss