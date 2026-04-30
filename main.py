from tests.LDG.run import run as runLDG
from tests.OneD_reaction.run import run as runReaction
from tests.AllenCahn.run import run as runAllenCahn
from src.lossFunctions.OneD_reactionPINNLoss import sourceTerm
import torch
from config import *

if __name__ == "__main__":
    if MY_DEVICE_IDX > -1:
        torch.cuda.set_device(MY_DEVICE_IDX)
    runAllenCahn()
