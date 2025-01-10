import torch
from torch import nn

from utils import problem


class ReLULayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a Rectified Linear Unit calculation (ReLU):
        Element-wise:
            - if x > 0: return x
            - else: return 0

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with some shape.
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with the same shape as x.
                Every negative element should be substituted with 0.
                Output data.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.ReLU (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        """
        n = len(x)
        for i in range(n):
            if x[i] >0:
                True
            else:
                x[i]=0
        
        return x #yeah that's not right
        """
        #returns max(x,0) i think
        return torch.clamp(x,0) #https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
        #raise NotImplementedError("Your Code Goes Here")
