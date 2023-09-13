'''
Authors: Pedro Gomes 
'''

import torch
import torch.nn as nn


class Resizer(nn.Module):
    def __init__(self, debug = False):
        super(Resizer, self).__init__()

    def forward(self, x):
        output = x
        if type(x) == tuple:
            if type(x[1]) == int: # x[1] == 0
                output = x[0]
            else:
                output = torch.cat(list(x), dim=1)
        return output
