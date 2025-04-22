import numpy as np
import sys
#import os
import pandas as pd
import time
#from numba import njit, jit, prange, float64, int64
#import numba as nb
import torch
import torch.nn as nn
import torch_geometric
import math
#import matplotlib.pyplot as plt


from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

if __name__=="__main__":
    M = int(sys.argv[1])

    cuda = torch.device('cuda')

    w_truth = torch.load('../May16_FOM_data/w_test_430_021')
    #w_truth = torch.load('May16_FOM_data/w_train_padded')
    N = w_truth.size()[0]
    nt = w_truth.size()[1]
    batchSize = 1

    w_pred = torch.zeros((N,nt))
    w_pred = torch.load('../CNN_outputs_May7/lat'+str(M)+'_430_021')

    input_hist = w_truth[:,:]
    pred_hist = w_pred[:,:]


    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist[:-30,:nt]-pred_hist[:,:nt], dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist[:-30,:nt], dim=0).flatten())
    print(full_rel_error)
