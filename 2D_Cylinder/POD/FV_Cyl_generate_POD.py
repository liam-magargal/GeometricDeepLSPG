import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric


from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = 1/(gamma-1)*P/rho + .5*(u*u + v*v)
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    Energy = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H



def loadDataSingle(train_file, mesh_location, output_location, volume_file, ns, gamma):
    cell_volumes = torch.tensor(np.load(mesh_location + '/' + volume_file))
    data = torch.tensor(np.array(pd.read_csv(output_location + '/' + train_file + "0000001.csv"))) 
    
    N = np.shape(data)[0]
    x_hist = torch.zeros((ns,N,4))
    
    for t in range(ns):
        print(t)
        if t<10:
            name = output_location + '/' + train_file + "000000" + str(t) + ".csv"
        elif t<100:
            name = output_location + '/' + train_file + "00000" + str(t) + ".csv"
        elif t<1000:
            name = output_location + '/' + train_file + "0000" + str(t) + ".csv"
        elif t<10000:
            name = output_location + '/' + train_file + "000" + str(t) + ".csv"
        elif t<100000:
            name = output_location + '/' + train_file + "00" + str(t) + ".csv"
        elif t<1000000:
            name = output_location + '/' + train_file+ "0" + str(t) + ".csv"
        else:
            name = output_location + '/' + train_file + str(t) + ".csv"
        
        data = torch.tensor(np.array(pd.read_csv(name)))
        
        rho = data[:,3]
        vx = data[:,4]
        vy = data[:,5]
        P = data[:,6]
        
        Mass, Momx, Momy, Energy, E, H = getConserved( rho, vx, vy, P, gamma, cell_volumes )
        
        x_hist[t,:,0] = Mass
        x_hist[t,:,1] = Momx
        x_hist[t,:,2] = Momy
        x_hist[t,:,3] = Energy
        
    return x_hist

def loadDataTotal(ns, N):
    gamma = 1.4
    
    ni = 6
    x_hist = torch.zeros((ni*ns,N,4))
   
    for i in range(ni):
        print('i: ', i)
        train_file = 'output'
        mesh_location = '../Mesh'
        output_location = '../fullOrderModel/Outputs/mach_'+str(1000+50*i)
        volume_file = 'cell_volumes.npy'
        x_hist_temp, cell_volumes = loadDataSingle(train_file, mesh_location, output_location, volume_file, ns, gamma) 
        x_hist[(i)*ns:(i+1)*ns,:,:] = torch.tensor(x_hist_temp)

    return x_hist


N = 4148
nt = 1001

w_train = loadDataTotal(nt, N)
n_samples = w_train.size(dim=0)

w_svd = torch.zeros((4*N,n_samples))
for i in range(n_samples):
    w_svd[:N,i] = w_train[i,:,0]
    w_svd[N:2*N,i] = w_train[i,:,1]
    w_svd[2*N:3*N,i] = w_train[i,:,2]
    w_svd[3*N:4*N,i] = w_train[i,:,3]
    
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
w_svd = w_svd.to(device)

U, S, Vh = torch.linalg.svd(w_svd,full_matrices=False)

U = U[:,:50]

data_out = U.detach().cpu().numpy()
df = pd.DataFrame(data_out)
df.to_csv('U.csv')
