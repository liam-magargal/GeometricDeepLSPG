import numpy as np
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
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

def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)

    return rho, u, v, P, E, H

def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, assign_index: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    # This function is slightly modified from its original form in Pytorch-Geometric

    with torch.no_grad():
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        
    y = scatter(x[:,x_idx] * weights, y_idx, 1, pos_y.size(0), reduce='sum')
    y = y / scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')
    
    return y


class architecture(nn.Module):
    def __init__(self,M):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv1.weight.data)
        nn.init.zeros_(self.conv1.bias.data)

        self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv2.weight.data)
        nn.init.zeros_(self.conv2.bias.data)

        self.conv3 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv3.weight.data)
        nn.init.zeros_(self.conv3.bias.data)

        self.conv4 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv4.weight.data)
        nn.init.zeros_(self.conv4.bias.data)

        self.linear1 = torch.nn.Linear(64*4*4,M)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)

        self.linear2 = torch.nn.Linear(M,64*4*4)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.zeros_(self.linear2.bias.data)

        self.conv5 = torch.nn.ConvTranspose2d(64,32,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv5.weight.data)
        nn.init.zeros_(self.conv5.bias.data)

        self.conv6 = torch.nn.ConvTranspose2d(32,16,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv6.weight.data)
        nn.init.zeros_(self.conv6.bias.data)

        self.conv7 = torch.nn.ConvTranspose2d(16,8,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv7.weight.data)
        nn.init.zeros_(self.conv7.bias.data)
 
        self.conv8 = torch.nn.ConvTranspose2d(8,4,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv8.weight.data)
        nn.init.zeros_(self.conv8.bias.data)


    def encode(self, x, pos_grid, pos_cloud, ai_enc, N_grid_x, batchSize):
        x[:,:,0] = (x[:,:,0] - .5) / .5
        x[:,:,1] = (x[:,:,1] + 2.) / 1.5
        x[:,:,2] = (x[:,:,2] + 1.) / .75
        x[:,:,3] = (x[:,:,3] - 1.) / 3.5
        
        x = knn_interpolate(x,pos_cloud,pos_grid,assign_index=ai_enc)
        
        x = x.transpose(1,2)
        x = x.reshape((batchSize, 4, N_grid_x, N_grid_x))
        x = torch.nn.functional.elu(self.conv1(x))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        x = torch.nn.functional.elu(self.conv4(x))
        x = x.reshape(batchSize, 64*4*4)
        x = torch.nn.functional.elu(self.linear1(x))
        
        return x

    def decode(self, x, pos_grid, pos_cloud, ai_dec, N_grid_x, batchSize):
        x = torch.nn.functional.elu(self.linear2(x))
        x = x.reshape(batchSize, 64, 4, 4)
        x = torch.nn.functional.elu(self.conv5(x))
        x = torch.nn.functional.elu(self.conv6(x))
        x = torch.nn.functional.elu(self.conv7(x))
        x = self.conv8(x)
        
        x = x.reshape((batchSize,4,N_grid_x*N_grid_x))
        x = x.transpose(2,1)
        x = knn_interpolate(x,pos_grid,pos_cloud,assign_index=ai_dec)

        x[:,:,0] = (x[:,:,0]*.5 + .5)
        x[:,:,1] = (x[:,:,1]*1.5 - 2.)
        x[:,:,2] = (x[:,:,2]*.75 - 1.)
        x[:,:,3] = (x[:,:,3]*3.5 + 1.)

        return x

    
def loadDataSingle(train_file, mesh_location, output_location, volume_file, ns, gamma):
    
    cell_volumes = torch.tensor(np.load(mesh_location + '/' + volume_file))
    
    data = torch.tensor(np.array(pd.read_csv(output_location + '/' + train_file + "0000001.csv")))
    
    N = np.shape(data)[0]
    
    x_hist = torch.zeros((ns,N,4))
    
    for t in range(ns):
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

def loadDataTotal(numTrain, numTest, numVal, ns, N, directory):
    
    gamma = 1.4
    
    test_file = 'output'
    mesh_location = '../Meshes/'
    volume_file = 'cell_volumes.npy'
    x_test = loadDataSingle(test_file, mesh_location, directory, volume_file, ns, gamma)
    
    return x_test


def saveOutput(cell_centroids, rho, vx, vy, P, output_directory, filename, N, t):

    cell_centroids = cell_centroids.flatten(order='F')

    data_out = np.concatenate((cell_centroids, rho, vx, vy, P), axis=None)
    data_out = np.reshape(data_out, (N,6), order='F')

    df = pd.DataFrame(data_out)

    if t<10:
        name = output_directory + filename + "000000" + str(t) + ".csv"
    elif t<100:
        name = output_directory + filename + "00000" + str(t) + ".csv"
    elif t<1000:
        name = output_directory + filename + "0000" + str(t) + ".csv"
    elif t<10000:
        name = output_directory + filename + "000" + str(t) + ".csv"
    elif t<100000:
        name = output_directory + filename + "00" + str(t) + ".csv"
    elif t<1000000:
        name = output_directory + filename + "0" + str(t) + ".csv"
    else:
        name = output_directory + filename + str(t) + ".csv"

    if os.path.exists(name):
        os.remove(name)

    df.to_csv(name)

    del cell_centroids, data_out

    return 0



if __name__=="__main__":
    lat_dim = int(sys.argv[1])
    sol = int(sys.argv[2])

    batchSize = 1
    device = torch.device('cpu')
    
    N = 4328
    ns = 301
    
    # load ground truth data
    filename = 'output'
    directory = '../fullOrderModel/Outputs/13_6.5'
    x_test = loadDataTotal(ns, N, directory)
    
    # load trained model
    model = architecture(lat_dim)
    model = torch.load('lat' + str(lat_dim) + '_model'+str(sol)).cpu()
    
    x_test = x_test.to(device)
    
    # generate grid for interpolation
    N_grid = 4096
    N_grid_x = 64
    x_grid = torch.linspace(0,1,steps=N_grid_x)
    grid_x, grid_y = torch.meshgrid(x_grid, x_grid)
    pos_grid = torch.zeros((N_grid,2))
    pos_grid[:,0] = grid_x.flatten()
    pos_grid[:,1] = grid_y.flatten()

    directory = '../Mesh/'
    pos = torch.tensor(np.load(directory + 'cell_centroids.npy'))

    ai_enc = knn(pos, pos_grid, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_dec = knn(pos_grid, pos, k=3, batch_x=None, batch_y=None, num_workers=1)
    
    input_hist = torch.zeros(4*N,ns)
    pred_hist = torch.zeros(4*N,ns)
    
    lat_hist = torch.zeros((ns,lat_dim))
    gamma = 1.4
    
    # get reconstruction error
    for i in range(ns):
        inputs = torch.zeros((1,N,4)).to(device)
        x_target = torch.zeros((1,N,4)).to(device)
        
        inputs[0,:,:] = x_test[i,:,:]
        x_target[0,:,:] = x_test[i,:,:]
        input_hist[:,i] = inputs[0,:,:].flatten()
        
        x_lat = model.encode(inputs,pos_grid,pos,ai_enc,N_grid_x,batchSize) 
        
        lat_hist[i,:] = x_lat.flatten()
        
        x_pred = model.decode(x_lat,pos_grid,pos,ai_dec,N_grid_x,batchSize)

        pred_hist[:,i] = x_pred[0,:,:].flatten()
        
        Mass = x_pred[0,:,0]
        Momx = x_pred[0,:,1]
        Momy = x_pred[0,:,2]
        Energy = x_pred[0,:,3]


    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

    print('dim:', lat_dim, ', sol:', sol, ', error:', full_rel_error)
