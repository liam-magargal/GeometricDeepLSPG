import numpy as np
import sys
import pandas as pd
import time
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


def trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, pos, pos_grid, ai_enc, ai_dec, modelNum, N_lat):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
    ns_train = w_train.size()[0]
    ns_val = w_val.size()[0]
    
    train_set = np.arange(ns_train)
    val_set = np.arange(ns_val)
    
    w_train = w_train.to(device)
    w_val = w_val.to(device)
    
    pos = pos.to(device)
    pos_grid = pos_grid.to(device)
    ai_enc = ai_enc.to(device)
    ai_dec = ai_dec.to(device)
    
    
    for epoch in range(maxEpoch):
        t1 = time.time()
        np.random.shuffle(train_set)
        for i in range(350):
            optimizer.zero_grad()
            train_set_r = train_set[i*batchSize:(i+1)*batchSize]
        
            np.random.shuffle(val_set)
            val_set_r = val_set[0:batchSize]
        
            ## train loss
            inputs = w_train[train_set_r,:,:]
            x_lat = model.encode(inputs,pos_grid,pos,ai_enc,N_grid_x,batchSize)
            x_pred = model.decode(x_lat,pos_grid,pos,ai_dec,N_grid_x,batchSize)

            x_target = w_train[train_set_r,:,:]
            loss = torch.linalg.norm(x_pred - x_target)
            loss = loss*loss
            
            loss.backward()
            optimizer.step()
            
        ## validation loss
        inputs = w_val[val_set_r,:,:]
        
        x_lat = model.encode(inputs,pos_grid,pos,ai_enc,N_grid_x,batchSize)
        x_pred = model.decode(x_lat,pos_grid,pos,ai_dec,N_grid_x,batchSize)
        
        x_target = w_val[val_set_r,:,:]
        val_loss = torch.linalg.norm(x_pred - x_target)
        val_loss = val_loss*val_loss
        
        t2 = time.time()
        print('epoch: ', epoch, ' train loss: ', loss.cpu().detach().numpy(), ', validation loss: ', val_loss.cpu().detach().numpy(), 't2-t1: ', t2-t1)
        
        torch.save(model, 'lat' + str(N_lat) + '_model'+str(modelNum))
        
    return model


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
        
    return x_hist, cell_volumes

def loadDataTotal(numTrain, numTest, numVal, ns, N):
    gamma = 1.4
    
    ni = 5
    nj = 5
    x_hist = torch.zeros((ni*nj*ns,N,4))
    
    for i in range(ni):
        for j in range(nj):
            train_file = 'output'
            mesh_location = '../Mesh'
            output_location = '../fullOrderModel/Outputs/'+str(2*i+12)+'_'+str(j+3)
            volume_file = 'cell_volumes.npy'
            x_hist_temp, cell_volumes = loadDataSingle(train_file, mesh_location, output_location, volume_file, ns, gamma)
            x_hist[(i*ni+j)*ns:(i*ni+j+1)*ns,:,:] = torch.tensor(x_hist_temp)
    
    total_set = np.arange(ni*nj*ns)
    np.random.shuffle(total_set)
    
    train_set = total_set[numVal:]
    val_set = total_set[:numVal]
    
    x_val = x_hist[val_set,:,:]
    x_train = x_hist[train_set,:,:]
    
    return x_train, x_val


def defineModel(N_lat):
    model = architecture(N_lat)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) 
    
    return model, optimizer

if __name__=="__main__":
    N_lat = int(sys.argv[1])
    modelNum = int(sys.argv[2])
    N = 4328
    nt = 301
    numTrain = 9
    numVal = 525
    batchSize = 20
    maxEpoch = 5000

    # set up grid for interpolation
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

    # load training/validation data
    w_train, w_val = loadDataTotal(numTrain, numVal, nt, N)

    # initialize model
    model, optimizer = defineModel(N_lat)

    # train model
    model = trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, pos, pos_grid, ai_enc, ai_dec, modelNum, N_lat)

