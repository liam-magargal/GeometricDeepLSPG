import numpy as np
import sys
import pandas as pd
import time
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from typing import Optional
from torch import Tensor

def _avg_pool_x(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
) -> Tensor:
    return scatter(x, cluster, dim=1, reduce='mean')


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
    # This function has been slightly modified from Pytorch-Geometric

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
        
        self.conv1_1 = torch_geometric.nn.SAGEConv(4,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_1.lin_r.weight)
        
        self.conv1_2 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_2.lin_r.weight)
        
        self.conv2_1 = torch_geometric.nn.SAGEConv(16,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_1.lin_r.weight)
        
        self.conv2_2 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_2.lin_r.weight)
        
        self.conv3_1 = torch_geometric.nn.SAGEConv(64,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_1.lin_r.weight)
        
        self.conv3_2 = torch_geometric.nn.SAGEConv(128,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_2.lin_r.weight)
        
        self.conv4_1 = torch_geometric.nn.SAGEConv(128,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_1.lin_r.weight)
        
        self.conv4_2 = torch_geometric.nn.SAGEConv(256,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_2.lin_r.weight)
        
        self.layer5 = torch.nn.Linear(512,M)
        nn.init.xavier_uniform_(self.layer5.weight.data)
        
        self.layer6 = torch.nn.Linear(M,512)
        nn.init.xavier_uniform_(self.layer6.weight.data)
        
        self.conv7_1 = torch_geometric.nn.SAGEConv(256,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_1.lin_r.weight)
        
        self.conv7_2 = torch_geometric.nn.SAGEConv(256,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_2.lin_r.weight)
        
        self.conv8_1 = torch_geometric.nn.SAGEConv(128,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_1.lin_r.weight)
        
        self.conv8_2 = torch_geometric.nn.SAGEConv(128,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_2.lin_r.weight)
        
        self.conv9_1 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_1.lin_r.weight)
        
        self.conv9_2 = torch_geometric.nn.SAGEConv(64,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_2.lin_r.weight)
        
        self.conv10_1 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_1.lin_r.weight)
        
        self.conv10_2 = torch_geometric.nn.SAGEConv(16,4,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_2.lin_r.weight)
        
        
        
    def encode(self,x,batchSize,e1,e2,e3,e4,e5,s1,s2,s3,s4,labels1,labels2,labels3,labels4):
        
        # scale
        x[:,:,0] = (x[:,:,0] - .5) / .5
        x[:,:,1] = (x[:,:,1] + 2.) / 1.5
        x[:,:,2] = (x[:,:,2] + 1.) / .75
        x[:,:,3] = (x[:,:,3] - 1.) / 3.5
        
        # MPP layer 1
        x = torch.nn.functional.elu(self.conv1_1(x,e1))
        x = torch.nn.functional.elu(self.conv1_2(x,e1))
        x = _avg_pool_x(labels1,x)
        
        # MPP layer 2
        x = torch.nn.functional.elu(self.conv2_1(x,e2))
        x = torch.nn.functional.elu(self.conv2_2(x,e2))
        x = _avg_pool_x(labels2,x)

        # MPP layer 3
        x = torch.nn.functional.elu(self.conv3_1(x,e3))
        x = torch.nn.functional.elu(self.conv3_2(x,e3))
        x = _avg_pool_x(labels3,x)

        # MPP layer 4
        x = torch.nn.functional.elu(self.conv4_1(x,e4))
        x = torch.nn.functional.elu(self.conv4_2(x,e4))
        x = _avg_pool_x(labels4,x)
        
        # MLP layer
        x = x.reshape((batchSize,512))
        x = self.layer5(x)
        
        return x
    
    
    def decode(self,x,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5, ai_54, ai_43, ai_32, ai_21):
        
        # MLP layer
        x = torch.nn.functional.elu(self.layer6(x))
        x = x.reshape((batchSize,2,256))
        
        # UMP layer 1
        x = knn_interpolate(x,pos5,pos4,assign_index=ai_54)
        x = torch.nn.functional.elu(self.conv7_1(x,e4))
        x = torch.nn.functional.elu(self.conv7_2(x,e4))
        
        # UMP layer 2
        x = knn_interpolate(x,pos4,pos3,assign_index=ai_43)
        x = torch.nn.functional.elu(self.conv8_1(x,e3))
        x = torch.nn.functional.elu(self.conv8_2(x,e3))
        
        # UMP layer 3
        x = knn_interpolate(x,pos3,pos2,assign_index=ai_32)
        x = torch.nn.functional.elu(self.conv9_1(x,e2))
        x = torch.nn.functional.elu(self.conv9_2(x,e2))
        
        # UMP layer 4
        x = knn_interpolate(x,pos2,pos1,assign_index=ai_21)
        x = torch.nn.functional.elu(self.conv10_1(x,e1))
        x = self.conv10_2(x,e1)
       
        # scale
        x[:,:,0] = (x[:,:,0] * .5) + .5
        x[:,:,1] = (x[:,:,1] * 1.5) - 2.
        x[:,:,2] = (x[:,:,2] * .75) - 1.
        x[:,:,3] = (x[:,:,3] * 3.5) + 1.


        x = x.reshape((batchSize,4328,4))
        
        return x


def trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, modelNum,labels1,labels2,labels3,labels4,N_lat):
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
    
    e1 = e1.to(device)
    e2 = e2.to(device)
    e3 = e3.to(device)
    e4 = e4.to(device)
    e5 = e5.to(device)
    
    s1 = s1.to(device)
    s2 = s2.to(device)
    s3 = s3.to(device)
    s4 = s4.to(device)
    
    u1 = u1.to(device)
    u2 = u2.to(device)
    u3 = u3.to(device)
    u4 = u4.to(device)
    
    pos1 = pos1.to(device)
    pos2 = pos2.to(device)
    pos3 = pos3.to(device)
    pos4 = pos4.to(device)
    pos5 = pos5.to(device)
    
    ai_54 = ai_54.to(device)
    ai_43 = ai_43.to(device)
    ai_32 = ai_32.to(device)
    ai_21 = ai_21.to(device)

    labels1 = labels1.to(device)
    labels2 = labels2.to(device)
    labels3 = labels3.to(device)
    labels4 = labels4.to(device)
    
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
            x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4, labels1, labels2, labels3, labels4)
            x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
            x_target = w_train[train_set_r,:,:]
            loss = torch.linalg.norm(x_pred - x_target)
            loss = loss*loss
            
            loss.backward()
            optimizer.step()
            
        ## validation loss
        inputs = w_val[val_set_r,:,:]
        x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4, labels1, labels2, labels3, labels4)
        x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
        x_target = w_val[val_set_r,:,:]
        val_loss = torch.linalg.norm(x_pred - x_target)
        val_loss = val_loss*val_loss
        
        torch.save(model, 'lat' + str(N_lat) + '_model'+str(modelNum))
        
        t2 = time.time()
        print('epoch: ', epoch, ' train loss: ', loss.cpu().detach().numpy(), ', validation loss: ', val_loss.cpu().detach().numpy(), ', epoch time (s): ', t2-t1)
        
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
        
    return x_hist

def loadDataTotal(numVal, ns, N):
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
            x_hist_temp = loadDataSingle(train_file, mesh_location, output_location, volume_file, ns, gamma)
            x_hist[(i*ni+j)*ns:(i*ni+j+1)*ns,:,:] = torch.tensor(x_hist_temp)
    
    total_set = np.arange(ni*nj*ns)
    np.random.shuffle(total_set)
    
    train_set = total_set[numVal:]
    val_set = total_set[:numVal]
    
    x_val = x_hist[val_set,:,:]
    x_train = x_hist[train_set,:,:]
    
    return x_train, x_val


def defineModel(M):
    model = architecture(M)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) 
    
    return model, optimizer


if __name__=="__main__":
    N_lat = int(sys.argv[1])
    modelNum = int(sys.argv[2])
    N = 4328
    nt = 301
    numVal = 525
    batchSize = 20
    maxEpoch = 5000


    directory = '../2D_pooling_unpooling/'
    e1 = torch.load(directory + 'edge_index')
    e2 = torch.load(directory + 'e2')
    e3 = torch.load(directory + 'e3')
    e4 = torch.load(directory + 'e4')
    e5 = torch.load(directory + 'e5')

    s1 = torch.load(directory + 's1')
    s2 = torch.load(directory + 's2')
    s3 = torch.load(directory + 's3')
    s4 = torch.load(directory + 's4')

    u1 = torch.load(directory + 'unpool1')
    u2 = torch.load(directory + 'unpool2')
    u3 = torch.load(directory + 'unpool3')
    u4 = torch.load(directory + 'unpool4')
    
    pos1 = torch.load(directory + 'pos1')
    pos2 = torch.load(directory + 'pos2')
    pos3 = torch.load(directory + 'pos3')
    pos4 = torch.load(directory + 'pos4')
    pos5 = torch.load(directory + 'pos5')
    
    labels1 = torch.load(directory + 'labels1')
    labels2 = torch.load(directory + 'labels2')
    labels3 = torch.load(directory + 'labels3')
    labels4 = torch.load(directory + 'labels4')


    ai_54 = knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_43 = knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_32 = knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_21 = knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)

    w_train, w_val = loadDataTotal(numVal, nt, N)
    
    model, optimizer = defineModel(N_lat)
    
    model = trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, modelNum,labels1,labels2,labels3,labels4,N_lat)

