import numpy as np
import time
import torch
import torch.nn as nn
import torch_geometric
import sys

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

def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, assign_index: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    # This function has been slightly modified from the original version from Pytorch-Geometric

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
        
        self.conv1_1 = torch_geometric.nn.SAGEConv(1,8,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_1.lin_r.weight)
        
        self.conv1_2 = torch_geometric.nn.SAGEConv(8,8,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_2.lin_r.weight)
        
        self.conv2_1 = torch_geometric.nn.SAGEConv(8,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_1.lin_r.weight)
        
        self.conv2_2 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_2.lin_r.weight)
        
        self.conv3_1 = torch_geometric.nn.SAGEConv(16,32,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_1.lin_r.weight)
        
        self.conv3_2 = torch_geometric.nn.SAGEConv(32,32,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_2.lin_r.weight)
        
        self.conv4_1 = torch_geometric.nn.SAGEConv(32,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_1.lin_r.weight)
        
        self.conv4_2 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_2.lin_r.weight)
        
        self.layer5 = torch.nn.Linear(128,M,bias=False)
        nn.init.xavier_uniform_(self.layer5.weight.data)
        
        self.layer6 = torch.nn.Linear(M,128,bias=False)
        nn.init.xavier_uniform_(self.layer6.weight.data)
        
        self.conv7_1 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_1.lin_r.weight)
        
        self.conv7_2 = torch_geometric.nn.SAGEConv(64,32,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_2.lin_r.weight)
        
        self.conv8_1 = torch_geometric.nn.SAGEConv(32,32,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_1.lin_r.weight)
        
        self.conv8_2 = torch_geometric.nn.SAGEConv(32,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_2.lin_r.weight)
        
        self.conv9_1 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_1.lin_r.weight)
        
        self.conv9_2 = torch_geometric.nn.SAGEConv(16,8,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_2.lin_r.weight)
        
        self.conv10_1 = torch_geometric.nn.SAGEConv(8,8,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_1.lin_r.weight)
        
        self.conv10_2 = torch_geometric.nn.SAGEConv(8,1,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_2.lin_r.weight)
        
        
    def encode(self,x,batchSize,e1,e2,e3,e4,e5,s1,s2,s3,s4,labels1,labels2,labels3,labels4):
        
        x = (x - 1.) / 7.5
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
        
        # MLP
        x = x.reshape((int(batchSize),128))
        x = (self.layer5(x))
        
        return x

    def decode(self,x,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5, ai_54, ai_43, ai_32, ai_21):
        
        # MLP
        x = torch.nn.functional.elu(self.layer6(x))
        
        x = x.reshape((int(batchSize),2,64))
        
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
        
        x = x * 7.5 + 1.
        
        x = x.reshape((int(batchSize),256+60,1))
        
        return x[:,30:-30,:]

    

def trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
    N = w_train.size()[0]
    ns_train = w_train.size()[1]
    ns_val = w_val.size()[1]
    
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
    
    numBatch = 1804
    
    for epoch in range(maxEpoch):
        np.random.shuffle(train_set)
        
        t1 = time.time()

        # loop over batches for training loss
        for batch in range(numBatch):
            optimizer.zero_grad()
            train_set_r = train_set[batch*batchSize:(batch+1)*batchSize]
            
            inputs = w_train[:,train_set_r].T.reshape((batchSize,N,1))
            
            x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4, labels1, labels2, labels3, labels4)
            
            x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
            
            x_target = w_train[:,train_set_r].T.reshape((batchSize,N,1))
            loss = torch.linalg.norm(x_pred - x_target[:,30:-30,:])
            loss = loss*loss
            
            loss.backward()
            optimizer.step()
        
        
        ## validation loss
        np.random.shuffle(val_set)
        inputs = w_val[:,val_set[:batchSize]].T.reshape((batchSize,N,1))

        x_lat = model.encode(inputs,batchSize.clone().detach(), e1, e2, e3, e4, e5, s1, s2, s3, s4, labels1, labels2, labels3, labels4).detach()
        x_pred = model.decode(x_lat,batchSize.clone().detach(), e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21).detach()
        
        x_target = w_val[:,val_set[:batchSize]].T.reshape((batchSize,N,1)).detach()
        
        val_loss = torch.linalg.norm(x_pred - x_target[:,30:-30,:])
        val_loss = val_loss*val_loss
        
        torch.save(model, 'lat1')

        t2 = time.time()
        
        print('epoch: ', epoch, ', loss: ', loss.cpu().detach(), ', val_loss: ', val_loss.cpu().detach(),', lr: ', optimizer.param_groups[0]['lr'], ', epoch time: ', t2-t1)
        
        
    return model


def loadData(numTrain, numVal, nt, N):
    w_hist_train = torch.load('fullOrderModel/w_train')
    
    ## append padded nodes to the domain
    x_leftBoundary = torch.zeros((30,nt*10*8))
    for i in range(10):
        mu1 = 4.25 + (1.25/9.)*i
        x_leftBoundary[:,8*i*nt:8*(i+1)*nt] = mu1*torch.ones((30,8*nt))
    w_hist_train = torch.cat((x_leftBoundary,w_hist_train),dim=0)
    ## end append
    
    ns_train = w_hist_train.size()[1]
    
    w_train = w_hist_train
    full_set = np.arange(ns_train)
    np.random.shuffle(full_set)
    val_set = full_set[:numVal]
    train_set = full_set[numVal:]
    
    w_val = w_train[:,val_set].clone()
    w_train = w_train[:,train_set].clone()
    
    return w_train, w_val

def defineModel(N_lat):
    model = architecture(N_lat)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    
    return model, optimizer

if __name__=="__main__":
    N_lat = int(sys.argv[1])
    N = 256
    N_lat = 1
    nt = 501
    numTrain = 80 # number of training parameter solutions
    numVal = 4000 # number of snapshots from training stored as validation
    batchSize = 20
    maxEpoch = 1000
    
    batchSize = torch.tensor(batchSize,dtype=int)
    
    # load hierarchical graph data
    directory = '1D_pooling_unpooling/'
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
    
    # load training data
    w_train, w_val = loadData(numTrain, numVal, nt, N)
    
    # set up model
    model, optimizer = defineModel(N_lat)
    
    # train model
    model = trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4)
    