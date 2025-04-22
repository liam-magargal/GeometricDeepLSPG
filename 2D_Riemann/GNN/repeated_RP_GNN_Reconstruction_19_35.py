import numpy as np
import sys
import os
import pandas as pd
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


def loadDataSingle(train_file, mesh_location, directory, volume_file, ns, gamma):
    
    cell_volumes = torch.tensor(np.load(mesh_location + '/' + volume_file))
    data = torch.tensor(np.array(pd.read_csv(directory + '/' + train_file + "0000001.csv")))
    
    N = np.shape(data)[0]
    x_hist = torch.zeros((ns,N,4))
    
    for t in range(ns):
        if t<10:
            name = directory+ '/' + train_file + "000000" + str(t) + ".csv"
        elif t<100:
            name = directory + '/' + train_file + "00000" + str(t) + ".csv"
        elif t<1000:
            name = directory + '/' + train_file + "0000" + str(t) + ".csv"
        elif t<10000:
            name = directory + '/' + train_file + "000" + str(t) + ".csv"
        elif t<100000:
            name = directory + '/' + train_file + "00" + str(t) + ".csv"
        elif t<1000000:
            name = directory + '/' + train_file+ "0" + str(t) + ".csv"
        else:
            name = directory + '/' + train_file + str(t) + ".csv"
        
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

def loadDataTotal(numTrain, ns, N, directory):
    x_hist = torch.zeros((ns,N,4))
    
    gamma = 1.4
    
    test_file = 'output'
    mesh_location = '../Mesh'
    volume_file = 'cell_volumes.npy'
    x_hist = loadDataSingle(test_file, mesh_location, directory, volume_file, ns, gamma)
    
    return x_hist

def saveOutput(cell_centroids, x_out, output_directory, filename, N, t):
    
    data_out = torch.cat((cell_centroids, x_out), dim=1)
    df = pd.DataFrame(data_out.detach().cpu().numpy())
    
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
    filename = 'output'
    directory = '../fullOrderModel/Outputs/19_3.5'
    x_test = loadDataTotal(ns, N, directory)
    
    model = architecture(lat_dim)
    model = torch.load('lat' + str(lat_dim) + '_model'+str(sol)).cpu().to(device)
    x_test = x_test.to(device)
    
    
    directory = '../2D_pooling_unpooling/'
    e1 = torch.load(directory + 'edge_index').to(device)
    e2 = torch.load(directory + 'e2').to(device)
    e3 = torch.load(directory + 'e3').to(device)
    e4 = torch.load(directory + 'e4').to(device)
    e5 = torch.load(directory + 'e5').to(device)
    
    s1 = torch.load(directory + 's1').to(device)
    s2 = torch.load(directory + 's2').to(device)
    s3 = torch.load(directory + 's3').to(device)
    s4 = torch.load(directory + 's4').to(device)
    
    u1 = torch.load(directory + 'unpool1').to(device)
    u2 = torch.load(directory + 'unpool2').to(device)
    u3 = torch.load(directory + 'unpool3').to(device)
    u4 = torch.load(directory + 'unpool4').to(device)
        
    pos1 = torch.load(directory + 'pos1').to(device)
    pos2 = torch.load(directory + 'pos2').to(device)
    pos3 = torch.load(directory + 'pos3').to(device)
    pos4 = torch.load(directory + 'pos4').to(device)
    pos5 = torch.load(directory + 'pos5').to(device)
    
    labels1 = torch.load(directory + 'labels1')
    labels2 = torch.load(directory + 'labels2')
    labels3 = torch.load(directory + 'labels3')
    labels4 = torch.load(directory + 'labels4')

    
    ai_54 = knn(pos5.cpu(), pos4.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_43 = knn(pos4.cpu(), pos3.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_32 = knn(pos3.cpu(), pos2.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_21 = knn(pos2.cpu(), pos1.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    
    x_out = torch.zeros((N,4))
    
    input_hist = torch.zeros(4*N,ns)
    pred_hist = torch.zeros(4*N,ns)
    
    lat_hist = torch.zeros((ns,lat_dim))
    
    
    for i in range(ns):
        inputs = torch.zeros((1,N,4)).to(device)
        x_target = torch.zeros((1,N,4)).to(device)
        
        inputs[0,:,:] = x_test[i,:,:]
        x_target[0,:,:] = x_test[i,:,:]
        input_hist[:,i] = inputs[0,:,:].flatten()
        
        x_lat = model.encode(inputs,batchSize,e1,e2,e3,e4,e5,s1,s2,s3,s4,labels1,labels2,labels3,labels4).detach()
        
        lat_hist[i,:] = x_lat.flatten()[:]
        
        x_pred = model.decode(x_lat,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5, ai_54, ai_43, ai_32, ai_21).detach()
        
        pred_hist[:,i] = x_pred[0,:,:].flatten()
        
        
    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

    print('lat_dim: ', lat_dim, ', sol: ', sol, ', error: ', full_rel_error)
