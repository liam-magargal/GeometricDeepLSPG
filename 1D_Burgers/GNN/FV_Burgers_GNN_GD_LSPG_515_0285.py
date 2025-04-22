import sys
import pandas as pd
import time as time_mod
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
    

if __name__=="__main__":
    N_lat = int(sys.argv[1])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load trained model
    model = architecture(N_lat)
    model = torch.load('lat' + str(N_lat)).cpu() 
    
    # load hierarchy of graphs
    directory = '../1D_pooling_unpooling/'
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

    ai_54 = knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_43 = knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_32 = knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_21 = knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)

    labels1 = torch.load(directory + 'labels1')
    labels2 = torch.load(directory + 'labels2')
    labels3 = torch.load(directory + 'labels3')
    labels4 = torch.load(directory + 'labels4')

    mu1 = 5.15
    mu2 = .0285
    
    t_end = 35.
    dt = .07
    N = 256
    batchSize = 1
    nt = 501

    # initialize domain
    x = torch.linspace(0,100,N)
    x_leftBoundary = torch.zeros((30,nt))
    x_leftBoundary = mu1*torch.ones((30,nt))

    w_out = torch.zeros((N,nt))
    w_out_lat = torch.zeros((N_lat,nt))
    
    # encode initial conditions to latent space
    w_hat_curr = model.encode(torch.ones((1,N+60,1)), batchSize, e1,e2,e3,e4,e5, s1,s2,s3,s4, labels1,labels2,labels3,labels4).clone()
    w_curr = (model.decode(w_hat_curr,batchSize,e1,e2,e3,e4,e5, u1, u2, u3, u4,pos1,pos2,pos3,pos4,pos5,ai_54,ai_43,ai_32,ai_21).flatten()).clone()

    w_hat_next = w_hat_curr.clone()
    w_next = w_curr.clone()

    # move tensors to device
    model = model.to(device)
    w_hat_curr = w_hat_curr.to(device)
    w_hat_next = w_hat_next.to(device)
    w_curr = w_curr.to(device)
    w_next = w_next.to(device)

    e1 = e1.to(device)
    e2 = e2.to(device)
    e3 = e3.to(device)
    e4 = e4.to(device)
    e5 = e5.to(device)

    u1 = u1.to(device)
    u2 = u2.to(device)
    u3 = u3.to(device)
    u4 = u4.to(device)

    s1 = s1.to(device)
    s2 = s2.to(device)
    s3 = s3.to(device)
    s4 = s4.to(device)

    pos1 = pos1.to(device)
    pos2 = pos2.to(device)
    pos3 = pos3.to(device)
    pos4 = pos4.to(device)
    pos5 = pos5.to(device)

    ai_54 = ai_54.to(device)
    ai_43 = ai_43.to(device)
    ai_32 = ai_32.to(device)
    ai_21 = ai_21.to(device)


    r = torch.zeros((N))
    J = torch.zeros((N,N))
    f = torch.zeros((N))
    flux = torch.zeros((N))
    v0 = x[2]-x[1]
    tol = 1e-5

    # initialize wall-clock times
    t1 = 0.
    t2 = 0.
    t3 = 0.
    t4 = 0.
    t5 = 0.
    t6 = 0.

    t_step = 0

    tol = 1e-12

    while t_step < nt:
        print(t_step)
        error1 = 1e6
        iter_count = 0
        
        if t_step>0:
            del w_hat_curr
            w_hat_curr = w_hat_next.clone()
            w_curr = w_next.clone()
            
        alpha = 1.
            
        iter_count = 0
        while True:
            ts = time_mod.time()
            w_next = (model.decode(w_hat_next,batchSize,e1,e2,e3,e4,e5, u1, u2, u3, u4,pos1,pos2,pos3,pos4,pos5,ai_54,ai_43,ai_32,ai_21).flatten()).clone()
            te = time_mod.time()
            t1 += (te-ts)
            
            ts = time_mod.time()
            r = torch.zeros((N)).to(device)
            J = torch.zeros((N,N)).to(device)

            # get residual and its Jacobian            
            r[0] = w_next[0] - w_curr[0] - dt/v0*(.5*mu1*mu1) + dt/v0*(.5*w_next[0]*w_next[0]) - .02*torch.exp(mu2*x[0])*dt
            J[0,0] = 1 + dt/v0*w_next[0]

            for i in range(N-1):
                cellID = i + 1
                
                r[cellID] = w_next[cellID] - w_curr[cellID] - dt/v0*(.5*(w_next[cellID-1]*w_next[cellID-1])) + dt/v0*(.5*(w_next[cellID]*w_next[cellID])) - .02*torch.exp(mu2*x[cellID])*dt
                
                J[cellID,cellID-1] = - dt/v0*w_next[cellID-1]
                J[cellID,cellID] = 1. + dt/v0*w_next[cellID]
                
            te = time_mod.time()
            t2 += (te-ts)
            
            # get Jacobian of the decoder
            ts = time_mod.time()
            J_decoder = torch.func.jacfwd(model.decode, argnums=0)(w_hat_next,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5,ai_54,ai_43,ai_32,ai_21)
            J_decoder = J_decoder[0,:,0,0,:]
            
            te = time_mod.time()
            t3 += (te-ts)
            
            # obtain normal form of Gauss-Newton
            ts = time_mod.time()
            psi = torch.matmul(J, J_decoder)
            A = torch.matmul(psi.T, psi)
            B = -torch.matmul(psi.T, r)
            te = time_mod.time()
            t4 += (te-ts)
            
            ts = time_mod.time()
            error = torch.linalg.norm(B)
            
            if error < tol:
                break

            te = time_mod.time()
            t5 += (te-ts)

            ts = time_mod.time()
            if iter_count == 0:
                alpha = .5
            elif iter_count%10==0:
                alpha = .9*alpha
            
            # LSPG step
            p = torch.linalg.solve(A, B).detach()
            w_hat_next = w_hat_next.detach() + alpha*p.detach()
            te = time_mod.time()
            t6 += (te-ts)

            if iter_count==0 and t_step==0:
                tol = 1e-3*error
                
            iter_count += 1
       
            if iter_count>50:
                print('DID NOT CONVERGE')
                sys.exit()
        
        w_out[:,t_step] = w_curr.clone()
        w_out_lat[:,t_step] = w_hat_curr.clone()
        
        del w_curr
        w_curr = w_next.clone()

        t_step += 1


    data_out = w_out.detach().numpy()
    df = pd.DataFrame(data_out)
    df.to_csv('lat'+str(N_lat)+'_515_0285.csv')

    torch.save(w_out.detach().cpu(),'lat'+str(N_lat)+'_515_0285')
    
    print('#######################')
    print('times:')
    print('t1: ', t1)
    print('t2: ', t2)
    print('t3: ', t3)
    print('t4: ', t4)
    print('t5: ', t5)
    print('t6: ', t6)
