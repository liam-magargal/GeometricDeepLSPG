import sys
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
        
    w_truth = torch.load('../fullOrderModel/w_test_430_021')
    
    nt = w_truth.size()[1]
    
    ## append padded nodes to the domain
    x_leftBoundary = torch.zeros((30,nt))
    mu1 = 4.3
    x_leftBoundary = mu1*torch.ones((30,nt))
    w_truth = torch.cat((x_leftBoundary,w_truth),dim=0).to(device)
    ## end append

    N = w_truth.size()[0]
    batchSize = 1

    w_pred = torch.zeros((N-60,nt))
    
    # load hierarchy of graphs
    directory = '1D_pooling_unpooling/'
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
    
    labels1 = torch.load(directory + 'labels1').to(device)
    labels2 = torch.load(directory + 'labels2').to(device)
    labels3 = torch.load(directory + 'labels3').to(device)
    labels4 = torch.load(directory + 'labels4').to(device)

    ai_54 = knn(pos5.cpu(), pos4.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_43 = knn(pos4.cpu(), pos3.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_32 = knn(pos3.cpu(), pos2.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)
    ai_21 = knn(pos2.cpu(), pos1.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(device)

    # load trained architecture
    model = architecture(N_lat)
    model = torch.load('lat' + str(N_lat))

    w_lat_out = torch.zeros((N_lat,nt))
    rel_error = torch.zeros((nt))
    
    input_hist = torch.zeros(N-60,nt)
    pred_hist = torch.zeros(N-60,nt)

    # get reconstruction error
    for i in range(nt):
        inputs = w_truth[:,i].reshape((1,N,1)).clone()
        input_hist[:,i] = inputs[0,30:-30,0].clone().flatten()
        
        x_lat = model.encode((inputs),batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4, labels1, labels2, labels3, labels4)
        w_lat_out[:,i] = x_lat.flatten().cpu()
        x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21) 
        
        pred_hist[:,i] = x_pred.flatten()
        w_pred[:,i] = x_pred.flatten().clone()
        
        rel_error[i] = (torch.linalg.vector_norm(x_pred.flatten()-w_truth[30:-30,i].flatten())) / (torch.linalg.vector_norm(w_truth[30:-30,i].flatten()))


    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())
    print(full_rel_error)
    
    