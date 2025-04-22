import torch
import torch_geometric

from torch_kmeans import KMeans
from torch_geometric.nn import radius_graph

def segment(N, Nc, edges, pos,x_left,x_right):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Compute adjacency matrix and graph laplacian
    A = torch_geometric.utils.to_dense_adj(edges).reshape((N,N)) 
    D = torch.diag(torch.sum(A,dim=1))
    L = D - A
    L = L.to(device)
    
    # Perform eigendecomposition
    l, v = torch.linalg.eig(L)
    l = l.cpu()
    v = v.cpu()

    # sort eigenvalues in increasing order
    l_sorted, indices = torch.sort(torch.abs(l),descending=False)

    U = torch.real(v[:,indices[1:Nc]])
    U = U.reshape((1,N,Nc-1))

    # Perform K-means clustering on the spectral features
    model_kmeans = KMeans(n_clusters=Nc,seed=123,init_method='k-means++',num_init=Nc)
    k_means_output = model_kmeans(U)
        
    labels = k_means_output.labels.flatten()
    
    S = torch.zeros((1,N,Nc))
    for i in range(N):
        j = labels[i]
        S[0,i,j] = 1.
    
    # Get assigment matrix
    summation = torch.sum(S,dim=1).reshape((1,1,Nc))
    unpool = S.transpose(1,2).clone()
    S = torch.div(S,summation)
   
    # Obtain positions of nodes in the next layer of the hierarchy
    pos2 = torch.matmul(S.transpose(1,2), pos).reshape((Nc,1))
    
    # Rescale positions
    rBound = torch.max(pos2[:,0])
    lBound = torch.min(pos2[:,0])

    pos2 = (pos2 - lBound*torch.ones((Nc,1))) / (rBound - lBound) * (x_right-x_left) + x_left*torch.ones((Nc,1))
    
    # Get edge index for next layer in the hierarchy of graphs
    edge_index2 = radius_graph(x=pos2,r=3.5/Nc*(x_right-x_left),loop=False)
    
    return S, unpool, edge_index2, pos2, labels



# Number of nodes at each layer in the hierarchy (user-specified)
n1 = 256 + 2*30
n2 = 64
n3 = 16
n4 = 4
n5 = 2

# Left/right ends of the padded domain (includes 30 nodes padded on both sides)
x_right = 100 + (n1-256 - 30) / (256-1) * 100
x_left = - (n1-256 - 30) / (256-1) * 100

# Positions and edge index of original input 1D mesh
pos = torch.linspace(x_left,x_right,n1)
edge_index = radius_graph(x=pos,r=3.5/n1*(x_right-x_left),loop=False)

# Obtain hierarchy of graphs
s1, unpool1, e2, pos2, labels1 = segment(n1,n2,edge_index,pos,x_left,x_right)
s2, unpool2, e3, pos3, labels2 = segment(n2,n3,e2,pos2,x_left,x_right)
s3, unpool3, e4, pos4, labels3 = segment(n3,n4,e3,pos3,x_left,x_right)
s4, unpool4, e5, pos5, labels4 = segment(n4,n5,e4,pos4,x_left,x_right)


# Save graphs to the output directory
directory = '1D_pooling_unpooling/'
torch.save(edge_index, directory + 'edge_index')
torch.save(e2, directory + 'e2')
torch.save(e3, directory + 'e3')
torch.save(e4, directory + 'e4')
torch.save(e5, directory + 'e5')

torch.save(s1, directory + 's1')
torch.save(s2, directory + 's2')
torch.save(s3, directory + 's3')
torch.save(s4, directory + 's4')

torch.save(unpool1, directory + 'unpool1')
torch.save(unpool2, directory + 'unpool2')
torch.save(unpool3, directory + 'unpool3')
torch.save(unpool4, directory + 'unpool4')

torch.save(labels1, directory + 'labels1')
torch.save(labels2, directory + 'labels2')
torch.save(labels3, directory + 'labels3')
torch.save(labels4, directory + 'labels4')

torch.save(pos, directory + 'pos1')
torch.save(pos2, directory + 'pos2')
torch.save(pos3, directory + 'pos3')
torch.save(pos4, directory + 'pos4')
torch.save(pos5, directory + 'pos5')


