import torch
import sys

if __name__=="__main__":
    M = int(sys.argv[1])

    # Load ground truth
    w_truth = torch.load('/w_test_515_0285')
    N = w_truth.size()[0]
    nt = w_truth.size()[1]
    
    w_pred = torch.zeros((N,nt))
    x_lat_hist = torch.zeros((M,nt))
    
    # Load POD modes
    U = torch.load('U_Burgers')
    phi = U[:,:M]
    
    # 'Approximation matrix': compresses and reconstructs ground truth solution
    A = torch.matmul(torch.transpose(phi,1,0),phi)
    rel_error = torch.zeros((nt))

    input_hist = torch.zeros(N,nt)
    pred_hist = torch.zeros(N,nt)

    # compute projection error
    for i in range(nt):
        inputs = w_truth[:,i].reshape((N,1)).clone()
    
        input_hist[:,i] = inputs[:,0].clone()
    
        x_lat = torch.matmul(phi.T,inputs)
        x_lat_hist[:,i] = x_lat.flatten().clone()
        
        x_pred = torch.matmul(phi,x_lat)
    
        pred_hist[:,i] = x_pred[:,0]
    
        w_pred[:,i] = x_pred.flatten().clone()

        rel_error[i] = (torch.linalg.vector_norm(x_pred.flatten()-w_truth[:,i].reshape((N,1)).flatten())) / (torch.linalg.vector_norm(w_truth[:,i].reshape((N,1)).flatten()))

    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

    print(full_rel_error)