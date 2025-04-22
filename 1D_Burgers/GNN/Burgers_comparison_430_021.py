import sys
import torch

if __name__=="__main__":
    M = int(sys.argv[1])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load ground truth
    w_truth = torch.load('../fullOrderModel/w_test_430_021')
    N = w_truth.size()[0]
    nt = w_truth.size()[1]
    batchSize = 1

    # load ROM-predicted solution
    w_pred = torch.zeros((N,nt))
    w_pred = torch.load('lat'+str(M)+'_430_021')

    input_hist = w_truth[:,:]
    pred_hist = w_pred[:,:]

    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist[:-30,:nt]-pred_hist[:,:nt], dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist[:-30,:nt], dim=0).flatten())
    print('lat: ', M, ', full_rel_error: ', full_rel_error)
