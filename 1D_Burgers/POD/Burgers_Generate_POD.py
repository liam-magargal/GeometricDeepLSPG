import torch

N = 256
M = 10

device = torch.device('cpu')

w_hist_train = torch.load('../fullOrderModel/w_train').to(device)
U, sigma, V = torch.linalg.svd(w_hist_train[:N,:])
torch.save(U[:,:M].cpu(), 'U_Burgers')
