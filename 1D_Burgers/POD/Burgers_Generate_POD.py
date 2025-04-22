import torch

N = 256
M = 10

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

w_hist_train = torch.load('w_train').to(device)
U, sigma, V = torch.linalg.svd(w_hist_train)
torch.save(U[:N,:M].cpu(), 'U_Burgers')
