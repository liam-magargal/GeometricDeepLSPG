import sys
import pandas as pd
import time as time_mod
import torch
import torch.nn as nn

    
class architecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv1d(1,8,kernel_size=25,stride=2,padding=12)
        nn.init.xavier_uniform_(self.layer1.weight.data)
        nn.init.zeros_(self.layer1.bias.data)
        
        self.layer2 = torch.nn.Conv1d(8,16,kernel_size=25,stride=4,padding=12)
        nn.init.xavier_uniform_(self.layer2.weight.data)
        nn.init.zeros_(self.layer2.bias.data)
        
        self.layer3 = torch.nn.Conv1d(16,32,kernel_size=25,stride=4,padding=12)
        nn.init.xavier_uniform_(self.layer3.weight.data)
        nn.init.zeros_(self.layer3.bias.data)
        
        self.layer4 = torch.nn.Conv1d(32,64,kernel_size=25,stride=4,padding=12)
        nn.init.xavier_uniform_(self.layer4.weight.data)
        nn.init.zeros_(self.layer4.bias.data)
        
        
        self.layer5 = torch.nn.Linear(128,5)
        nn.init.xavier_uniform_(self.layer5.weight.data)
        nn.init.zeros_(self.layer5.bias.data)
        
        self.layer6 = torch.nn.Linear(5,128)
        nn.init.xavier_uniform_(self.layer6.weight.data)
        nn.init.zeros_(self.layer6.bias.data)
        
        self.layer7 = torch.nn.ConvTranspose1d(64,32,kernel_size=25,stride=4,padding=11,output_padding=1)
        nn.init.xavier_uniform_(self.layer7.weight.data)
        nn.init.zeros_(self.layer7.bias.data)
        
        self.layer8 = torch.nn.ConvTranspose1d(32,16,kernel_size=25,stride=4,padding=11,output_padding=1)
        nn.init.xavier_uniform_(self.layer8.weight.data)
        nn.init.zeros_(self.layer8.bias.data)
        
        self.layer9 = torch.nn.ConvTranspose1d(16,8,kernel_size=25,stride=4,padding=11,output_padding=1)
        nn.init.xavier_uniform_(self.layer9.weight.data)
        nn.init.zeros_(self.layer9.bias.data)
        
        self.layer10 = torch.nn.ConvTranspose1d(8,1,kernel_size=25,stride=2,padding=12,output_padding=1)
        nn.init.xavier_uniform_(self.layer10.weight.data)
        nn.init.zeros_(self.layer10.bias.data)
        
        
        
    def encode(self,x,batchSize):
        x = (x - 1.) / 7.5
        x = torch.nn.functional.elu(self.layer1(x))
        x = torch.nn.functional.elu(self.layer2(x))
        x = torch.nn.functional.elu(self.layer3(x))
        x = torch.nn.functional.elu(self.layer4(x))
        x = x.reshape((batchSize,2*64))
        x = torch.nn.functional.elu(self.layer5(x))
        
        return x
    
    def decode(self,x,batchSize):
        x = torch.nn.functional.elu(self.layer6(x))
        x = x.reshape((batchSize,64,2))
        x = torch.nn.functional.elu(self.layer7(x))
        x = torch.nn.functional.elu(self.layer8(x))
        x = torch.nn.functional.elu(self.layer9(x))
        x = (self.layer10(x))
        x = x * 7.5 + 1.
        
        return x


if __name__=="__main__":
    N_lat = int(sys.argv[1])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = architecture()
    model = torch.load('lat'+str(N_lat))

    mu1 = 4.3
    mu2 = .021

    t_end = 35.
    dt = .07
    N = 256
    batchSize = 1
    nt = 501

    # set up domain and initial conditions
    x = torch.linspace(0,100,N)
    w0 = torch.ones((1,1,N))

    # encode initial conditions to latent representation
    w_hat_curr = model.encode(torch.ones((1,1,N)).to(device), 1)
    w_hat_next = w_hat_curr.clone()

    w_curr = model.decode(w_hat_curr, 1).flatten()
    w_next = w_curr.clone()

    model = model.to(device)
    w_hat_curr = w_hat_curr.to(device)
    w_hat_next = w_hat_next.to(device)
    w_curr = w_curr.to(device)
    w_next = w_next.to(device)

    v0 = x[2]-x[1]
    
    t_step = 0
    w_out = torch.zeros((N,nt))
    w_out_lat = torch.zeros((N_lat,nt))

    tol = 1e-12

    # initialize wall-clock times for each component of the ROM
    t1 = 0.
    t2 = 0.
    t3 = 0.
    t4 = 0.
    t5 = 0.
    t6 = 0.

    while t_step < nt:
        error = 1e6
        iter_count = 0
        
        if t_step>0:
            del w_hat_curr
            w_hat_curr = w_hat_next.clone()
            
        iter_count = 0
        
        while error > tol:
            ts = time_mod.time()
            
            # decode
            w_next = (model.decode(w_hat_next, 1).flatten()).clone()
            te = time_mod.time()
            t1 += (te-ts)

            r = torch.zeros((N)).to(device)
            J = torch.zeros((N,N)).to(device)
            
            ts = time_mod.time()
            
            # get residual and its Jacobian
            r[0] = w_next[0] - w_curr[0] - dt/v0*(.5*mu1*mu1) + dt/v0*(.5*w_next[0]*w_next[0]) - .02*torch.exp(mu2*x[0])*dt
            J[0,0] = 1 + dt/v0*w_next[0]

            for i in range(int(N)-1):
                cellID = i + 1
                r[cellID] = w_next[cellID] - w_curr[cellID] - dt/v0*(.5*(w_next[cellID-1]*w_next[cellID-1])) + dt/v0*(.5*(w_next[cellID]*w_next[cellID])) - .02*torch.exp(mu2*x[cellID])*dt
                
                J[cellID,cellID-1] = - dt/v0*w_next[cellID-1]
                J[cellID,cellID] = 1. + dt/v0*w_next[cellID]
            ####
            te = time_mod.time()
            t2 += (te-ts)

            # get Jacobian of decoder
            ts = time_mod.time()
            J_decoder_temp = torch.func.jacfwd(model.decode, argnums=0)(w_hat_next,1)
            J_decoder = J_decoder_temp[0,0,:,0,:]
            
            te = time_mod.time()
            t3 += (te-ts)
            
            # Set up normal form of LSPG step
            ts = time_mod.time()
            psi = torch.matmul(J, J_decoder)
            A = torch.matmul(psi.T,psi)
            B = -torch.matmul(psi.T,r)
            te = time_mod.time()
            t4 += (te-ts)

            # get error and check convergence
            ts = time_mod.time()
            error = torch.linalg.norm(B)

            if error<tol:
                break
            te = time_mod.time()
            t5 += (te-ts)
           
            ts = time_mod.time()
            if iter_count == 0:
                alpha = 1.
            elif iter_count%5==0:
                alpha = .95*alpha
            
            # get LSPG step
            p = torch.linalg.solve(A, B).detach()
            w_hat_next = w_hat_next.detach() + alpha*p.detach()
            te = time_mod.time()
            t6 += (te-ts)
            
            if iter_count==0 and t_step==0:
                tol = 1e-3*error
            
            iter_count += 1
            
        w_out[:,t_step] = w_curr.clone()
        w_out_lat[:,t_step] = w_hat_curr.clone()
        
        del w_curr
        w_curr = w_next.clone()
        
        t_step += 1

    print('lat: ', N_lat) 
    print('################')
    print('times: ')
    print('t1: ', t1)
    print('t2: ', t2)
    print('t3: ', t3)
    print('t4: ', t4)
    print('t5: ', t5)
    print('t6: ', t6)


    data_out = w_out.detach().numpy()
    df = pd.DataFrame(data_out)
    df.to_csv('lat'+str(N_lat)+'_430_021.csv')
    
    torch.save(w_out.detach().cpu(),'lat'+str(N_lat)+'_430_021')
