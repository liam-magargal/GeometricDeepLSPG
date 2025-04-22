import torch
import time as time_mod
import pandas as pd
import sys

if __name__=="__main__":
    M = int(sys.argv[1])

    N = 256
    nt = 501
    
    # Load POD modes
    U = torch.load('U_Burgers')
    phi = U[:,:M]

    r = torch.zeros((N))
    J = torch.zeros((N,N))
    
    mu1 = 4.3 #test parameter 1
    mu2 = .021 #test parameter 1

    t_end = 35.
    dt = .07
    N = 256
    nt = 501

    # set up domain and initial conditions
    x = torch.linspace(0,100,N)
    v0 = x[2]-x[1]
    w0 = torch.ones((N))
    w_curr = w0.clone()
    w_next = w_curr.clone()
    w_out = torch.zeros((N,nt))
    w_lat_out = torch.zeros((M,nt))

    # project initial conditions onto latent space
    w_hat_curr = torch.matmul(phi.T,w_curr)
    w_hat_next = w_hat_curr.clone()

    w_curr = torch.matmul(phi,w_hat_curr)
    w_next = w_curr.clone()

    w_out = torch.zeros((N,nt))


    tol = 1e-12

    # initialize total wall-clock times for each component of ROM
    t1 = 0.
    t2 = 0.
    t3 = 0.
    t4 = 0.
    t5 = 0.
    t6 = 0.


    for t in range(nt):
        
        if t>0:
            w_hat_curr = w_hat_next.clone()
            w_curr = torch.matmul(phi,w_hat_curr)
        
        error = 1e5
        iter_count = 0
        
        while error > tol:
            # Decode
            ts = time_mod.time()
            w_next = torch.matmul(phi,w_hat_next)
            te = time_mod.time()
            t1 += (te-ts)
            
            ts = time_mod.time()
            
            # Obtain residual and Jacobian
            r[0] = w_next[0] - w_curr[0] - dt/v0*(.5*mu1*mu1) + dt/v0*(.5*w_next[0]*w_next[0]) - .02*torch.exp(mu2*x[0])*dt
            J[0,0] = 1 + dt/v0*w_next[0]
            
            for i in range(int(N)-1):
                cellID = i + 1
            
                r[cellID] = w_next[cellID] - w_curr[cellID] - dt/v0*(.5*(w_next[cellID-1]*w_next[cellID-1])) + dt/v0*(.5*(w_next[cellID]*w_next[cellID])) - .02*torch.exp(mu2*x[cellID])*dt
            
                J[cellID,cellID-1] = - dt/v0*w_next[cellID-1]
                J[cellID,cellID] = 1. + dt/v0*w_next[cellID]
            te = time_mod.time()
            t2 += (te-ts)
            
            # Least-squares Petrov-Galerkin step
            ts = time_mod.time()
            A = torch.matmul(J,phi)
            B = torch.matmul(-A.T,r)
            C = torch.matmul(A.T,A)
            te = time_mod.time()
            t4 += (te-ts)
            
            ts = time_mod.time()
            p_k = torch.linalg.solve(C,B)
            w_hat_next = w_hat_next + p_k
            te = time_mod.time()
            t5 += (te-ts)
            
            # Convergence check
            ts = time_mod.time()
            error = torch.linalg.norm(B)
            if iter_count==0 and t==0:
                tol = 1e-4*error
            te = time_mod.time()
            t6 += (te-ts)

            iter_count += 1
        
        w_out[:,t] = w_curr
        w_lat_out[:,t] = w_hat_curr
        

    torch.save(w_out,'lat'+str(M)+'_430_021')

    data_out = w_out.detach().numpy()
    df = pd.DataFrame(data_out)

    df.to_csv('lat'+str(M)+'_430_021.csv')

    print('###############')
    print('times:')
    print('t1: ', t1)
    print('t2: ', t2)
    print('t3: ', t3)
    print('t4: ', t4)
    print('t5: ', t5)
    print('t6: ', t6)
