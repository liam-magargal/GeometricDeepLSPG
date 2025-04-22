import torch

def getRandJ(r,J,dt,v0,x,w_next,w_curr,mu1,mu2,N):
    
    r[0] = w_next[0] - w_curr[0] - dt/v0*(.5*mu1*mu1) + dt/v0*(.5*w_next[0]*w_next[0]) - .02*torch.exp(mu2*x[0])*dt
    J[0,0] = 1 + dt/v0*w_next[0]
    for i in range(int(N)-1):
        cellID = i + 1
        
        r[cellID] = w_next[cellID] - w_curr[cellID] - dt/v0*(.5*(w_next[cellID-1]*w_next[cellID-1])) + dt/v0*(.5*(w_next[cellID]*w_next[cellID])) - .02*torch.exp(mu2*x[cellID])*dt
        
        J[cellID,cellID-1] = - dt/v0*w_next[cellID-1]
        J[cellID,cellID] = 1. + dt/v0*w_next[cellID]
    
    return r, J


def getSolution(i,j,N,dt,nt):
    print('Getting solution: (', i, ', ', j, ')')
    mu1 = 4.25 + (1.25/9.)*i
    mu2 = .015 + (.015/7.)*j

    x_left = 0
    x_right = 100 + (N-256) / (256-1) * 100
    x = torch.linspace(x_left,x_right,N)
    
    w_curr = torch.ones((N))
    w_next = torch.ones((N))
    r = torch.zeros((N))
    J = torch.zeros((N,N))
    
    v0 = x[2]-x[1]
    tol = 1e-4

    w_hist = torch.zeros((N,nt))

    t_count = 0
    out_count = 0
    
    device = torch.device('cpu')
    r = r.to(device)
    J = J.to(device)
    w_curr = w_curr.to(device)
    w_next = w_next.to(device)
    x = x.to(device)
    
    
    while t_count < nt:
        error = 1e6
        while error > tol:
            r, J = getRandJ(r,J,dt,v0,x,w_next,w_curr,mu1,mu2,N)
            error = torch.linalg.norm(r)
            w_next = w_next - torch.linalg.solve(J,r)
            
        
        w_hist[:,out_count] = w_curr.clone()
        out_count += 1

        w_curr = w_next.clone()
        t_count += 1
        
    return w_hist.cpu()


dt = .07
N = 256 + 30

ni = 10
nj = 8

nt = 501
w_train = torch.zeros((N,nt*(ni)*(nj)))

sol_count = 0
for i in range(ni):
    for j in range(nj):
        w_train[:,nt*sol_count:nt*(sol_count+1)] = getSolution(i,j,N,dt,nt)
        sol_count += 1


torch.save(w_train, 'w_train')

