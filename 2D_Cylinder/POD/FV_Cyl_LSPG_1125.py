import numpy as np
import sys
import os
import pandas as pd
import torch
from numba import njit, jit, float64, int64
import numba as nb
import shutil

def genDomain(cell_centroids,gamma,u_1):
    N = np.shape(cell_centroids)[0]

    p1 = 1.
    rho1 = 1.
    u1 = u_1*np.sqrt(1.4)
    v1 = 0.

    rho = rho1*np.ones((N),dtype=np.float64)
    vx = u1*np.ones((N),dtype=np.float64)
    vy = v1*np.ones((N),dtype=np.float64)
    P = p1*np.ones((N),dtype=np.float64)

    return rho, vx, vy, P


@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:],int64) )
def fromState(x_curr, N):
    Mass = x_curr[:N]
    Momx = x_curr[N:2*N]
    Momy = x_curr[2*N:3*N]
    Energy = x_curr[3*N:]
    
    return Mass, Momx, Momy, Energy

@jit((float64[:])(float64[:],float64[:],float64[:],float64[:],int64) )
def toState(Mass, Momx, Momy, Energy, N):
    x = np.zeros((4*N),dtype=np.float64)
    x[:N] = Mass
    x[N:2*N] = Momx
    x[2*N:3*N] = Momy
    x[3*N:] = Energy
    
    return x

@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]) )
def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    
    return rho, u, v, P, E, H

@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]) )
def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = (1/(gamma-1)*P/rho + .5*(u*u + v*v))
    H = ((gamma) / (gamma-1) * P / rho + .5*(u*u + v*v))
    Energy = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H

@njit((float64)(float64,float64))
def np_max(val1, val2):
    if val1>val2:
        return val1
    else:
        return val2

@njit((float64)(float64,float64))
def np_min(val1, val2):
    if val2>val1:
        return val1
    else:
        return val2


@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:],int64))
def getResidual(x_next, Mass_n, Momx_n, Momy_n, Energy_n,N):
    f = np.zeros((4*N))
    f[:N] = x_next[:N] - Mass_n
    f[N:2*N] = x_next[N:2*N] - Momx_n
    f[2*N:3*N] = x_next[2*N:3*N] - Momy_n
    f[3*N:] = x_next[3*N:] - Energy_n
    
    return f
    
@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64,float64))
def getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1):
    edge_index = edge_data[:,:2]
    n_hat = edge_data[:,4:6]
    faceArea = edge_data[:,6]

    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)

    n_edges = np.shape(edge_index)[0]

    N = np.shape(cell_centroids)[0]

    vel_Mass = np.zeros((N))
    vel_Momx = np.zeros((N))
    vel_Momy = np.zeros((N))
    vel_Energy = np.zeros((N))

    ## using RHLL flux (Nishikawa 2008)
    for i in range(n_edges):
        cell0 = int(edge_index[i,0])
        cell1 = int(edge_index[i,1])

        rho_in = 1.
        vx_in = u_1*np.sqrt(1.4)

        if cell1==-1:
            # left boundary (inflow)
            rho_c1 = rho_in
            vx_c1 = vx_in
            vy_c1 = 0.
            P_c1 = 1.

        elif cell1==-2:
            # right boundary (outflow)
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
            
        elif cell1==-3:
            # top or bottom boundary (outflow)
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
            
        elif cell1==-4:
            # cylinder boundary (slip)
            rho_c1 = rho_c[cell0]
            k = 2.0*(vx_c[cell0]*n_hat[i,0] + vy_c[cell0]*n_hat[i,1])
            vx_c1 = (vx_c[cell0] - k*n_hat[i,0])
            vy_c1 = (vy_c[cell0] - k*n_hat[i,1])
            P_c1 = P_c[cell0]
            
        else:
            # normal cell
            rho_c1 = rho_c[cell1]
            vx_c1 = vx_c[cell1]
            vy_c1 = vy_c[cell1]
            P_c1 = P_c[cell1]
            
        
        H_c1 = (gamma) / (gamma-1) * P_c1 / rho_c1 + .5*(vx_c1*vx_c1 + vy_c1*vy_c1)
        
        # Get direction for Rotated Roe flux (from Ren 2003)
        n1 = np.zeros((2))
        n2 = np.zeros((2))
        n1_tilde = np.zeros((2))
        n2_tilde = np.zeros((2))
        
        epsilon = 1e-12*1. #1.0 is the reference velocity in this case (its a rough estimation of the maximum initial velocity)
        if np.sqrt((vx_c[cell0]-vx_c1)*(vx_c[cell0]-vx_c1) + (vy_c[cell0]-vy_c1)*(vy_c[cell0]-vy_c1)) <= epsilon:
            n1[0] = n_hat[i,0]
            n1[1] = n_hat[i,1]
        else:
            n1[0] = (vx_c1-vx_c[cell0]) / np.sqrt((vx_c1-vx_c[cell0])*(vx_c1-vx_c[cell0]) + (vy_c1-vy_c[cell0])*(vy_c1-vy_c[cell0]))
            n1[1] = (vy_c1-vy_c[cell0]) / np.sqrt((vx_c1-vx_c[cell0])*(vx_c1-vx_c[cell0]) + (vy_c1-vy_c[cell0])*(vy_c1-vy_c[cell0]))
        n2[0] = -n1[1]
        n2[1] = n1[0]
        
        alpha1 = n_hat[i,0] * n1[0] + n_hat[i,1] * n1[1]
        alpha2 = n_hat[i,0] * n2[0] + n_hat[i,1] * n2[1]
        
        if alpha1 < 0:
            n1_tilde[0] = -1*n1[0]
            n1_tilde[1] = -1*n1[1]
        else:
            n1_tilde[0] = n1[0]
            n1_tilde[1] = n1[1]
        if alpha2 < 0:
            n2_tilde[0] = -1*n2[0]
            n2_tilde[1] = -1*n2[1]
        else:
            n2_tilde[0] = n2[0]
            n2_tilde[1] = n2[1]

        alpha1 = n_hat[i,0] * n1_tilde[0] + n_hat[i,1] * n1_tilde[1]
        alpha2 = n_hat[i,0] * n2_tilde[0] + n_hat[i,1] * n2_tilde[1]

        # align cell state values with their normal vectors
        # hold onto these cartesian values to reassign in the future
        vx_cart0 = (vx_c[cell0])
        vy_cart0 = (vy_c[cell0])
        vx_cart1 = vx_c1
        vy_cart1 = vy_c1

        # rotate velocity components to align with cell face
        vx_c[cell0] = n_hat[i,0]*vx_cart0 + n_hat[i,1]*vy_cart0
        vy_c[cell0] = - n_hat[i,1]*vx_cart0 + n_hat[i,0]*vy_cart0
        vx_c1 = n_hat[i,0]*vx_cart1 + n_hat[i,1]*vy_cart1
        vy_c1 = - n_hat[i,1]*vx_cart1 + n_hat[i,0]*vy_cart1
        
        # get nominal cell-centered fluxes based on rotated states
        # compute fluxes (based on cell 0)
        flux_Mass_0 = rho_c[cell0]*vx_c[cell0]
        flux_Momx_0 = rho_c[cell0]*vx_c[cell0]*vx_c[cell0] + P_c[cell0]
        flux_Momy_0 = rho_c[cell0]*vx_c[cell0]*vy_c[cell0]
        flux_Energy_0 = rho_c[cell0]*vx_c[cell0]*H_c[cell0]

        # compute fluxes (based on cell 1)
        flux_Mass_1 = rho_c1*vx_c1
        flux_Momx_1 = rho_c1*vx_c1*vx_c1 + P_c1
        flux_Momy_1 = rho_c1*vx_c1*vy_c1
        flux_Energy_1 = rho_c1*vx_c1*H_c1

        # rotate fluxes back
        flux_Momx_0_r = flux_Momx_0
        flux_Momy_0_r = flux_Momy_0
        flux_Momx_0 = flux_Momx_0_r * n_hat[i,0] - flux_Momy_0_r * n_hat[i,1]
        flux_Momy_0 = flux_Momx_0_r * n_hat[i,1] + flux_Momy_0_r * n_hat[i,0]

        flux_Momx_1_r = flux_Momx_1
        flux_Momy_1_r = flux_Momy_1
        flux_Momx_1 = flux_Momx_1_r * n_hat[i,0] - flux_Momy_1_r * n_hat[i,1]
        flux_Momy_1 = flux_Momx_1_r * n_hat[i,1] + flux_Momy_1_r * n_hat[i,0]

        # rotate velocities back to cartesian values
        vx_c[cell0] = vx_cart0
        vy_c[cell0] = vy_cart0
        vx_c1 = vx_cart1
        vy_c1 = vy_cart1

        # Get Roe-averaged state at boundary
        rho_hat = np.sqrt(rho_c[cell0]*rho_c1)
        u_hat = (vx_c[cell0]*np.sqrt(rho_c[cell0]) + vx_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        v_hat = (vy_c[cell0]*np.sqrt(rho_c[cell0]) + vy_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        H_hat = (H_c[cell0]*np.sqrt(rho_c[cell0]) + H_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        c_hat = np.sqrt((gamma-1)*(H_hat - (u_hat*u_hat + v_hat*v_hat)/2))
        
        # get SRp and SLm
        qn_L = vx_c[cell0] * n1_tilde[0] + vy_c[cell0] * n1_tilde[1]
        qn_R = vx_c1 * n1_tilde[0] + vy_c1 * n1_tilde[1]
        c_L = np.sqrt((gamma-1)*(H_c[cell0] - (vx_c[cell0]*vx_c[cell0] + vy_c[cell0]*vy_c[cell0])/2))
        c_R = np.sqrt((gamma-1)*(H_c1 - (vx_c1*vx_c1 + vy_c1*vy_c1)/2))
        qn_hat_n1 = u_hat * n1_tilde[0] + v_hat * n1_tilde[1]

        # get SL and SR
        SL = np_min(qn_L-c_L, qn_hat_n1-c_hat)
        SR = np_max(qn_R+c_R, qn_hat_n1+c_hat)

        SRp = np_max(0,SR)
        SLm = np_min(0,SL)

        flux_Mass = (SRp*flux_Mass_0 - SLm*flux_Mass_1) / (SRp - SLm)
        flux_Momx = (SRp*flux_Momx_0 - SLm*flux_Momx_1) / (SRp - SLm)
        flux_Momy = (SRp*flux_Momy_0 - SLm*flux_Momy_1) / (SRp - SLm)
        flux_Energy = (SRp*flux_Energy_0 - SLm*flux_Energy_1) / (SRp - SLm)

        # get eigenvalues (based on n2_tilde)
        qn_hat_n2 = u_hat * n2_tilde[0] + v_hat * n2_tilde[1]
        lambda1_n2 = qn_hat_n2 - c_hat
        lambda2_n2 = qn_hat_n2
        lambda3_n2 = qn_hat_n2 + c_hat
        lambda4_n2 = qn_hat_n2

        # get lambda_star terms (for all k)
        delta = .2
        if np.abs(lambda1_n2) >= delta:
            lambda1_n2_star = np.abs(lambda1_n2)
        else:
            lambda1_n2_star = 1/(2*delta) * (np.abs(lambda1_n2)*np.abs(lambda1_n2) + delta*delta)

        if np.abs(lambda2_n2) >= delta:
            lambda2_n2_star = np.abs(lambda2_n2)
        else:
            lambda2_n2_star = 1/(2*delta) * (np.abs(lambda2_n2)*np.abs(lambda2_n2) + delta*delta)


        if np.abs(lambda3_n2) >= delta:
            lambda3_n2_star = np.abs(lambda3_n2)
        else:
            lambda3_n2_star = 1/(2*delta) * (np.abs(lambda3_n2)*np.abs(lambda3_n2) + delta*delta)

        if np.abs(lambda4_n2) >= delta:
            lambda4_n2_star = np.abs(lambda4_n2)
        else:
            lambda4_n2_star = 1/(2*delta) * (np.abs(lambda4_n2)*np.abs(lambda4_n2) + delta*delta)
            
        s_hat_1_RHLL = alpha2*lambda1_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda1_n2 + 2*alpha1*SRp*SLm)
        s_hat_2_RHLL = alpha2*lambda2_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda2_n2 + 2*alpha1*SRp*SLm)
        s_hat_3_RHLL = alpha2*lambda3_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda3_n2 + 2*alpha1*SRp*SLm)
        s_hat_4_RHLL = alpha2*lambda4_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda4_n2 + 2*alpha1*SRp*SLm)


        drho = rho_c1 - rho_c[cell0]
        dP = P_c1 - P_c[cell0]

        # the diffusion term here is computed wrt n2_tilde
        dq_n = (vx_c1 - vx_c[cell0]) * n2_tilde[0] + (vy_c1 - vy_c[cell0]) * n2_tilde[1]
        dq_t = (vx_c1 - vx_c[cell0]) * (-1*n2_tilde[1]) + (vy_c1 - vy_c[cell0]) * n2_tilde[0]

        w1_n2 = (dP - rho_hat*c_hat*dq_n)/(2*c_hat*c_hat)
        w2_n2 = drho - dP/(c_hat*c_hat)
        w3_n2 = (dP + rho_hat*c_hat*dq_n)/(2*c_hat*c_hat)
        w4_n2 = rho_hat*dq_t

        qn_hat = u_hat*n2_tilde[0] + v_hat*n2_tilde[1]
        qt_hat = u_hat*(-1*n2_tilde[1]) + v_hat*n2_tilde[0]
        
        flux_Mass = flux_Mass - 1/2 * (s_hat_1_RHLL * w1_n2 * 1 + s_hat_2_RHLL * w2_n2 * 1 + s_hat_3_RHLL * w3_n2 * 1)
        flux_Momx = flux_Momx - 1/2 * (s_hat_1_RHLL * w1_n2 * (u_hat-c_hat*n2_tilde[0]) + s_hat_2_RHLL * w2_n2 * (u_hat) + s_hat_3_RHLL * w3_n2 * (u_hat+c_hat*n2_tilde[0]) + s_hat_4_RHLL * w4_n2 * (-n2_tilde[1]))
        flux_Momy = flux_Momy - 1/2 * (s_hat_1_RHLL * w1_n2 * (v_hat-c_hat*n2_tilde[1]) + s_hat_2_RHLL * w2_n2 * (v_hat) + s_hat_3_RHLL * w3_n2 * (v_hat+c_hat*n2_tilde[1]) + s_hat_4_RHLL * w4_n2 * (n2_tilde[0]))
        flux_Energy = flux_Energy - 1/2 * (s_hat_1_RHLL * w1_n2 * (H_hat - qn_hat*c_hat) + s_hat_2_RHLL * w2_n2 * (.5*(u_hat*u_hat+v_hat*v_hat)) + s_hat_3_RHLL * w3_n2 * (H_hat+qn_hat*c_hat) + s_hat_4_RHLL * w4_n2 * (qt_hat))

        vel_Mass[cell0] = vel_Mass[cell0] - flux_Mass * faceArea[i] / cell_volumes[cell0]
        vel_Momx[cell0] = vel_Momx[cell0] - flux_Momx * faceArea[i] / cell_volumes[cell0]
        vel_Momy[cell0] = vel_Momy[cell0] - flux_Momy * faceArea[i] / cell_volumes[cell0]
        vel_Energy[cell0] = vel_Energy[cell0] - flux_Energy * faceArea[i] / cell_volumes[cell0]
        

    return vel_Mass, vel_Momx, vel_Momy, vel_Energy


@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64,float64))
def updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1):

    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
    N = np.shape(cell_centroids)[0]

    vel_Mass_c, vel_Momx_c, vel_Momy_c, vel_Energy_c = getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)
    
    Mass_n = Mass_c + dt*vel_Mass_c
    Momx_n = Momx_c + dt*vel_Momx_c
    Momy_n = Momy_c + dt*vel_Momy_c
    Energy_n = Energy_c + dt*vel_Energy_c

    return Mass_n, Momx_n, Momy_n, Energy_n
    
    
def saveOutput(cell_centroids, rho, vx, vy, P, output_directory, filename, N, t):
    
    cell_centroids = cell_centroids.flatten(order='F')
    
    data_out = np.concatenate((cell_centroids, rho, vx, vy, P), axis=None)
    
    data_out = np.reshape(data_out, (N,6), order='F')
    df = pd.DataFrame(data_out)
    
    if t<10:
        name = output_directory + filename + "000000" + str(t) + ".csv"
    elif t<100:
        name = output_directory + filename + "00000" + str(t) + ".csv"
    elif t<1000:
        name = output_directory + filename + "0000" + str(t) + ".csv"
    elif t<10000:
        name = output_directory + filename + "000" + str(t) + ".csv"
    elif t<100000:
        name = output_directory + filename + "00" + str(t) + ".csv"
    elif t<1000000:
        name = output_directory + filename + "0" + str(t) + ".csv"
    else:
        name = output_directory + filename + str(t) + ".csv"
           
    if os.path.exists(name):
        os.remove(name)
    
    df.to_csv(name)
    
    del cell_centroids, data_out
    
    return 0


if __name__=="__main__":
    lat_dim = int(sys.argv[1])

    ## Import mesh data
    mesh_file = '../Mesh/'
    edge_index = np.load(mesh_file + 'edge_index.npy')
    cell_centroids = np.load(mesh_file + 'cell_centroids.npy')
    n_hat = np.load(mesh_file + 'n_hat.npy')
    faceArea = np.load(mesh_file + 'faceArea.npy') 
    cell_volumes = np.load(mesh_file + 'cell_volumes.npy')

                            
    cell_centroids = np.float64(cell_centroids)
    n_hat = np.float64(n_hat)
    faceArea = np.float64(faceArea)
    cell_volumes = np.float64(cell_volumes)

    edge_data = np.concatenate((edge_index, n_hat), axis=1)
    edge_data = np.concatenate((edge_data, np.reshape(faceArea, (np.size(faceArea),1))), axis=1)
    edge_data = edge_data[edge_data[:, 0].argsort()]


    u_1                    = 1.125
    output_directory       = 'lat'+str(lat_dim) + '_mach_1125/'
    filename               = 'output'
    N                      = np.shape(cell_centroids)[0]
    gamma                  = 1.4
    courant_fac            = 0.4
    t                      = 0
    dt                     = 1e-3
    tEnd                   = 1.

    if os.path.exists((output_directory)):
        shutil.rmtree(output_directory)
    
    os.mkdir(output_directory)
    
    x_curr = torch.zeros((4*N),dtype=torch.float32)

    rho, vx, vy, P = genDomain(cell_centroids,gamma,u_1)
    Mass, Mom_x, Mom_y, Energy, E, H = getConserved(rho, vx, vy, P, gamma, cell_volumes)

    x_curr[:N] = torch.tensor(Mass)
    x_curr[N:2*N] = torch.tensor(Mom_x)
    x_curr[2*N:3*N] = torch.tensor(Mom_y)
    x_curr[3*N:] = torch.tensor(Energy)


    U = torch.tensor((pd.read_csv('U.csv')).to_numpy(),dtype=torch.float32)
    phi = U[:,1:(lat_dim+1)]
    device = torch.device('cpu')
    N_torch = torch.tensor(N)
    phi = phi.to(device)
    x_curr = x_curr.to(device)

    x_next = x_curr.clone()
    x_hat_next = torch.matmul(phi.T,x_next)
    x_hat_curr = x_hat_next.clone().to(device)

    x_curr = x_curr.detach().clone().type(torch.DoubleTensor).cpu().numpy()
    x_next = x_next.detach().clone().type(torch.DoubleTensor).cpu().numpy()

    edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    edge_index = edge_index.to(device)

    f = np.zeros((4*N))
    
    t_step = 0
    t_out = 0

    iter_out = 0
    for t_step in range(1001):
        del x_curr, x_hat_curr
        x_hat_curr = x_hat_next.clone()
        x_curr = torch.matmul(phi, x_hat_curr)

        Mass_c = x_curr[:N].detach().type(torch.DoubleTensor).cpu().numpy()
        Momx_c = x_curr[N:2*N].detach().type(torch.DoubleTensor).cpu().numpy()
        Momy_c = x_curr[2*N:3*N].detach().type(torch.DoubleTensor).cpu().numpy()
        Energy_c = x_curr[3*N:].detach().type(torch.DoubleTensor).cpu().numpy()

        rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
        void = saveOutput(cell_centroids, rho_c, vx_c, vy_c, P_c, output_directory, filename, N, t_out)
        t_out += 1

        Mass_n = np.zeros(N)
        Momx_n = np.zeros(N)
        Momy_n = np.zeros(N)
        Energy_n = np.zeros(N)
        
        Mass_n, Momx_n, Momy_n, Energy_n = updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)        

        tol = 1e-12
        error_LDM = 1e5
        iter_count = 0
        
        J_decoder = torch.zeros((4*N,lat_dim))
        
        while error_LDM>tol:
            # 'decode' to high dimensional approximation
            x_next_torch = torch.matmul(phi,x_hat_next)
            
            # get residual
            x_next[:N] = x_next_torch[:N].detach().cpu().numpy()
            x_next[N:2*N] = x_next_torch[N:2*N].detach().cpu().numpy()
            x_next[2*N:3*N] = x_next_torch[2*N:3*N].detach().cpu().numpy()
            x_next[3*N:] = x_next_torch[3*N:].detach().cpu().numpy()
            
            f = getResidual(x_next,Mass_n,Momx_n,Momy_n,Energy_n,N)
            
            # LSPG step
            B = torch.matmul(phi.T,torch.tensor(f).type(torch.FloatTensor).to(device))
            C = torch.matmul(phi.T,phi)
            p = -torch.linalg.solve(C,B)
            x_hat_next_old = x_hat_next.clone()
            
            # error/convergence check
            error_LDM = torch.linalg.norm(B)
            
            if iter_count==0 and t_step==0:
                tol = 1e-3*error_LDM
            
            if error_LDM<tol:
                break
            
            x_hat_next = x_hat_next_old + p
            if iter_count == 30:
                print('DID NOT CONVERGE')
                sys.exit()
            iter_count += 1
            
        t += dt
