import numpy as np
import sys
import os
import pandas as pd
import time
import torch
import torch.nn as nn
from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter
import shutil
from numba import njit, jit, float64, int64
import numba as nb


def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, assign_index: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    # This function is slightly modified from its original form in Pytorch-Geometric

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

        self.conv1 = torch.nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv1.weight.data)
        nn.init.zeros_(self.conv1.bias.data)

        self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv2.weight.data)
        nn.init.zeros_(self.conv2.bias.data)

        self.conv3 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv3.weight.data)
        nn.init.zeros_(self.conv3.bias.data)

        self.conv4 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=2,padding=2)
        nn.init.xavier_uniform_(self.conv4.weight.data)
        nn.init.zeros_(self.conv4.bias.data)

        self.linear1 = torch.nn.Linear(64*4*4,M)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.zeros_(self.linear1.bias.data)

        self.linear2 = torch.nn.Linear(M,64*4*4)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.zeros_(self.linear2.bias.data)

        self.conv5 = torch.nn.ConvTranspose2d(64,32,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv5.weight.data)
        nn.init.zeros_(self.conv5.bias.data)

        self.conv6 = torch.nn.ConvTranspose2d(32,16,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv6.weight.data)
        nn.init.zeros_(self.conv6.bias.data)

        self.conv7 = torch.nn.ConvTranspose2d(16,8,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv7.weight.data)
        nn.init.zeros_(self.conv7.bias.data)
 
        self.conv8 = torch.nn.ConvTranspose2d(8,4,kernel_size=5,stride=2,padding=2,output_padding=1)
        nn.init.xavier_uniform_(self.conv8.weight.data)
        nn.init.zeros_(self.conv8.bias.data)


    def encode(self, x, pos_grid, pos_cloud, ai_enc, N_grid_x, batchSize):
        x[:,:,0] = (x[:,:,0] - .5) / .5
        x[:,:,1] = (x[:,:,1] + 2.) / 1.5
        x[:,:,2] = (x[:,:,2] + 1.) / .75
        x[:,:,3] = (x[:,:,3] - 1.) / 3.5
        
        x = knn_interpolate(x,pos_cloud,pos_grid,assign_index=ai_enc)
        
        x = x.transpose(1,2)
        x = x.reshape((batchSize, 4, N_grid_x, N_grid_x))
        x = torch.nn.functional.elu(self.conv1(x))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        x = torch.nn.functional.elu(self.conv4(x))
        x = x.reshape(batchSize, 64*4*4)
        x = torch.nn.functional.elu(self.linear1(x))
        
        return x

    def decode(self, x, pos_grid, pos_cloud, ai_dec, N_grid_x, batchSize):
        x = torch.nn.functional.elu(self.linear2(x))
        x = x.reshape(batchSize, 64, 4, 4)
        x = torch.nn.functional.elu(self.conv5(x))
        x = torch.nn.functional.elu(self.conv6(x))
        x = torch.nn.functional.elu(self.conv7(x))
        x = self.conv8(x)
        
        x = x.reshape((batchSize,4,N_grid_x*N_grid_x))
        x = x.transpose(2,1)
        x = knn_interpolate(x,pos_grid,pos_cloud,assign_index=ai_dec)

        x[:,:,0] = (x[:,:,0]*.5 + .5)
        x[:,:,1] = (x[:,:,1]*1.5 - 2.)
        x[:,:,2] = (x[:,:,2]*.75 - 1.)
        x[:,:,3] = (x[:,:,3]*3.5 + 1.)

        return x


def genDomain(cell_centroids,gamma,u_1,v_1):
    N = np.shape(cell_centroids)[0]
    
    rho = np.zeros((N),dtype=np.float64)
    vx = np.zeros((N),dtype=np.float64)
    vy = np.zeros((N),dtype=np.float64)
    P = np.zeros((N),dtype=np.float64)
    
    
    p1 = 1.
    p2 = .4
    rho1 = 1.
    rho3 = .8
    u1 = -.1*float(u_1)
    v1 = -.1*float(v_1)
    
    p3=p2
    p4=p2
    
    u3=u1
    u4=u1
    
    v2=v1
    v3=v1
    
    # from rarefaction condition
    rho2 = rho1*((p2/p1)**(1/gamma))
    
    # from shock condition
    pi41 = (p4/p1+(gamma-1)/(gamma+1))/(1+(gamma-1)/(gamma+1)*p4/p1)
    rho4 = rho1*pi41
    
    # from conditions provided by configuration G
    phi_21 = 2*np.sqrt(gamma)/(gamma-1)*(np.sqrt(p2/rho2) - np.sqrt(p1/rho1))
    u2 = u1+phi_21
    
    psi_41 = np.sqrt((p4-p1)*(rho4-rho1)/(rho4*rho1))
    v4 = v1 + psi_41
    
    for i in range(N):
        if cell_centroids[i,0] > .5 and cell_centroids[i,1] > .5: # top right quadrant
            rho[i] = rho1
            vx[i] = u1
            vy[i] = v1
            P[i] = p1
        elif cell_centroids[i,0] <= .5 and cell_centroids[i,1] > .5: # top left quadrant
            rho[i] = rho2
            vx[i] = u2
            vy[i] = v2
            P[i] = p2
        elif cell_centroids[i,0] <= .5 and cell_centroids[i,1] <= .5: # bottom left quadrant
            rho[i] = rho3
            vx[i] = u3
            vy[i] = v3
            P[i] = p3
        elif cell_centroids[i,0] > .5 and cell_centroids[i,1] <= .5: # bottom right quadrant
            rho[i] = rho4
            vx[i] = u4
            vy[i] = v4
            P[i] = p4
            
           
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
    

@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64))
def getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt):
    edge_index = edge_data[:,:2]
    n_hat = edge_data[:,2:4]
    faceArea = edge_data[:,4]

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

        if cell1==-1 or cell1==-2 or cell1==-3 or cell1==-4:
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
        else:
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

        # # rotate velocity components to align with cell face
        vx_c[cell0] = n_hat[i,0]*vx_cart0 + n_hat[i,1]*vy_cart0
        vy_c[cell0] = - n_hat[i,1]*vx_cart0 + n_hat[i,0]*vy_cart0
        vx_c1 = n_hat[i,0]*vx_cart1 + n_hat[i,1]*vy_cart1
        vy_c1 = - n_hat[i,1]*vx_cart1 + n_hat[i,0]*vy_cart1

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

    
@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64))
def updateState(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt):
    
    Mass_n = np.zeros((N))
    Momx_n = np.zeros((N))
    Momy_n = np.zeros((N))
    Energy_n = np.zeros((N))
    
    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
    
    N = np.shape(cell_centroids)[0]
    
    vel_Mass_c, vel_Momx_c, vel_Momy_c, vel_Energy_c = getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt)
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
    modelNum = int(sys.argv[2])
    
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


    u_1                    = 19
    v_1                    = 3.5
    output_directory       = 'Outputs/lat'+str(lat_dim) + '_' + str(u_1) + '_' + str(v_1) + '/'
    filename               = 'output'
    N                      = np.shape(cell_centroids)[0]
    gamma                  = 1.4
    courant_fac            = 0.4
    t                      = 0
    dt                     = 1e-3
    tEnd                   = .301

    train_file             = "output"
    mesh_location          = "../Mesh"
    volume_file            = "cell_volumes.npy"
    edge_index_file        = "edge_index.npy"

    # reset output directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    
    os.mkdir(output_directory)

    x_curr = np.zeros((N,4))

    # initialize domain
    rho, vx, vy, P = genDomain(cell_centroids,gamma,u_1,v_1)
    Mass, Mom_x, Mom_y, Energy, E, H = getConserved(rho, vx, vy, P, gamma, cell_volumes)

    x_curr[:,0] = torch.tensor(Mass)
    x_curr[:,1] = torch.tensor(Mom_x)
    x_curr[:,2] = torch.tensor(Mom_y)
    x_curr[:,3] = torch.tensor(Energy)

    # load trained model
    model = architecture(lat_dim)
    model = torch.load('lat' + str(lat_dim) + '_model'+str(modelNum)).cpu()

    model = model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    N_torch = torch.tensor(N)

    # generate structured mesh for interpolation
    N_grid = 4096
    N_grid_x = 64
    x_grid = torch.linspace(0,1,steps=N_grid_x)
    grid_x, grid_y = torch.meshgrid(x_grid, x_grid)
    pos_grid = torch.zeros((N_grid,2))
    pos_grid[:,0] = grid_x.flatten()
    pos_grid[:,1] = grid_y.flatten()

    directory = '../Mesh/'
    pos = torch.tensor(np.load(directory + 'cell_centroids.npy'))

    ai_enc = knn(pos, pos_grid, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_dec = knn(pos_grid, pos, k=3, batch_x=None, batch_y=None, num_workers=1)

    # generate and encode initial conditions to latent representation
    x_next = np.copy(x_curr)
    inputs = torch.zeros((1,N,4))
    inputs[0,:,:] = torch.tensor(x_curr[:,:])
    inputs=inputs.to(device)
    x_hat_next = model.encode(inputs,pos_grid,pos,ai_enc,N_grid_x,1)

    x_next = model.decode(x_hat_next,pos_grid,pos,ai_dec,N_grid_x,1)

    x_next = x_next[0,:,:]

    x_hat_curr = x_hat_next.clone().to(device)
    x_curr = x_next.clone()

    x_curr = x_curr.detach().clone().type(torch.DoubleTensor).cpu().numpy()
    x_next = x_next.detach().clone().type(torch.DoubleTensor).numpy()

    x_next_state = np.zeros((4*N))

    Mass = x_curr[:,0]
    Momx = x_curr[:,1]
    Momy = x_curr[:,2]
    Energy = x_curr[:,3]

    edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    edge_index = edge_index.to(device)

    x_curr = toState(Mass, Momx, Momy, Energy, N)

    f = np.zeros((4*N))

    t_step = 0
    t_out = 0

    iter_out = 0
    while t < tEnd:
        del x_curr, x_hat_curr
        
        x_hat_curr = torch.clone(x_hat_next)
        
        # decode to high-dimensional representation
        x_curr = model.decode(x_hat_curr,pos_grid,pos,ai_dec,N_grid_x,1)
        x_curr = x_curr[0,:,:].detach().type(torch.DoubleTensor).cpu().numpy()
        
        Mass_c = x_curr[:,0]
        Momx_c = x_curr[:,1]
        Momy_c = x_curr[:,2]
        Energy_c = x_curr[:,3]

        # get primitive variables
        rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)        
        
        # save outputs
        void = saveOutput(cell_centroids, rho_c, vx_c, vy_c, P_c, output_directory, filename, N, t_out)
        t_out += 1
        
        # update state
        Mass_n, Momx_n, Momy_n, Energy_n = updateState(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt)
        
        
        k=0
        tol = 1e-12
        error_LDM = 1e5
        iter_count = 0
        
        J_decoder = torch.zeros((4*N,lat_dim))
        
        model = model.eval()
        
        while error_LDM<tol:
            t0 = time.time()

            x_next_torch = model.decode(x_hat_next,pos_grid,pos,ai_dec,N_grid_x,1).detach()
            x_next_torch = x_next_torch[0,:,:]
            
            x_next_state[:N] = x_next_torch[:,0].detach().cpu().numpy()
            x_next_state[N:2*N] = x_next_torch[:,1].detach().cpu().numpy()
            x_next_state[2*N:3*N] = x_next_torch[:,2].detach().cpu().numpy()
            x_next_state[3*N:] = x_next_torch[:,3].detach().cpu().numpy()
            
            f = getResidual(x_next_state,Mass_n,Momx_n,Momy_n,Energy_n,N)
            
            J_decoder_temp = torch.func.jacfwd(model.decode, argnums=0)(x_hat_next,pos_grid,pos,ai_dec,N_grid_x,1)
            J_decoder[:N,:] = J_decoder_temp[0,:,0,0,:]
            J_decoder[N:2*N,:] = J_decoder_temp[0,:,1,0,:]
            J_decoder[2*N:3*N,:] = J_decoder_temp[0,:,2,0,:]
            J_decoder[3*N:4*N,:] = J_decoder_temp[0,:,3,0,:]
            J_decoder = J_decoder.to(device)


            # LSPG Step
            B = torch.matmul(J_decoder.T,torch.tensor(f).type(torch.FloatTensor).to(device))
            C = torch.matmul(J_decoder.T,J_decoder)
            p = -torch.linalg.solve(C,B)
            x_hat_next_old = x_hat_next.clone()
            
            # error/convergence check
            error_LDM = torch.linalg.norm(B)
            
            if iter_count==0 and t_step==0:
                tol = 1e-3*error_LDM
            
            if error_LDM<tol:
                break
            
            if iter_count==0:
               alpha = 1.
            
            x_hat_next = x_hat_next_old + alpha*p
            if iter_count==30:
                print('DID NOT CONVERGE !!!!!!!!!!!!!!!!!!')
                sys.exit()

            iter_count += 1
            
        t += dt
        t_step += 1
