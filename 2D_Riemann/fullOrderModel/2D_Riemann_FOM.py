import numpy as np
import sys
import os
import pandas as pd
import shutil
from numba import njit, float64, int64
import numba as nb


@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:],int64))
def fromState(x_curr, N):
    Mass = x_curr[:N]
    Momx = x_curr[N:2*N]
    Momy = x_curr[2*N:3*N]
    Energy = x_curr[3*N:]
    
    return Mass, Momx, Momy, Energy

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],int64))
def toState(Mass, Momx, Momy, Energy, N):
    x = np.zeros((4*N))
    x[:N] = Mass
    x[N:2*N] = Momx
    x[2*N:3*N] = Momy
    x[3*N:] = Energy
    
    return x

@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]))
def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = 1/(gamma-1)*P/rho + .5*(u*u + v*v)
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    Energy = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H


@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]))
def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    
    return rho, u, v, P, E, H


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
        
        # if boundary cell, apply outflow conditions
        if cell1==-1:
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
        
        # otherwise, not boundary cell
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
        
        epsilon = 1e-12*1.0 #1.0 is the reference velocity in this case (its a rough estimation of the maximum initial velocity)
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



@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64))
def updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt):
    
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
    u_1 = sys.argv[1]
    v_1 = sys.argv[2]
    
    print('u_1 : ', u_1, ', v_1: ', v_1)
    
    ## Import mesh data
    mesh_file = '../Mesh/'
    edge_index = np.load(mesh_file + 'edge_index.npy') 
    cell_centroids = np.load(mesh_file + 'cell_centroids.npy')
    n_hat = np.load(mesh_file + 'n_hat.npy')
    faceArea = np.load(mesh_file + 'faceArea.npy')
    cell_volumes = np.load(mesh_file + 'cell_volumes.npy')
    
    edge_data = np.concatenate((edge_index, n_hat), axis=1)
    edge_data = np.concatenate((edge_data, np.reshape(faceArea, (np.size(faceArea),1))), axis=1)
    edge_data = edge_data[edge_data[:, 0].argsort()]
    

    output_directory       = 'Outputs/'+str(u_1)+'_'+str(v_1)+'/'
    filename               = 'output'
    N                      = np.shape(cell_centroids)[0]
    gamma                  = 1.4 # ideal gas gamma
    t                      = 0
    dt                     = 1e-3
    tEnd                   = .301

    
    # reset output directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    
    os.mkdir(output_directory)
    
    # initialize state values
    rho_c, vx_c, vy_c, P_c = genDomain(cell_centroids,gamma,u_1,v_1)
    Mass_c, Momx_c, Momy_c, Energy_c, E_c, H_c = getConserved( rho_c, vx_c, vy_c, P_c, gamma, cell_volumes )
    rho_n, vx_n, vy_n, P_n = genDomain(cell_centroids,gamma,u_1,v_1)
    Mass_n, Momx_n, Momy_n, Energy_n, E_n, H_n = getConserved( rho_n, vx_n, vy_n, P_n, gamma, cell_volumes )

    t_step = 0
    t_out = 0
    while t < tEnd:
        rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
        void = saveOutput(cell_centroids, rho_c, vx_c, vy_c, P_c, output_directory, filename, N, t_out)
        t_out += 1
        
        Mass_n, Momx_n, Momy_n, Energy_n = updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt)
        
        Mass_c = np.copy(Mass_n)
        Momx_c = np.copy(Momx_n)
        Momy_c = np.copy(Momy_n)
        Energy_c = np.copy(Energy_n)
        
        t += dt
        t_step += 1
