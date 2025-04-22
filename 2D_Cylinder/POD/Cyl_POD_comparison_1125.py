import numpy as np
import sys
import pandas as pd
import torch


def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = 1/(gamma-1)*P/rho + .5*(u*u + v*v)
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    Energy = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H

    
def loadDataSingle(train_file, mesh_location, directory, volume_file, ns, gamma):
    
    cell_volumes = torch.tensor(np.load(mesh_location + '/' + volume_file))
    data = torch.tensor(np.array(pd.read_csv(directory + '/' + train_file + "0000001.csv"))) ####NOTE: THIS NEEDS TO HAVE A HEADER=NONE STATEMENT!!!!! WE ARE MISSING ONE OF THE INPUTS RIGHT NOW!!!!!!!!!!!!!!!!!!!
    
    N = np.shape(data)[0]
    x_hist = torch.zeros((ns,N,4))
    
    
    for t in range(ns):
        if t<10:
            name = directory+ '/' + train_file + "000000" + str(t) + ".csv"
        elif t<100:
            name = directory + '/' + train_file + "00000" + str(t) + ".csv"
        elif t<1000:
            name = directory + '/' + train_file + "0000" + str(t) + ".csv"
        elif t<10000:
            name = directory + '/' + train_file + "000" + str(t) + ".csv"
        elif t<100000:
            name = directory + '/' + train_file + "00" + str(t) + ".csv"
        elif t<1000000:
            name = directory + '/' + train_file+ "0" + str(t) + ".csv"
        else:
            name = directory + '/' + train_file + str(t) + ".csv"
        
        data = torch.tensor(np.array(pd.read_csv(name)))
        
        rho = data[:,3]
        vx = data[:,4]
        vy = data[:,5]
        P = data[:,6]
        
        
        Mass, Momx, Momy, Energy, E, H = getConserved( rho, vx, vy, P, gamma, cell_volumes )
        
        x_hist[t,:,0] = Mass
        x_hist[t,:,1] = Momx
        x_hist[t,:,2] = Momy
        x_hist[t,:,3] = Energy
    
    return x_hist


if __name__=="__main__":
    lat_dim = sys.argv[1]

    N = 4148
    ns = 1001
    mesh_location = '../Mesh'
    GT_directory = '../fullOrderModel/Outputs/mach_1125'
    pred_directory = 'lat' + str(lat_dim) + '_mach_1125'
    filename = 'output'
    
    # load ground truth and predicted solutions
    x_hist_GT, cell_volumes, pos = loadDataSingle('output', mesh_location, GT_directory, 'cell_volumes.npy', ns, 1.4)
    x_hist_pred, cell_volumes, pos = loadDataSingle('output', mesh_location, pred_directory, 'cell_volumes.npy', ns, 1.4)

    input_hist = x_hist_GT
    pred_hist = x_hist_pred

    # get error
    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())
    print('lat: ', lat_dim, ', full_rel_error: ', full_rel_error)

