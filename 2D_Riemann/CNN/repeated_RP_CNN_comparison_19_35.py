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

    
def loadData(train_file, mesh_location, directory, volume_file, ns, gamma):
    
    cell_volumes = torch.tensor(np.load(mesh_location + '/' + volume_file))
    data = torch.tensor(np.array(pd.read_csv(directory + '/' + train_file + "0000001.csv")))
    
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
	modelNum = sys.argv[2]

	batchSize = 1

	N = 4328
	ns = 301
	mesh_location = '../Meshes/2D_Mesh_4k'
	GT_directory = '../RP_Jan14_unstructured/19_3.5'
	pred_directory = 'Outputs/lat' + str(lat_dim) + '_19_3.5'
	filename = 'output'

    # load ground truth and predicted solution series
	x_hist_GT, cell_volumes, pos = loadData('output', mesh_location, GT_directory, 'cell_volumes.npy', ns, 1.4)
	x_hist_pred, cell_volumes, pos = loadData('output', mesh_location, pred_directory, 'cell_volumes.npy', ns, 1.4)

	input_hist = x_hist_GT
	pred_hist = x_hist_pred

    # get state prediction error
	full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

	print(lat_dim, modelNum, full_rel_error)
