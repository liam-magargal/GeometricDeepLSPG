import numpy as np
import sys
import os
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


def loadDataTotal(numTrain, numTest, numVal, ns, N, directory):
    gamma = 1.4
    
    # load test result
    test_file = 'output'
    mesh_location = '../Mesh'
    volume_file = 'cell_volumes.npy'
    x_test = loadDataSingle(test_file, mesh_location, directory, volume_file, ns, gamma)
    
        
    return x_test


def saveOutput(cell_centroids, x_out, output_directory, filename, N, t):
    
    data_out = torch.cat((cell_centroids, x_out), dim=1)
    
    df = pd.DataFrame(data_out.detach().cpu().numpy())
    
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

	N = 4328
	ns = 301
	directory = '../fullOrderModel/Outputs/13_6.5'
	filename = 'output'
	x_test = loadDataTotal(ns, N, directory)

	U = torch.tensor((pd.read_csv('U.csv')).to_numpy(),dtype=torch.float32)
	phi = U[:,1:(lat_dim+1)]

	x_out = torch.zeros((N,4))
	inputs = torch.zeros((4*N))

	input_hist = torch.zeros(4*N,ns)
	pred_hist = torch.zeros(4*N,ns)

	for i in range(ns):
		inputs[:N] = x_test[i,:,0]
		inputs[N:2*N] = x_test[i,:,1]
		inputs[2*N:3*N] = x_test[i,:,2]
		inputs[3*N:4*N] = x_test[i,:,3]
    
		input_hist[:,i] = inputs[:].clone()
        
		x_lat = torch.matmul(phi.T,inputs)
		x_pred = torch.matmul(phi,x_lat)
    
		x_out[:,0] = x_pred[:N,0]
		x_out[:,1] = x_pred[N:2*N,0]
		x_out[:,2] = x_pred[2*N:3*N,0]
		x_out[:,3] = x_pred[3*N:4*N,0]
    
		pred_hist[:,i] = x_pred[:,0]
        

	full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

	print(full_rel_error)
