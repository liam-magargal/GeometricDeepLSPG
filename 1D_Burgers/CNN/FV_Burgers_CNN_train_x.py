import numpy as np
import time
import torch
import torch.nn as nn
import sys

class architecture(nn.Module):
    def __init__(self,M):
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
        
        
        self.layer5 = torch.nn.Linear(128,M)
        nn.init.xavier_uniform_(self.layer5.weight.data)
        nn.init.zeros_(self.layer5.bias.data)
        
        self.layer6 = torch.nn.Linear(M,128)
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

    
def trainModel(model, optimizer, w_train, w_val, w_test, batchSize, maxEpoch, M):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
    N = w_train.size()[0]
    ns_train = w_train.size()[1]
    ns_val = w_val.size()[1]
    
    train_set = np.arange(ns_train)
    val_set = np.arange(ns_val)
    
    w_train = w_train.to(device)
    w_val = w_val.to(device)
    w_test = w_test.to(device)
   
    numBatch = 1804 

    for epoch in range(maxEpoch):
        np.random.shuffle(train_set)
        t1 = time.time()

        # loop over all mini batches
        for batch in range(numBatch):
            optimizer.zero_grad()
            train_set_r = train_set[batch*batchSize:(batch+1)*batchSize]
            
            # training loss
            inputs = w_train[:,train_set_r].T.reshape((batchSize,1,N))
            x_lat = model.encode(inputs,batchSize)
            x_pred = model.decode(x_lat,batchSize)
            x_target = w_train[:,train_set_r].T.reshape((batchSize,1,N))
            loss = torch.linalg.norm(x_pred - x_target)
            loss = loss*loss

            loss.backward()
            optimizer.step()
            
        ## validation loss
        np.random.shuffle(val_set)

        inputs = w_val[:,val_set[:batchSize]].T.reshape((batchSize,1,N))
        x_lat = model.encode(inputs,batchSize)
        x_pred = model.decode(x_lat,batchSize)
        x_target = w_val[:,val_set[:batchSize]].T.reshape((batchSize,1,N))
        val_loss = torch.linalg.norm(x_pred - x_target)
        val_loss = val_loss*val_loss
        
               
        torch.save(model, 'lat' + str(M))
        
        t2 = time.time()

        print('epoch: ', epoch, ', loss: ', loss.cpu().detach(), ', val_loss: ', val_loss.cpu().detach(), 'time for epoch: ', t2-t1)
    
    return model



def loadData(numTrain, numVal, nt, N):
    w_hist_train = torch.load('w_train')
    
    w_hist_train = w_hist_train[:-30,:]
    
    w_train = w_hist_train
    
    train_val_set = np.arange(nt*numTrain)
    np.random.shuffle(train_val_set)
    val_set = train_val_set[:numVal]
    train_set = train_val_set[numVal:]
    
    w_val = w_train[:,val_set]
    w_train = w_train[:,train_set]
    
    return w_train, w_val


def defineModel(M):
    model = architecture(M)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    
    return model, optimizer

if __name__=="__main__":
    M = int(sys.argv[1])
    N = 256
    nt = 501
    numTrain = 80 # number of parameter sets in training data
    numVal = 4000 # number of snapshots from training stored as validation
    batchSize = 20
    maxEpoch = 1000
    
    # load training data and perform train/test split
    w_train, w_val = loadData(numTrain, numVal, nt, N)
    
    # construct CNN-based autoencoder and optimizer
    model, optimizer = defineModel(M)
    
    # train CNN-based autoencoder
    model = trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, M)

