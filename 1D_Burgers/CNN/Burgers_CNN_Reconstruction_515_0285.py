import sys
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
    
    # load ground truth solution
    w_truth = torch.load('../fullOrderModel/w_test_515_0285').to(device)
    w_truth = w_truth[:-30,:]
    N = w_truth.size()[0]
    nt = w_truth.size()[1]
    batchSize = 1

    w_pred = torch.zeros((N,nt))

    # load trained model
    model = architecture()
    model = (torch.load('lat' + str(N_lat)))


    w_lat_out = torch.zeros((N_lat,nt))
    rel_error = torch.zeros((nt))

    input_hist = torch.zeros(N,nt)
    pred_hist = torch.zeros(N,nt)

    # get error
    for i in range(nt):
        inputs = w_truth[:,i].reshape((1,1,N))
        input_hist[:,i] = inputs[0,0,:].clone()
        x_lat = model.encode(inputs,batchSize)
        x_pred = model.decode(x_lat,batchSize)

        w_lat_out[:,i] = x_lat.flatten()
        pred_hist[:,i] = x_pred[0,0,:]
        w_pred[:,i] = x_pred.flatten().clone()

        rel_error[i] = (torch.linalg.vector_norm(x_pred.reshape((1,1,N)).flatten()-w_truth[:,i].reshape((1,1,N)).flatten())) / (torch.linalg.vector_norm(w_truth[:,i].reshape((1,1,N)).flatten()))
    
    full_rel_error = torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist-pred_hist, dim=0).flatten()) / torch.linalg.vector_norm(torch.linalg.vector_norm(input_hist, dim=0).flatten())

    print(full_rel_error)
    
    