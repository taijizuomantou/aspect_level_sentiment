import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
MAX_LENGTH= 128
class GCNConv(nn.Module):
    def __init__(self, head, in_channels, out_channels):
        super(GCNConv, self).__init__()
        adj = [self.inputs_to_tree_reps(onehead).reshape(1, MAX_LENGTH, MAX_LENGTH) for onehead in head]
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj)
        adj.cuda()
        A = adj
        print(A.size())
        self.A_hat = A
        self.D     = torch.diag(torch.sum(A,1))
        self.D     = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W     = nn.Parameter(torch.rand(in_channels,out_channels))
    def inputs_to_tree_reps(self,head):
            adj = np.zeros((MAX_LENGTH, MAX_LENGTH), dtype=np.float32)
            for i in range(0,MAX_LENGTH):
                if head[i] == -1:break
                adj[i][i] = 1
                adj[head[i]][i] = 1
            return adj
    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out

class Net(torch.nn.Module):
    def __init__(self,A, nfeat, nhid, nout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(A,nfeat, nhid)
        self.conv2 = GCNConv(A,nhid, nout)
        
    def forward(self,X):
        H  = self.conv1(X)
        H2 = self.conv2(H)
        return H2
