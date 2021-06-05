import torch
import torch.nn as nn
import torch.nn.functional as F



MAX_LENGTH = 128
import numpy as np
def normalize(A , symmetric=True):
	# A = A+I
	# 所有节点的度
	d = A.sum(1)
	D =np.diag(np.power(d,-1))
	return np.dot(D,A)


class GCN(nn.Module):

    def __init__(self ):
        super(GCN,self).__init__()
        dim_in = 768
        dim_out = 768
        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)
    def forward(self,head,X):
          def inputs_to_tree_reps(head):
            adj = np.zeros((MAX_LENGTH, MAX_LENGTH), dtype=np.float32)

            
            for i in range(0,MAX_LENGTH):
                if head[i] != -1 :
                    # adj[0][i] = 1
                     adj[head[i]][i] = 1
                adj[i][i] = 1;
                   #  adj[i][root] = 1
            #adj = adj * adj
            adj = normalize(adj)
            return adj
          adj = []  
          for i,onehead in enumerate(head):
              adj.append(inputs_to_tree_reps(onehead).reshape(1, MAX_LENGTH, MAX_LENGTH) )
          adj = np.concatenate(adj, axis=0)
          adj = torch.from_numpy(adj) 
          adj.cuda()
          A = adj.cuda()
          X = F.relu(self.fc1(torch.matmul(A,X)))
          X = F.relu(self.fc2(torch.matmul(A,X)))
          return self.fc3(torch.matmul(A,X))
