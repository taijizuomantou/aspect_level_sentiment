import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAX_LENGTH= 128
#class GCNClassifier(nn.Module):
#    def __init__(self,num_labels):
#        super().__init__()
#        in_dim = 768
#        self.gcn_model = GCNAbsaModel()
#        self.classifier = nn.Linear(in_dim, num_labels)
#    def forward(self, inputs):
#        outputs = self.gcn_model(inputs)
#        logits = self.classifier(outputs)
#        return logits, outputs

class GCNAbsaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn layer
        self.gcn = GCN()

    def forward(self, head, gcn_input):
   #     print("head"+ str(head.size()))
      #  print(head)
        def inputs_to_tree_reps(head):
            adj = np.zeros((MAX_LENGTH, MAX_LENGTH), dtype=np.float32)

            
            for i in range(0,MAX_LENGTH):
                if head[i] != -1 :
                    # adj[0][i] = 1
                     adj[i][head[i]] = 1
                adj[i][i] = 1;
                   #  adj[i][root] = 1
            #adj = adj * adj
            return adj
        
       # adj = [inputs_to_tree_reps(onehead,typee,loc).reshape(1, MAX_LENGTH, MAX_LENGTH) for onehead in head]
        adj = []  
        for i,onehead in enumerate(head):
            adj.append(inputs_to_tree_reps(onehead).reshape(1, MAX_LENGTH, MAX_LENGTH) )
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj)
        
        adj.cuda()
        h = self.gcn(adj, gcn_input)
        #outputs = h
                # avg pooling asp feature
#        print(h.size())
       # outputs = (h).sum(dim=2)  
       # print(outputs.size())
       # print("outputs"+str(outputs.size()))
        return h

class GCN(nn.Module):
    def __init__(self, num_layers = 2,gcn_dropout = 0.5):
        super(GCN, self).__init__()
        self.gcn_drop = nn.Dropout(gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            self.W.append(nn.Linear(768, 768))



    def forward(self, adj, gcn_inputs,layer =2):
        # gcn layer
        #gcn_inputs.cuda()
        denom = adj.sum(2).unsqueeze(2) + 1    # norm
       # print("adj"+ str(adj.size()))
      #  print("gcn_inputs"+ str(gcn_inputs.size()))
        for l in range(layer):
            Ax = torch.bmm(adj.cuda(),gcn_inputs.cuda())
            AxW = self.W[l](Ax)
            AxW = AxW/ denom.cuda()
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < layer - 1 else gAxW
        return gcn_inputs


