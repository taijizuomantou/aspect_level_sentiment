import torch
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
MAX_LENGTH = 128
class Net(torch.nn.Module):
    # torch.nn.Module 是所有神经网络单元的基类
    def __init__(self):
        super(Net, self).__init__()  ###复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = GCNConv(768, 16)
        self.conv2 = GCNConv(16, 768)

    def forward(self, head,x):
        def inputs_to_tree_reps(head):
            begin =[]
            end = []
            for i in range(0,MAX_LENGTH):
                if head[i] != -1 :
                    # adj[0][i] = 1
                     begin.append(i)
                     end.append(head[i])
                   #  adj[i][root] = 1
            #adj = adj * adj
            begin = np.asarray(begin)
            end = np.asarray(end)
            ans = np.vstack((begin,end))
            return ans
        
        edge_index = []
        for i,onehead in enumerate(head):
            oneindex = inputs_to_tree_reps(onehead)

            oneindex = oneindex.reshape(1,2,-1)
            edge_index.append(oneindex)

        edge_index = np.concatenate(edge_index, axis=0)
        edge_index = torch.from_numpy(edge_index)

        

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
