import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GCNConv,ARMAConv,ChebConv
import torch
class ClassifyNet(nn.Module):
    def __init__(self,feature_dim,hid_dim,class_num,dropout,mask1,mask2,device):
        super().__init__()
        self.authorgcn1=ARMAConv(feature_dim,hid_dim,2)
        self.papergcn1=ARMAConv(hid_dim,hid_dim,2)
        self.authorgcn2=ARMAConv(hid_dim,hid_dim)
        self.papergcn2=ARMAConv(hid_dim,class_num)
        self.a2p1=author2paper()
        self.dropout=dropout
    def forward(self,x,adj_author,adj_paper,mask1,mask2):
        x = F.relu(self.authorgcn1(x, adj_author))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.authorgcn2(x, adj_author))
        x = F.dropout(x, self.dropout)
        x = self.a2p1(x,mask1)
        x = F.relu(self.papergcn1(x, adj_paper))
        x = F.dropout(x,self.dropout)
        x = self.papergcn2(x, adj_paper)
        return x
class author2paper(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,feature,mask):
        return torch.mm(mask,feature)   


