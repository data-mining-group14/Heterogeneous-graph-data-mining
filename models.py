import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv,GCNConv,ARMAConv,ChebConv
from torch.nn.parameter import Parameter
import torch

class ClassifyNet(nn.Module):
    def __init__(self,feature_dim,hid_dim,class_num,dropout,mask1,mask2,device):
        super().__init__()
        self.authorgcn1=ARMAConv(feature_dim,hid_dim,2)
        self.papergcn1=ARMAConv(hid_dim,hid_dim,2)
        self.authorgcn2=ARMAConv(hid_dim,hid_dim)
        self.papergcn2=ARMAConv(hid_dim,class_num)
        # self.authorgcn3=GCNConv(hid_dim,hid_dim)
        # self.papergcn3=GCNConv(hid_dim,hid_dim)
        # self.authorgcn4=GCNConv(hid_dim,hid_dim)
        # self.papergcn4=GCNConv(hid_dim,class_num)
        self.a2p1=author2paper(mask1,device)
        # self.a2p2=author2paper(mask1,device)
        # self.a2p3=author2paper(mask1,device)
        # self.p2a1=paper2author(mask2,device)
        # self.p2a2=paper2author(mask2,device)
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
        # x = F.dropout(x,self.dropout)
        # x = self.p2a1(x,mask2)
        # x = F.relu(self.authorgcn3(x, adj_author))
        # x = F.dropout(x, self.dropout)
        # x = F.relu(self.authorgcn4(x, adj_author))
        # x = F.dropout(x, self.dropout)
        # x = self.a2p2(x,mask1)
        # x = F.relu(self.papergcn3(x, adj_paper))
        # x = F.dropout(x,self.dropout)
        # x = self.papergcn4(x, adj_paper)
        return x


class author2paper(nn.Module):
    def __init__(self,mask,device):
        super().__init__()
        # self.weight = mask.clone()
        # self.weight.requires_grad=True
        # self.device=device
    def forward(self,feature,mask):
        # self.weight=self.weight.cpu()
        # feature=feature.cpu()
        # trans=torch.mul(mask,self.weight)
        return torch.mm(mask,feature)

class paper2author(nn.Module):
    def __init__(self,mask,device):
        super().__init__()
        # self.weight = mask.clone()
        # self.weight.requires_grad=True
        # self.device=device
    def forward(self,feature,mask):
        # self.weight=self.weight.cpu()
        # feature=feature.cpu()
        # trans=torch.mul(mask.t(),self.weight)
        return torch.mm(mask,feature)    

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
