import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SplineConv, GATConv, ARMAConv, SuperGATConv, TransformerConv
import numpy as np
import torch
from torch.nn.parameter import Parameter

class Net(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached=True) 
        self.conv2 = GCNConv(nhid, nclass, cached=True)
        self.p = dropout_rate

    def forward(self, x, adj):
        x = F.relu(self.conv1(x,adj))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj)
        x = F.log_softmax(x, dim=1)
        return x

class NetGAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, heads = 8) 
        self.conv2 = GATConv(nhid * 8, nclass)
        self.p = dropout_rate

    def forward(self, x, adj):
        x = F.relu(self.conv1(x,adj))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj)
        x = F.log_softmax(x, dim=1)
        return x
class NetARMA(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(nfeat, nhid, num_stacks = 2, num_layers =1)
        self.conv2 = ARMAConv(nhid, nclass, num_stacks = 2, num_layers = 1)
        self.p = dropout_rate

    def forward(self, x, adj):
        x = self.conv1(x,adj)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj)
        x = F.log_softmax(x, dim=1)
        return x

class NetTransformer(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = TransformerConv(nfeat, nhid, heads = 4)
        self.conv2 = TransformerConv(nhid * 4, nclass)
        self.p = dropout_rate

    def forward(self, x, adj):
        x = self.conv1(x,adj)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj)
        x = F.log_softmax(x, dim=1)
        return x

class NetSuperGAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = SuperGATConv(nfeat, nhid, heads = 8, neg_sample_ratio = 0.5, edge_sample_ratio = 0.8)
        self.conv2 = SuperGATConv(nhid * 8, nclass)
        self.p = dropout_rate

    def forward(self, x, adj):
        x = F.elu(self.conv1(x,adj))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj)
        x = F.log_softmax(x, dim=1)
        return x

class NetSpline(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout_rate):
        super(Net, self).__init__()
        self.conv1 = SplineConv(nfeat, nhid, dim = 1, kernel_size = 2)
        self.conv2 = SplineConv(nhid, nclass, dim = 1, kernel_size = 2)
        self.p = dropout_rate

    def forward(self, x, adj, pseudoOne):
        x = F.elu(self.conv1(x,adj,pseudoOne))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x,adj,pseudoOne)
        x = F.log_softmax(x, dim=1)
        return x