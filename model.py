import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from utils import sim

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, edge_weight=None):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x, edge_weight=edge_weight))
        x = self.convs[-1](graph, x)

        return x

class HomoGCL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_nodes, num_proj_hidden, tau: float=0.5):
        super(HomoGCL, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim, n_layers)
        self.tau: float = tau
        self.fc1 = th.nn.Linear(out_dim, num_proj_hidden)
        self.fc2 = th.nn.Linear(num_proj_hidden, out_dim)
        self.num_nodes = num_nodes
        self.num_proj_hidden = num_proj_hidden

        self.neighboraggr = GraphConv(num_nodes, num_nodes, norm='both', weight=False, bias=False)
    
    def posaug(self, graph, x, edge_weight):
        return self.neighboraggr(graph, x, edge_weight=edge_weight)

    def forward(self, graph1, feat1, graph2, feat2, graph, feat):
        z1 = self.encoder(graph1, feat1)
        z2 = self.encoder(graph2, feat2)
        z = self.encoder(graph, feat)
        return z1, z2, z

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def semi_loss(self, z1, adj1, z2, adj2, confmatrix, mean):
        f = lambda x: th.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        if mean:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1) / (adj1.sum(1)+0.01) 
        else:
            pos = between_sim.diag() + (refl_sim * adj1 * confmatrix).sum(1)
        neg = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (refl_sim * adj1).sum(1) - (between_sim * adj2).sum(1)
        loss = -th.log(pos / (pos + neg))

        return loss

    def loss(self, z1, graph1, z2, graph2, confmatrix, mean):
        if self.num_proj_hidden > 0:
            h1 = self.projection(z1)
            h2 = self.projection(z2)
        else:
            h1 = z1
            h2 = z2

        l1 = self.semi_loss(h1, graph1, h2, graph2, confmatrix, mean)
        l2 = self.semi_loss(h2, graph2, h1, graph1, confmatrix, mean)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def get_embedding(self, graph, feat):
        with th.no_grad():
            out = self.encoder(graph, feat)
            return out.detach()

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret
