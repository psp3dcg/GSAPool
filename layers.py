import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv

class GSAPool(torch.nn.Module):

    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.6, pooling_conv="GCNConv", fusion_conv="false",
	                    min_score=None, multiplier=1, non_linearity=torch.tanh):
        super(GSAPool,self).__init__()
        self.in_channels = in_channels
		
        self.ratio = pooling_ratio
        self.alpha = alpha
		
        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer = nn.Linear(in_channels, 1)
        self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)
		
        self.min_score = min_score
        self.multiplier = multiplier
        self.fusion_flag = 0
        if(fusion_conv!="false"):
            self.fusion_flag = 1
        self.non_linearity = non_linearity

    def conv_selection(self, conv, in_channels, conv_type=0):
        if(conv_type == 0):
            out_channels = 1
        elif(conv_type == 1):
            out_channels = in_channels
        if(conv == "GCNConv"):
            return GCNConv(in_channels,out_channels)
        elif(conv == "ChebConv"):
            return ChebConv(in_channels,out_channels,1)
        elif(conv == "SAGEConv"):
            return SAGEConv(in_channels,out_channels)
        elif(conv == "GATConv"):
            return GATConv(in_channels,out_channels, heads=1, concat=True)
        elif(conv == "GraphConv"):
            return GraphConv(in_channels,out_channels)
        else:
            raise ValueError

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        #SBTL
        score_s = self.sbtl_layer(x,edge_index).squeeze()
        #FBTL
        score_f = self.fbtl_layer(x).squeeze()
        #hyperparametr alpha
        score = score_s*self.alpha + score_f*(1-self.alpha)

        score = score.unsqueeze(-1) if score.dim()==0 else score
		
        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = softmax(score, batch)
        perm = topk(score, self.ratio, batch)
		
		#fusion
        if(self.fusion_flag == 1):
            x = self.fusion(x, edge_index)
    
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
