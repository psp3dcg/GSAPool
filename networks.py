import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from layers import GSAPool

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
		
        self.args = args
        self.nhid = args.nhid

        self.num_features = args.num_features
        self.num_classes = args.num_classes
        
        self.alpha = args.alpha
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
		
        self.pooling_layer_type = args.pooling_layer_type
        self.feature_fusion_type = args.feature_fusion_type
		
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = GSAPool(self.nhid, pooling_ratio=self.pooling_ratio, alpha = self.alpha, 
		                     pooling_conv=self.pooling_layer_type, fusion_conv=self.feature_fusion_type)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = GSAPool(self.nhid, pooling_ratio=self.pooling_ratio, alpha = self.alpha, 
		                     pooling_conv=self.pooling_layer_type, fusion_conv=self.feature_fusion_type)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = GSAPool(self.nhid, pooling_ratio=self.pooling_ratio, alpha = self.alpha, 
		                     pooling_conv=self.pooling_layer_type, fusion_conv=self.feature_fusion_type)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

		
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
		
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    
