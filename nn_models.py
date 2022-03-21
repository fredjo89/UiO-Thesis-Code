import torch, torch_geometric

from torch_geometric.nn import HeteroConv, NNConv, SAGEConv

################################################################################################
# One Layer NNConv
class NNConv1Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(NNConv1Layer, self).__init__()
        
        self.conv = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1, torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(1,num_features_ind), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(1,num_features_org), aggr = "add"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################

################################################################################################
# Two Layer NNConv
class NNConv2Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(NNConv2Layer, self).__init__()
        
        self.conv1 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
  
        self.conv2 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1, torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(1,num_features_ind), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(1,num_features_org), aggr = "add"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################

################################################################################################
# Three Layer NNConv
class NNConv3Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(NNConv3Layer, self).__init__()
        
        kwargs1 = {'aggr': 'add', 'flow': 'source_to_target'}

        
        self.conv1 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), **kwargs1),  
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), **kwargs1), 
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), **kwargs1), 
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), **kwargs1), 
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), **kwargs1), 
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), **kwargs1), 
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), **kwargs1), 
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), **kwargs1), 
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), **kwargs1), 
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), **kwargs1),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), **kwargs1),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), **kwargs1),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), **kwargs1),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), **kwargs1),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), **kwargs1),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), **kwargs1),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), **kwargs1),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
        
        self.conv2 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), **kwargs1), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), **kwargs1),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), **kwargs1),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), **kwargs1),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), **kwargs1),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), **kwargs1),
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), **kwargs1),
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), **kwargs1),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), **kwargs1),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), **kwargs1),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), **kwargs1),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), **kwargs1),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), **kwargs1),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), **kwargs1),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), **kwargs1),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), **kwargs1),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
  
        self.conv3 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1, torch.nn.Linear(num_features_txn_edge,num_features_ind), **kwargs1),
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), **kwargs1),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), **kwargs1),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), **kwargs1),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), **kwargs1),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), **kwargs1),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(1,num_features_ind), **kwargs1),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), **kwargs1),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), **kwargs1),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), **kwargs1),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), **kwargs1),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), **kwargs1),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), **kwargs1),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(1,num_features_org), **kwargs1),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        
        #del x_dict
        #del edge_index_dict
        #del edge_attr_dict
        
        return pred_ind, pred_org
################################################################################################


################################################################################################
# Four Layer NNConv
class NNConv4Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(NNConv4Layer, self).__init__()
        
        self.conv1 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
        
        self.conv2 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
        
        self.conv3 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind, torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(1,num_features_ind*num_features_org), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ind), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), num_features_ind ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_ind), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_org), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), num_features_org ,torch.nn.Linear(num_features_txn_edge,num_features_ext*num_features_org), aggr = "add"),
            ('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_ind*num_features_ext), aggr = "add"),
            ('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), num_features_ext ,torch.nn.Linear(num_features_txn_edge,num_features_org*num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), num_features_ind ,torch.nn.Linear(1,num_features_org*num_features_ind), aggr = "add"),
        }, aggr = 'sum')
  
        self.conv4 = HeteroConv({
            ('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1, torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"), 
            ('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(1,num_features_ind), aggr = "add"),
        
            ('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ind), aggr = "add"),
            ('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_org), aggr = "add"),
            ('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), 1 ,torch.nn.Linear(num_features_txn_edge,num_features_ext), aggr = "add"),
            ('org', 'rev_role', 'ind'): NNConv((num_features_org, num_features_ind), 1 ,torch.nn.Linear(1,num_features_org), aggr = "add"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv3(x_dict, edge_index_dict, edge_attr_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv4(x_dict, edge_index_dict, edge_attr_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################

################################################################################################
class ind_receiver(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, each_dim_out = 1):
        super().__init__()
        
        kwargs1 = {'aggr': 'add', 'flow': 'source_to_target'}        
        self.conv_1 = HeteroConv({('ind', 'txn', 'ind'): NNConv((num_features_ind, num_features_ind), each_dim_out, torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum') 
        self.conv_2 = HeteroConv({('org', 'txn', 'ind'): NNConv((num_features_org, num_features_ind), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
        self.conv_3 = HeteroConv({('ext', 'txn', 'ind'): NNConv((num_features_ext, num_features_ind), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ext), **kwargs1)}, aggr = 'sum')
        self.conv_4 = HeteroConv({('ind', 'rev_txn', 'ind'): NNConv((num_features_ind, num_features_ind), each_dim_out, torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum') 
        self.conv_5 = HeteroConv({('org', 'rev_txn', 'ind'): NNConv((num_features_org, num_features_ind), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
        self.conv_6 = HeteroConv({('ext', 'rev_txn', 'ind'): NNConv((num_features_ext, num_features_ind), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ext), **kwargs1)}, aggr = 'sum')
        self.conv_7 = HeteroConv({('org', 'rev_role', 'ind'):NNConv((num_features_org, num_features_ind), each_dim_out ,torch.nn.Linear(1,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')     
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        z_1 = self.conv_1(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_2 = self.conv_2(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_3 = self.conv_3(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_4 = self.conv_4(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_5 = self.conv_5(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_6 = self.conv_6(x_dict, edge_index_dict, edge_attr_dict)['ind']
        z_7 = self.conv_7(x_dict, edge_index_dict, edge_attr_dict)['ind']
        
        z_cat = torch.cat((z_1, z_2, z_3, z_4, z_5, z_6, z_7),1)
        return z_cat
################################################################################################


################################################################################################
class org_receiver(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, each_dim_out = 1):
        super().__init__()
        
        kwargs1 = {'aggr': 'add', 'flow': 'source_to_target'}
        self.conv_1 = HeteroConv({('ind', 'txn', 'org'): NNConv((num_features_ind, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum')
        self.conv_2 = HeteroConv({('org', 'txn', 'org'): NNConv((num_features_org, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
        self.conv_3 = HeteroConv({('ext', 'txn', 'org'): NNConv((num_features_ext, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ext), **kwargs1)}, aggr = 'sum')
        self.conv_4 = HeteroConv({('ind', 'role', 'org'): NNConv((num_features_ind, num_features_org), each_dim_out ,torch.nn.Linear(1,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum')     
        self.conv_5 = HeteroConv({('ind', 'rev_txn', 'org'): NNConv((num_features_ind, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum')
        self.conv_6 = HeteroConv({('org', 'rev_txn', 'org'): NNConv((num_features_org, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
        self.conv_7 = HeteroConv({('ext', 'rev_txn', 'org'): NNConv((num_features_ext, num_features_org), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ext), **kwargs1)}, aggr = 'sum')
           
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        z_1 = self.conv_1(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_2 = self.conv_2(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_3 = self.conv_3(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_4 = self.conv_4(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_5 = self.conv_5(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_6 = self.conv_6(x_dict, edge_index_dict, edge_attr_dict)['org']
        z_7 = self.conv_7(x_dict, edge_index_dict, edge_attr_dict)['org']
        
        z_cat = torch.cat((z_1, z_2, z_3, z_4, z_5, z_6, z_7),1)
        return z_cat
################################################################################################

################################################################################################
class ext_receiver(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, each_dim_out = 1):
        super().__init__()
        
        kwargs1 = {'aggr': 'add', 'flow': 'source_to_target'}        
        self.conv_1 = HeteroConv({('ind', 'txn', 'ext'): NNConv((num_features_ind, num_features_ext), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum')
        self.conv_2 = HeteroConv({('org', 'txn', 'ext'): NNConv((num_features_org, num_features_ext), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
        self.conv_3 = HeteroConv({('ind', 'rev_txn', 'ext'): NNConv((num_features_ind, num_features_ext), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_ind), **kwargs1)}, aggr = 'sum')
        self.conv_4 = HeteroConv({('org', 'rev_txn', 'ext'): NNConv((num_features_org, num_features_ext), each_dim_out ,torch.nn.Linear(num_features_txn_edge,each_dim_out*num_features_org), **kwargs1)}, aggr = 'sum')
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        z_1 = self.conv_1(x_dict, edge_index_dict, edge_attr_dict)['ext']
        z_2 = self.conv_2(x_dict, edge_index_dict, edge_attr_dict)['ext']
        z_3 = self.conv_3(x_dict, edge_index_dict, edge_attr_dict)['ext']
        z_4 = self.conv_4(x_dict, edge_index_dict, edge_attr_dict)['ext']
        
        z_cat = torch.cat((z_1, z_2, z_3, z_4),1)
        return z_cat
################################################################################################
    
################################################################################################
class NNConv1Layer_concatmsg(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        kwargs1 = {'aggr': 'add', 'flow': 'source_to_target'}
        each_dim_out = 5
        
        self.conv_ind_1 = ind_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, each_dim_out)
        self.linear_ind_1 = torch.nn.Linear(7*each_dim_out,1)
        
        self.conv_org_1 = org_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, each_dim_out)
        self.linear_org_1 = torch.nn.Linear(7*each_dim_out,1)
        
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z_1_dict = {"ind": None, "org": None, "ext": None}
        
        z_1_dict["ind"] = self.conv_ind_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        z_1_dict["ind"] = self.linear_ind_1(z_1_dict["ind"])
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        
        z_1_dict["org"] = self.conv_org_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        z_1_dict["org"] = self.linear_org_1(z_1_dict["org"])
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        
        
        return z_1_dict["ind"][:,0], z_1_dict["org"][:,0]
################################################################################################

################################################################################################
class NNConv2Layer_concatmsg(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        message_dim_1 = 2
        emb_dim_1 = 5
        
        message_dim_2 = 10
        emb_dim_2 = 1
        # Layer 1
        self.conv_ind_1 = ind_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_org_1 = org_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_ext_1 = ext_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        
        self.linear_ind_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_org_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_ext_1 = torch.nn.Linear(4*message_dim_1, emb_dim_1)
        
        # Layer 2
        self.conv_ind_2 = ind_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_org_2 = org_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        
        self.linear_ind_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_org_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z_1_dict = {"ind": None, "org": None, "ext": None}
        
        z_1_dict["ind"] = self.conv_ind_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        z_1_dict["ind"] = self.linear_ind_1(z_1_dict["ind"])
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        
        z_1_dict["org"] = self.conv_org_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        z_1_dict["org"] = self.linear_org_1(z_1_dict["org"])
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        
        z_1_dict["ext"] = self.conv_ext_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        z_1_dict["ext"] = self.linear_ext_1(z_1_dict["ext"])
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        
        # Layer 2
        z_2_dict = {"ind": None, "org": None, "ext": None}
        
        z_2_dict["ind"] = self.conv_ind_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        z_2_dict["ind"] = self.linear_ind_2(z_2_dict["ind"])
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        
        z_2_dict["org"] = self.conv_org_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        z_2_dict["org"] = self.linear_org_2(z_2_dict["org"])
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        
        return z_2_dict["ind"][:,0], z_2_dict["org"][:,0]
################################################################################################

################################################################################################
class NNConv3Layer_concatmsg(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        message_dim_1 = 2
        emb_dim_1 = 5
        
        message_dim_2 =  10  
        emb_dim_2 = 5
        
        message_dim_3 = 10
        emb_dim_3 = 1
        
        # Layer 1
        self.conv_ind_1 = ind_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_org_1 = org_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_ext_1 = ext_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        
        self.linear_ind_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_org_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_ext_1 = torch.nn.Linear(4*message_dim_1, emb_dim_1)
        
        # Layer 2
        self.conv_ind_2 = ind_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_org_2 = org_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_ext_2 = ext_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        
        self.linear_ind_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_org_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_ext_2 = torch.nn.Linear(4*message_dim_2, emb_dim_2)
        
        # Layer 3
        self.conv_ind_3 = ind_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        self.conv_org_3 = org_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        
        self.linear_ind_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)
        self.linear_org_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z_1_dict = {"ind": None, "org": None, "ext": None}
        
        z_1_dict["ind"] = self.conv_ind_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        z_1_dict["ind"] = self.linear_ind_1(z_1_dict["ind"])
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        
        z_1_dict["org"] = self.conv_org_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        z_1_dict["org"] = self.linear_org_1(z_1_dict["org"])
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        
        z_1_dict["ext"] = self.conv_ext_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        z_1_dict["ext"] = self.linear_ext_1(z_1_dict["ext"])
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        
        # Layer 2
        z_2_dict = {"ind": None, "org": None, "ext": None}
        
        z_2_dict["ind"] = self.conv_ind_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        z_2_dict["ind"] = self.linear_ind_2(z_2_dict["ind"])
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        
        z_2_dict["org"] = self.conv_org_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        z_2_dict["org"] = self.linear_org_2(z_2_dict["org"])
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        
        z_2_dict["ext"] = self.conv_ext_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        z_2_dict["ext"] = self.linear_ext_2(z_2_dict["ext"])
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        
        # Layer 3
        z_3_dict = {"ind": None, "org": None, "ext": None}
        
        z_3_dict["ind"] = self.conv_ind_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        z_3_dict["ind"] = self.linear_ind_3(z_3_dict["ind"])
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        
        z_3_dict["org"] = self.conv_org_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        z_3_dict["org"] = self.linear_org_3(z_3_dict["org"])
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        
        
        return z_3_dict["ind"][:,0], z_3_dict["org"][:,0]
################################################################################################

################################################################################################
class NNConv3Layer_concatmsg2(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        message_dim_1 = 2
        emb_dim_1 = 5
        
        message_dim_2 = 2
        emb_dim_2 = 5
        
        message_dim_3 = 10
        emb_dim_3 = 1
        
        # Layer 1
        self.conv_ind_1 = ind_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_org_1 = org_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_ext_1 = ext_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        
        self.linear_ind_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_org_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_ext_1 = torch.nn.Linear(4*message_dim_1, emb_dim_1)
        
        # Layer 2
        self.conv_ind_2 = ind_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_org_2 = org_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_ext_2 = ext_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        
        self.linear_ind_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_org_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_ext_2 = torch.nn.Linear(4*message_dim_2, emb_dim_2)
        
        # Layer 3
        self.conv_ind_3 = ind_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        self.conv_org_3 = org_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        
        self.linear_ind_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)
        self.linear_org_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z_1_dict = {"ind": None, "org": None, "ext": None}
        
        z_1_dict["ind"] = self.conv_ind_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        z_1_dict["ind"] = self.linear_ind_1(z_1_dict["ind"])
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        
        z_1_dict["org"] = self.conv_org_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        z_1_dict["org"] = self.linear_org_1(z_1_dict["org"])
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        
        z_1_dict["ext"] = self.conv_ext_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        z_1_dict["ext"] = self.linear_ext_1(z_1_dict["ext"])
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        
        # Layer 2
        z_2_dict = {"ind": None, "org": None, "ext": None}
        
        z_2_dict["ind"] = self.conv_ind_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        z_2_dict["ind"] = self.linear_ind_2(z_2_dict["ind"])
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        
        z_2_dict["org"] = self.conv_org_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        z_2_dict["org"] = self.linear_org_2(z_2_dict["org"])
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        
        z_2_dict["ext"] = self.conv_ext_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        z_2_dict["ext"] = self.linear_ext_2(z_2_dict["ext"])
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        
        # Layer 3
        z_3_dict = {"ind": None, "org": None, "ext": None}
        
        z_3_dict["ind"] = self.conv_ind_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        z_3_dict["ind"] = self.linear_ind_3(z_3_dict["ind"])
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        
        z_3_dict["org"] = self.conv_org_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        z_3_dict["org"] = self.linear_org_3(z_3_dict["org"])
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        
        
        return z_3_dict["ind"][:,0], z_3_dict["org"][:,0]
################################################################################################

################################################################################################
class NNConv4Layer_concatmsg(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        message_dim_1 = 2
        emb_dim_1 = 5
        
        message_dim_2 = 2
        emb_dim_2 = 5
        
        message_dim_3 = 2
        emb_dim_3 = 5
        
        message_dim_4 = 10
        emb_dim_4 = 1
        
        # Layer 1
        self.conv_ind_1 = ind_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_org_1 = org_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        self.conv_ext_1 = ext_receiver(num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, message_dim_1)
        
        self.linear_ind_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_org_1 = torch.nn.Linear(7*message_dim_1, emb_dim_1)
        self.linear_ext_1 = torch.nn.Linear(4*message_dim_1, emb_dim_1)
        
        # Layer 2
        self.conv_ind_2 = ind_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_org_2 = org_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        self.conv_ext_2 = ext_receiver(emb_dim_1,  emb_dim_1, emb_dim_1, num_features_txn_edge, message_dim_2)
        
        self.linear_ind_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_org_2 = torch.nn.Linear(7*message_dim_2, emb_dim_2)
        self.linear_ext_2 = torch.nn.Linear(4*message_dim_2, emb_dim_2)
        
        # Layer 3
        self.conv_ind_3 = ind_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        self.conv_org_3 = org_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        self.conv_ext_3 = ext_receiver(emb_dim_2,  emb_dim_2, emb_dim_2, num_features_txn_edge, message_dim_3)
        
        self.linear_ind_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)
        self.linear_org_3 = torch.nn.Linear(7*message_dim_3, emb_dim_3)
        self.linear_ext_3 = torch.nn.Linear(4*message_dim_3, emb_dim_3)
        
        
        # Layer 4
        self.conv_ind_4 = ind_receiver(emb_dim_3,  emb_dim_3, emb_dim_3, num_features_txn_edge, message_dim_4)
        self.conv_org_4 = org_receiver(emb_dim_3,  emb_dim_3, emb_dim_3, num_features_txn_edge, message_dim_4)
        
        self.linear_ind_4 = torch.nn.Linear(7*message_dim_4, emb_dim_4)
        self.linear_org_4 = torch.nn.Linear(7*message_dim_4, emb_dim_4)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z_1_dict = {"ind": None, "org": None, "ext": None}
        
        z_1_dict["ind"] = self.conv_ind_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        z_1_dict["ind"] = self.linear_ind_1(z_1_dict["ind"])
        z_1_dict["ind"] = torch.sigmoid(z_1_dict["ind"])
        
        z_1_dict["org"] = self.conv_org_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        z_1_dict["org"] = self.linear_org_1(z_1_dict["org"])
        z_1_dict["org"] = torch.sigmoid(z_1_dict["org"])
        
        z_1_dict["ext"] = self.conv_ext_1(x_dict, edge_index_dict, edge_attr_dict)
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        z_1_dict["ext"] = self.linear_ext_1(z_1_dict["ext"])
        z_1_dict["ext"] = torch.sigmoid(z_1_dict["ext"])
        
        # Layer 2
        z_2_dict = {"ind": None, "org": None, "ext": None}
        
        z_2_dict["ind"] = self.conv_ind_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        z_2_dict["ind"] = self.linear_ind_2(z_2_dict["ind"])
        z_2_dict["ind"] = torch.sigmoid(z_2_dict["ind"])
        
        z_2_dict["org"] = self.conv_org_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        z_2_dict["org"] = self.linear_org_2(z_2_dict["org"])
        z_2_dict["org"] = torch.sigmoid(z_2_dict["org"])
        
        z_2_dict["ext"] = self.conv_ext_2(z_1_dict, edge_index_dict, edge_attr_dict)
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        z_2_dict["ext"] = self.linear_ext_2(z_2_dict["ext"])
        z_2_dict["ext"] = torch.sigmoid(z_2_dict["ext"])
        
        # Layer 3
        z_3_dict = {"ind": None, "org": None, "ext": None}
        
        z_3_dict["ind"] = self.conv_ind_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        z_3_dict["ind"] = self.linear_ind_3(z_3_dict["ind"])
        z_3_dict["ind"] = torch.sigmoid(z_3_dict["ind"])
        
        z_3_dict["org"] = self.conv_org_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        z_3_dict["org"] = self.linear_org_3(z_3_dict["org"])
        z_3_dict["org"] = torch.sigmoid(z_3_dict["org"])
        
        z_3_dict["ext"] = self.conv_ext_3(z_2_dict, edge_index_dict, edge_attr_dict)
        z_3_dict["ext"] = torch.sigmoid(z_3_dict["ext"])
        z_3_dict["ext"] = self.linear_ext_3(z_3_dict["ext"])
        z_3_dict["ext"] = torch.sigmoid(z_3_dict["ext"])
        
        # Layer 4
        z_4_dict = {"ind": None, "org": None, "ext": None}
        
        z_4_dict["ind"] = self.conv_ind_4(z_3_dict, edge_index_dict, edge_attr_dict)
        z_4_dict["ind"] = torch.sigmoid(z_4_dict["ind"])
        z_4_dict["ind"] = self.linear_ind_4(z_4_dict["ind"])
        z_4_dict["ind"] = torch.sigmoid(z_4_dict["ind"])
        
        z_4_dict["org"] = self.conv_org_4(z_3_dict, edge_index_dict, edge_attr_dict)
        z_4_dict["org"] = torch.sigmoid(z_4_dict["org"])
        z_4_dict["org"] = self.linear_org_4(z_4_dict["org"])
        z_4_dict["org"] = torch.sigmoid(z_4_dict["org"])
        
        
        return z_4_dict["ind"][:,0], z_4_dict["org"][:,0]
################################################################################################