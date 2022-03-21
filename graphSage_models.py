import torch, torch_geometric
    
################################################################################################
# One Layer SAGEConv
class SAGEConv1Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(SAGEConv1Layer, self).__init__()

        self.conv = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################
    
    
################################################################################################
# Two Layer SAGEConv
class SAGEConv2Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(SAGEConv2Layer, self).__init__()

        self.conv1 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv2 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv2(x_dict, edge_index_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################
    


################################################################################################
# Three Layer SAGEConv
class SAGEConv3Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(SAGEConv3Layer, self).__init__()

        self.conv1 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv2 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv3 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            #('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), 1, aggr="mean"),
            #('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), 1, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) # Can test this with relu
        x_dict['org'] = torch.sigmoid(x_dict['org']) # Can test this with relu
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) # Can test this with relu
        x_dict = self.conv3(x_dict, edge_index_dict)
        pred_ind = torch.sigmoid(x_dict['ind'])[:,0]
        pred_org = torch.sigmoid(x_dict['org'])[:,0]
        return pred_ind, pred_org
################################################################################################
    

    
    
################################################################################################
# Four Layer SAGEConv
class SAGEConv4Layer(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super(SAGEConv4Layer, self).__init__()

        self.conv1 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv2 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv3 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), num_features_ind, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), num_features_ind, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), num_features_org, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), num_features_org, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), num_features_org, aggr="mean"),
            ('ind', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_txn', 'ext'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ext), num_features_ext, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), num_features_ind, aggr="mean"),
        }, aggr = 'sum')
        
        self.conv4 = torch_geometric.nn.HeteroConv({
            ('ind', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            ('ind', 'role', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
        
            ('ind', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_ind), 1, aggr="mean"),
            ('org', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
            ('ext', 'rev_txn', 'ind'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_ind), 1, aggr="mean"),
            ('ind', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ind, num_features_org), 1, aggr="mean"),
            ('org', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_org, num_features_org), 1, aggr="mean"),
            ('ext', 'rev_txn', 'org'): torch_geometric.nn.SAGEConv((num_features_ext, num_features_org), 1, aggr="mean"),
            ('org', 'rev_role', 'ind'): torch_geometric.nn.SAGEConv((num_features_org, num_features_ind), 1, aggr="mean"),
        }, aggr = 'sum')
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) 
        x_dict['org'] = torch.sigmoid(x_dict['org']) 
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) 
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) 
        x_dict['org'] = torch.sigmoid(x_dict['org']) 
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) 
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind']) 
        x_dict['org'] = torch.sigmoid(x_dict['org']) 
        x_dict['ext'] = torch.sigmoid(x_dict['ext']) 
        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict['ind'] = torch.sigmoid(x_dict['ind'])
        x_dict['org'] = torch.sigmoid(x_dict['org'])
        
        return x_dict['ind'][:,0], x_dict['org'][:,0]
################################################################################################