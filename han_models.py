import torch, torch_geometric

from torch_geometric.nn import HANConv
from torch.nn import Linear



################################################################################################
# One Layer HANConv
from typing import Union, Dict, List
class HAN1Layer(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
    
        self.han_conv = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind = Linear(hidden_channels, 1)
        self.lin_org = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.han_conv(x_dict, edge_index_dict)
        pred_ind = self.lin_ind(x_dict['ind'])
        pred_ind = torch.sigmoid(pred_ind)
        pred_org = self.lin_org(x_dict['org'])
        pred_org = torch.sigmoid(pred_org)
        return pred_ind[:,0], pred_org[:,0]
################################################################################################


################################################################################################
# Two Layer HANConv
from typing import Union, Dict, List
class HAN2Layer(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
        
        # Layer 1
        self.conv_1 = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_1 = Linear(hidden_channels, num_features_ind)
        self.lin_org_1 = Linear(hidden_channels, num_features_org)
        self.lin_ext_1 = Linear(hidden_channels, num_features_ext)
        
        # Layer 2
        self.conv_2 = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_2 = Linear(hidden_channels, 1)
        self.lin_org_2 = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z1 = self.conv_1(x_dict, edge_index_dict)
        
        z1['ind'] = self.lin_ind_1(z1['ind'])
        z1['ind'] = torch.sigmoid(z1['ind'])
        
        z1['org'] = self.lin_org_1(z1['org'])
        z1['org'] = torch.sigmoid(z1['org'])
        
        z1['ext'] = self.lin_ext_1(z1['ext'])
        z1['ext'] = torch.sigmoid(z1['ext'])
        
        # Layer 2
        z2 = self.conv_2(z1, edge_index_dict)
        
        z2['ind'] = self.lin_ind_2(z2['ind'])
        z2['ind'] = torch.sigmoid(z2['ind'])
        
        z2['org'] = self.lin_org_2(z2['org'])
        z2['org'] = torch.sigmoid(z2['org'])
        
        return z2['ind'][:,0], z2['org'][:,0]
################################################################################################



################################################################################################
# Three Layer HANConv
from typing import Union, Dict, List
class HAN3Layer(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
        
        # Layer 1
        self.conv_1 = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_1 = Linear(hidden_channels, num_features_ind)
        self.lin_org_1 = Linear(hidden_channels, num_features_org)
        self.lin_ext_1 = Linear(hidden_channels, num_features_ext)
        
        # Layer 2
        self.conv_2 = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_2 = Linear(hidden_channels, num_features_ind)
        self.lin_org_2 = Linear(hidden_channels, num_features_org)
        self.lin_ext_2 = Linear(hidden_channels, num_features_ext)
        
        # Layer 3
        self.conv_3 = HANConv(-1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_3 = Linear(hidden_channels, 1)
        self.lin_org_3 = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z1 = self.conv_1(x_dict, edge_index_dict)
        
        z1['ind'] = self.lin_ind_1(z1['ind'])
        z1['ind'] = torch.sigmoid(z1['ind'])
        
        z1['org'] = self.lin_org_1(z1['org'])
        z1['org'] = torch.sigmoid(z1['org'])
        
        z1['ext'] = self.lin_ext_1(z1['ext'])
        z1['ext'] = torch.sigmoid(z1['ext'])
        
        # Layer 2
        z2 = self.conv_2(z1, edge_index_dict)
        
        z2['ind'] = self.lin_ind_2(z2['ind'])
        z2['ind'] = torch.sigmoid(z2['ind'])
        
        z2['org'] = self.lin_org_2(z2['org'])
        z2['org'] = torch.sigmoid(z2['org'])
        
        z2['ext'] = self.lin_ext_2(z2['ext'])
        z2['ext'] = torch.sigmoid(z2['ext'])
        
        # Layer 3
        z3 = self.conv_3(z2, edge_index_dict)
        
        z3['ind'] = self.lin_ind_3(z3['ind'])
        z3['ind'] = torch.sigmoid(z3['ind'])
        
        z3['org'] = self.lin_org_3(z3['org'])
        z3['org'] = torch.sigmoid(z3['org'])
        
        return z3['ind'][:,0], z3['org'][:,0]
################################################################################################




################################################################################################
# One Layer HANConv
from typing import Union, Dict, List
class HAN1Layer_new(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
        
        my_dict = {"ind": num_features_ind, "org": num_features_org, "ext": num_features_ext}
    
        self.han_conv = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind = Linear(hidden_channels, 1)
        self.lin_org = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.han_conv(x_dict, edge_index_dict)
        pred_ind = self.lin_ind(x_dict['ind'])
        pred_ind = torch.sigmoid(pred_ind)
        pred_org = self.lin_org(x_dict['org'])
        pred_org = torch.sigmoid(pred_org)
        return pred_ind[:,0], pred_org[:,0]
################################################################################################

################################################################################################
# Two Layer HANConv
from typing import Union, Dict, List
class HAN2Layer_new(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
        
        my_dict = {"ind": num_features_ind, "org": num_features_org, "ext": num_features_ext}
        
        # Layer 1
        self.conv_1 = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_1 = Linear(hidden_channels, num_features_ind)
        self.lin_org_1 = Linear(hidden_channels, num_features_org)
        self.lin_ext_1 = Linear(hidden_channels, num_features_ext)
        
        # Layer 2
        self.conv_2 = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_2 = Linear(hidden_channels, 1)
        self.lin_org_2 = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z1 = self.conv_1(x_dict, edge_index_dict)
        
        z1['ind'] = self.lin_ind_1(z1['ind'])
        z1['ind'] = torch.sigmoid(z1['ind'])
        
        z1['org'] = self.lin_org_1(z1['org'])
        z1['org'] = torch.sigmoid(z1['org'])
        
        z1['ext'] = self.lin_ext_1(z1['ext'])
        z1['ext'] = torch.sigmoid(z1['ext'])
        
        # Layer 2
        z2 = self.conv_2(z1, edge_index_dict)
        
        z2['ind'] = self.lin_ind_2(z2['ind'])
        z2['ind'] = torch.sigmoid(z2['ind'])
        
        z2['org'] = self.lin_org_2(z2['org'])
        z2['org'] = torch.sigmoid(z2['org'])
        
        return z2['ind'][:,0], z2['org'][:,0]
################################################################################################


################################################################################################
# Three Layer HANConv
from typing import Union, Dict, List
class HAN3Layer_new(torch.nn.Module):        
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge, metadata):
        super().__init__()
        
        hidden_channels=8
        heads = 1
        dropout = 0
        
        my_dict = {"ind": num_features_ind, "org": num_features_org, "ext": num_features_ext}
        
        # Layer 1
        self.conv_1 = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_1 = Linear(hidden_channels, num_features_ind)
        self.lin_org_1 = Linear(hidden_channels, num_features_org)
        self.lin_ext_1 = Linear(hidden_channels, num_features_ext)
        
        # Layer 2
        self.conv_2 = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_2 = Linear(hidden_channels, num_features_ind)
        self.lin_org_2 = Linear(hidden_channels, num_features_org)
        self.lin_ext_2 = Linear(hidden_channels, num_features_ext)
        
        # Layer 3
        self.conv_3 = HANConv(my_dict, hidden_channels, heads=heads, dropout=dropout, metadata=metadata)
        self.lin_ind_3 = Linear(hidden_channels, 1)
        self.lin_org_3 = Linear(hidden_channels, 1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        
        # Layer 1
        z1 = self.conv_1(x_dict, edge_index_dict)
        
        z1['ind'] = self.lin_ind_1(z1['ind'])
        z1['ind'] = torch.sigmoid(z1['ind'])
        
        z1['org'] = self.lin_org_1(z1['org'])
        z1['org'] = torch.sigmoid(z1['org'])
        
        z1['ext'] = self.lin_ext_1(z1['ext'])
        z1['ext'] = torch.sigmoid(z1['ext'])
        
        # Layer 2
        z2 = self.conv_2(z1, edge_index_dict)
        
        z2['ind'] = self.lin_ind_2(z2['ind'])
        z2['ind'] = torch.sigmoid(z2['ind'])
        
        z2['org'] = self.lin_org_2(z2['org'])
        z2['org'] = torch.sigmoid(z2['org'])
        
        z2['ext'] = self.lin_ext_2(z2['ext'])
        z2['ext'] = torch.sigmoid(z2['ext'])
        
        # Layer 3
        z3 = self.conv_3(z2, edge_index_dict)
        
        z3['ind'] = self.lin_ind_3(z3['ind'])
        z3['ind'] = torch.sigmoid(z3['ind'])
        
        z3['org'] = self.lin_org_3(z3['org'])
        z3['org'] = torch.sigmoid(z3['org'])
        
        return z3['ind'][:,0], z3['org'][:,0]
################################################################################################