import torch
from torch_geometric.nn import HeteroConv, NNConv
from torch.nn.functional import leaky_relu, relu
from torch import sigmoid

################################################################################################
class logistic_01(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        self.linear_ind = torch.nn.Linear(num_features_ind,1)
        
        self.linear_org = torch.nn.Linear(num_features_org,1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):

        pred_ind = self.linear_ind(x_dict["ind"])
        pred_ind = sigmoid(pred_ind)
        
        pred_org = self.linear_org(x_dict["org"])
        pred_org = sigmoid(pred_org)
        
        return pred_ind[:,0], pred_org[:,0]
################################################################################################


################################################################################################
class logistic_02(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        self.linear_ind_1 = torch.nn.Linear(num_features_ind,num_features_ind)
        self.linear_ind_2 = torch.nn.Linear(num_features_ind,1)
        
        self.linear_org_1 = torch.nn.Linear(num_features_org,num_features_org)
        self.linear_org_2 = torch.nn.Linear(num_features_org,1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):

        pred_ind = self.linear_ind_1(x_dict["ind"])
        pred_ind = sigmoid(pred_ind)
        pred_ind = self.linear_ind_2(pred_ind)
        pred_ind = sigmoid(pred_ind)
        
        pred_org = self.linear_org_1(x_dict["org"])
        pred_org = sigmoid(pred_org)
        pred_org = self.linear_org_2(pred_org)
        pred_org = sigmoid(pred_org)
        
        return pred_ind[:,0], pred_org[:,0]
################################################################################################

################################################################################################
class logistic_03(torch.nn.Module):
    def __init__(self, num_features_ind,  num_features_org, num_features_ext, num_features_txn_edge):
        super().__init__()
        
        self.linear_ind_1 = torch.nn.Linear(num_features_ind,num_features_ind)
        self.linear_ind_2 = torch.nn.Linear(num_features_ind,num_features_ind)
        self.linear_ind_3 = torch.nn.Linear(num_features_ind,1)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):

        pred_ind = self.linear_ind_1(x_dict["ind"])
        pred_ind = leaky_relu(pred_ind)

        pred_ind = self.linear_ind_2(pred_ind)
        pred_ind = leaky_relu(pred_ind)
        
        pred_ind = self.linear_ind_3(pred_ind)
        pred_ind = sigmoid(pred_ind)
        
        pred_ind = pred_ind[:,0]
        
        return pred_ind, x_dict['org'][:,0]
################################################################################################


# leaky_relu

