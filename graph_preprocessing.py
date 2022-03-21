import torch, torch_geometric, math, random



################################################################################################
def create_test_mask(y, p):
    n = y.shape[0]    
    pos_indices = (y == True).nonzero(as_tuple=True)[0]
    n_pos = pos_indices.shape[0]
    n_pos_draws = math.ceil(n_pos*p)
    pos_indices_test = pos_indices[random.sample(range(0, n_pos), n_pos_draws)]

    neg_indices = (y == False).nonzero(as_tuple=True)[0]
    n_neg = neg_indices.shape[0]
    n_neg_draws = math.ceil(n_neg*p)
    neg_indices_test = neg_indices[random.sample(range(0, n_neg), n_neg_draws)]

    indices_test = torch.concat((pos_indices_test, neg_indices_test))
    test_mask = torch.zeros(n, dtype=bool)
    test_mask[indices_test] = True
    return test_mask
################################################################################################



################################################################################################
def create_test_mask_old(y, p):
    n = y.shape[0]
    n_0 = y.unique(return_counts = True)[1][0]
    n_1 = y.unique(return_counts = True)[1][1]
    
    indices_0 = torch.tensor(np.arange(n))[y==0]
    indices_1 = torch.tensor(np.arange(n))[y==1]
    good_split = False
    while not good_split:
        indices_1_test = indices_1[torch.rand(indices_1.shape[0]) >= (1-p)]
        indices_0_test = indices_0[torch.rand(indices_0.shape[0]) >= (1-p)]
        test_mask = torch.zeros(n, dtype=bool)
        test_mask[indices_1_test] = True
        test_mask[indices_0_test] = True
        
        if abs(len(indices_1_test)/n_1-p)<0.005: good_split = True
    return test_mask
################################################################################################

################################################################################################
from sklearn.model_selection import KFold 
import numpy as np
def create_cv_mask(train_mask, n_folds):
    n = len(train_mask)
    train_indices = torch.tensor(np.arange(n))[train_mask]
    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state = 1).split(train_indices))
    
    cv_mask = torch.zeros(n, dtype = torch.int)
    for i in range(len(folds)):
        cv_mask[train_indices[folds[i][1]]]=i+1

    return cv_mask
################################################################################################


################################################################################################
def reverse_edges(data):
    # Reversing all edges 
    # Reversing ('ind', 'txn', 'ind')
    data['ind','rev_txn','ind'].edge_attr = data[('ind', 'txn', 'ind')].edge_attr
    data['ind','rev_txn','ind'].edge_index = torch.zeros([2, data[('ind', 'txn', 'ind')].edge_index.shape[1]], dtype = torch.int64)
    data['ind','rev_txn','ind'].edge_index[0,:] = data[('ind', 'txn', 'ind')].edge_index[1,:]
    data['ind','rev_txn','ind'].edge_index[1,:] = data[('ind', 'txn', 'ind')].edge_index[0,:]
    # Reversing ('org', 'txn', 'org')
    data['org','rev_txn','org'].edge_attr = data[('org', 'txn', 'org')].edge_attr
    data['org','rev_txn','org'].edge_index = torch.zeros([2, data[('org', 'txn', 'org')].edge_index.shape[1]], dtype = torch.int64)
    data['org','rev_txn','org'].edge_index[0,:] = data[('org', 'txn', 'org')].edge_index[1,:]
    data['org','rev_txn','org'].edge_index[1,:] = data[('org', 'txn', 'org')].edge_index[0,:]
    # Reversing ('ind', 'txn', 'org')
    data['org','rev_txn','ind'].edge_attr = data[('ind', 'txn', 'org')].edge_attr
    data['org','rev_txn','ind'].edge_index = torch.zeros([2, data[('ind', 'txn', 'org')].edge_index.shape[1]], dtype = torch.int64)
    data['org','rev_txn','ind'].edge_index[0,:] = data[('ind', 'txn', 'org')].edge_index[1,:]
    data['org','rev_txn','ind'].edge_index[1,:] = data[('ind', 'txn', 'org')].edge_index[0,:]
    # Reversing ('org', 'txn', 'ind')
    data['ind','rev_txn','org'].edge_attr = data[('org', 'txn', 'ind')].edge_attr
    data['ind','rev_txn','org'].edge_index = torch.zeros([2, data[('org', 'txn', 'ind')].edge_index.shape[1]], dtype = torch.int64)
    data['ind','rev_txn','org'].edge_index[0,:] = data[('org', 'txn', 'ind')].edge_index[1,:]
    data['ind','rev_txn','org'].edge_index[1,:] = data[('org', 'txn', 'ind')].edge_index[0,:]
    # Reversing ('ind', 'txn', 'ext')
    data['ext','rev_txn','ind'].edge_attr = data[('ind', 'txn', 'ext')].edge_attr
    data['ext','rev_txn','ind'].edge_index = torch.zeros([2, data[('ind', 'txn', 'ext')].edge_index.shape[1]], dtype = torch.int64)
    data['ext','rev_txn','ind'].edge_index[0,:] = data[('ind', 'txn', 'ext')].edge_index[1,:]
    data['ext','rev_txn','ind'].edge_index[1,:] = data[('ind', 'txn', 'ext')].edge_index[0,:]
    # Reversing ('ext', 'txn', 'ind')
    data['ind','rev_txn','ext'].edge_attr = data[('ext', 'txn', 'ind')].edge_attr
    data['ind','rev_txn','ext'].edge_index = torch.zeros([2, data[('ext', 'txn', 'ind')].edge_index.shape[1]], dtype = torch.int64)
    data['ind','rev_txn','ext'].edge_index[0,:] = data[('ext', 'txn', 'ind')].edge_index[1,:]
    data['ind','rev_txn','ext'].edge_index[1,:] = data[('ext', 'txn', 'ind')].edge_index[0,:]
    # Reversing ('org', 'txn', 'ext')
    data['ext','rev_txn','org'].edge_attr = data[('org', 'txn', 'ext')].edge_attr
    data['ext','rev_txn','org'].edge_index = torch.zeros([2, data[('org', 'txn', 'ext')].edge_index.shape[1]], dtype = torch.int64)
    data['ext','rev_txn','org'].edge_index[0,:] = data[('org', 'txn', 'ext')].edge_index[1,:]
    data['ext','rev_txn','org'].edge_index[1,:] = data[('org', 'txn', 'ext')].edge_index[0,:]
    # Reversing ('ext', 'txn', 'org')
    data['org','rev_txn','ext'].edge_attr = data[('ext', 'txn', 'org')].edge_attr
    data['org','rev_txn','ext'].edge_index = torch.zeros([2, data[('ext', 'txn', 'org')].edge_index.shape[1]], dtype = torch.int64)
    data['org','rev_txn','ext'].edge_index[0,:] = data[('ext', 'txn', 'org')].edge_index[1,:]
    data['org','rev_txn','ext'].edge_index[1,:] = data[('ext', 'txn', 'org')].edge_index[0,:]
    # Reversing ('ind', 'role', 'org')
    data['org','rev_role','ind'].edge_attr = data[('ind', 'role', 'org')].edge_attr
    data['org','rev_role','ind'].edge_index = torch.zeros([2, data[('ind', 'role', 'org')].edge_index.shape[1]], dtype = torch.int64)
    data['org','rev_role','ind'].edge_index[0,:] = data[('ind', 'role', 'org')].edge_index[1,:]
    data['org','rev_role','ind'].edge_index[1,:] = data[('ind', 'role', 'org')].edge_index[0,:]
    return data
################################################################################################

################################################################################################
def apply_log_to_txns(data):
    # Applying log to node feature transaction amounts: 
    data['ind'].x[:, 5], data['ind'].x[:, 7] = torch.log10(data['ind'].x[:, 5]+1), torch.log10(data['ind'].x[:, 7]+1)
    data['org'].x[:, 4], data['org'].x[:, 6] = torch.log10(data['org'].x[:, 4]+1), torch.log10(data['org'].x[:, 6]+1)
    # Applying log to edge feature transaction amounts: 
    data['ind','txn','ind'].edge_attr[:,1] = torch.log10(data['ind','txn','ind'].edge_attr[:,1]+1)
    data['org','txn','org'].edge_attr[:,1] = torch.log10(data['org','txn','org'].edge_attr[:,1]+1)
    data['ind','txn','org'].edge_attr[:,1] = torch.log10(data['ind','txn','org'].edge_attr[:,1]+1)
    data['org','txn','ind'].edge_attr[:,1] = torch.log10(data['org','txn','ind'].edge_attr[:,1]+1)
    data['ind','txn','ext'].edge_attr[:,1] = torch.log10(data['ind','txn','ext'].edge_attr[:,1]+1)
    data['ext','txn','ind'].edge_attr[:,1] = torch.log10(data['ext','txn','ind'].edge_attr[:,1]+1)
    data['org','txn','ext'].edge_attr[:,1] = torch.log10(data['org','txn','ext'].edge_attr[:,1]+1)
    data['ext','txn','org'].edge_attr[:,1] = torch.log10(data['ext','txn','org'].edge_attr[:,1]+1)
    data['ind','rev_txn','ind'].edge_attr[:,1] = torch.log10(data['ind','rev_txn','ind'].edge_attr[:,1]+1)
    data['org','rev_txn','org'].edge_attr[:,1] = torch.log10(data['org','rev_txn','org'].edge_attr[:,1]+1)
    data['ind','rev_txn','org'].edge_attr[:,1] = torch.log10(data['ind','rev_txn','org'].edge_attr[:,1]+1)
    data['org','rev_txn','ind'].edge_attr[:,1] = torch.log10(data['org','rev_txn','ind'].edge_attr[:,1]+1)
    data['ind','rev_txn','ext'].edge_attr[:,1] = torch.log10(data['ind','rev_txn','ext'].edge_attr[:,1]+1)
    data['ext','rev_txn','ind'].edge_attr[:,1] = torch.log10(data['ext','rev_txn','ind'].edge_attr[:,1]+1)
    data['org','rev_txn','ext'].edge_attr[:,1] = torch.log10(data['org','rev_txn','ext'].edge_attr[:,1]+1)
    data['ext','rev_txn','org'].edge_attr[:,1] = torch.log10(data['ext','rev_txn','org'].edge_attr[:,1]+1)
    return data
################################################################################################

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
################################################################################################
def normalize_node_features(data):
    # Normalizing node features
    data['ind'].x = torch.tensor(StandardScaler().fit_transform(data['ind'].x.detach().numpy()), dtype = torch.float32)
    data['org'].x = torch.tensor(StandardScaler().fit_transform(data['org'].x.detach().numpy()), dtype = torch.float32)
    return data
################################################################################################

################################################################################################
def scaling_edge_attr(data):
    data['ind','txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ind','txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ind','txn','ext'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','ext'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','txn','ext'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ext','txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','txn','ext'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','ext'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','txn','ext'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ext','txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ind','rev_txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','rev_txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ind','rev_txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','rev_txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ind','rev_txn','ext'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','ext'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ind','rev_txn','ext'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ext','rev_txn','ind'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','rev_txn','ind'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','rev_txn','ind'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['org','rev_txn','ext'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','ext'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['org','rev_txn','ext'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    data['ext','rev_txn','org'].edge_attr = torch.cat((torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','rev_txn','org'].edge_attr[:,1:2].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1],
                                                   torch.tensor(MinMaxScaler(feature_range = (0.01,1)).fit_transform(data['ext','rev_txn','org'].edge_attr[:,0:1].detach().numpy().reshape(-1, 1)), dtype = torch.float32)[:,0:1]), 1)
    return data
################################################################################################

################################################################################################
def remove_num_txns(data):
    data['ind','txn','ind'].edge_attr     = data['ind','txn','ind'].edge_attr[:,1:2]
    data['org','txn','org'].edge_attr     = data['org','txn','org'].edge_attr[:,1:2]
    data['ind','txn','org'].edge_attr     = data['ind','txn','org'].edge_attr[:,1:2] 
    data['org','txn','ind'].edge_attr     = data['org','txn','ind'].edge_attr[:,1:2]
    data['ind','txn','ext'].edge_attr     = data['ind','txn','ext'].edge_attr[:,1:2] 
    data['ext','txn','ind'].edge_attr     = data['ext','txn','ind'].edge_attr[:,1:2]
    data['org','txn','ext'].edge_attr     = data['org','txn','ext'].edge_attr[:,1:2]
    data['ext','txn','org'].edge_attr     = data['ext','txn','org'].edge_attr[:,1:2] 
    data['ind','rev_txn','ind'].edge_attr = data['ind','rev_txn','ind'].edge_attr[:,1:2]
    data['org','rev_txn','org'].edge_attr = data['org','rev_txn','org'].edge_attr[:,1:2]
    data['ind','rev_txn','org'].edge_attr = data['ind','rev_txn','org'].edge_attr[:,1:2]
    data['org','rev_txn','ind'].edge_attr = data['org','rev_txn','ind'].edge_attr[:,1:2]
    data['ind','rev_txn','ext'].edge_attr = data['ind','rev_txn','ext'].edge_attr[:,1:2]
    data['ext','rev_txn','ind'].edge_attr = data['ext','rev_txn','ind'].edge_attr[:,1:2]
    data['org','rev_txn','ext'].edge_attr = data['org','rev_txn','ext'].edge_attr[:,1:2]
    data['ext','rev_txn','org'].edge_attr = data['ext','rev_txn','org'].edge_attr[:,1:2]
    return data
################################################################################################