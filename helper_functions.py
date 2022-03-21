from datetime import datetime
import random, torch, math

# Defining a function that will be used to time runtime
def stopwatch(start_time = False):
    if start_time == False:
        return datetime.now()
    
    t = datetime.now() - start_time
    tms = divmod(t.days * 86400 + t.seconds, 60)
    print("Done in: {} min {} sek".format(tms[0], tms[1]))
    
# Defining a function that will be used to time runtime
def stopwatch_seconds(start_time = False):
    if start_time == False:
        return datetime.now()
    
    t = datetime.now() - start_time
    return t.seconds+t.microseconds*1e-6

# Defining a function that will be used to fetch password from .env-file
def readenvfile(envfile):
    env_vars = {}
    with open(envfile) as f:
        for l in f:
            try:
                key, val = l.split("=")
                env_vars[key] = val.strip()
            except ValueError as e:
                print("Could not read env file - is it properly formatted?")
                #print(l)
                #print(e)
    return env_vars

# Sound Alert
def sound_alert():
    from IPython.display import Audio
    return Audio('/home/ec2-user/SageMaker/repos/fredriks-thesis/audio/bjeffbjeff.wav', autoplay=True)

# ROC CURVES
def plot_roc(train_probs, train_y, test_probs, test_y):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot
    
    pyplot.subplots(figsize=(10, 4))
    
    # calculate probabilities on test set
    # calculate scores - train
    auc_train = roc_auc_score(train_y, train_probs)
    print('Train: ROC AUC=%.3f' % (auc_train))
    # calculate scores - Test
    auc_test = roc_auc_score(test_y, test_probs)
    print('Test: ROC AUC=%.3f' % (auc_test))

    # plot the roc curve - Train
    fpr_train, tpr_train, _ = roc_curve(train_y, train_probs)
    pyplot.plot(fpr_train, tpr_train, marker='.', label='Train')

    # plot the roc curve - test
    fpr, tpr, _ = roc_curve(test_y, test_probs)
    pyplot.plot(fpr, tpr, marker='.', label='Test')
    
    # Plot line for random model
    ns_probs = [0 for _ in range(len(test_y))]
    ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()

def plot_precision_recall(train_probs, train_y, test_probs, test_y):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from matplotlib import pyplot

    pyplot.subplots(figsize=(10, 4))

    # predict class values
    train_precision, train_recall, _ = precision_recall_curve(train_y, train_probs)
    train_auc = auc(train_recall, train_precision)
    print('Train: Precision Recall auc = %.3f' % (train_auc))

    test_precision, test_recall, _ = precision_recall_curve(test_y, test_probs)
    test_auc = auc(test_recall, test_precision)
    print('Test: Precision Recall auc = %.3f' % (test_auc))

    no_skill = len(train_y[train_y==1]) / len(train_y)
    print('No Skill = %.3f' % (no_skill))

    pyplot.plot(train_recall, train_precision, marker='.', label='Train')
    pyplot.plot(test_recall, test_precision, marker='.', label='Test')
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()
    

def pr_auc_score(train_probs, train_y):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    train_precision, train_recall, _ = precision_recall_curve(train_probs, train_y)
    return auc(train_recall, train_precision)
    
################################################################################################
def compute_lift(df, top_frac):
    df = df.sort_values(by = 'score', ascending = False)
    n = df.shape[0]
    n_pos = df[df.y==1].shape[0]
    n_frac = math.ceil(n*top_frac)
    pos_frac = df.head(n_frac).y.value_counts()[1]/n_frac
    lift = pos_frac/(n_pos/n)
    return pos_frac, lift
################################################################################################

################################################################################################
def del_graph(data):
    del data['ind'].x  
    del data['org'].x  
    del data['ext'].x  
    del data['ind'].y  
    del data['org'].y  
    del data['ind'].map  
    del data['org'].map  
    del data['ext'].map 
    del data['ind'].test_mask
    del data['org'].test_mask
    del data['ind'].train_mask
    del data['org'].train_mask
    del data['ind'].cv_mask
    del data['org'].cv_mask

    del data['ind'].num_features
    del data['ind'].num_features
    del data['ext'].num_features
    
    del data['ind','txn','ind'].edge_index 
    del data['org','txn','org'].edge_index 
    del data['ind','txn','org'].edge_index 
    del data['org','txn','ind'].edge_index 
    del data['ind','txn','ext'].edge_index 
    del data['org','txn','ext'].edge_index 
    del data['ext','txn','org'].edge_index 
    del data['ext','txn','ind'].edge_index 
    del data['ind','role','org'].edge_index 
    del data['ind','txn','ind'].edge_attr 
    del data['org','txn','org'].edge_attr 
    del data['ind','txn','org'].edge_attr 
    del data['org','txn','ind'].edge_attr 
    del data['ind','txn','ext'].edge_attr 
    del data['org','txn','ext'].edge_attr 
    del data['ext','txn','org'].edge_attr 
    del data['ext','txn','ind'].edge_attr 
    del data['ind','role','org'].edge_attr 
    del data['ind','txn','ind']
    del data['org','txn','org']
    del data['ind','txn','org']
    del data['org','txn','ind']
    del data['ind','txn','ext']
    del data['org','txn','ext']
    del data['ext','txn','org']
    del data['ext','txn','ind']
    del data['ind','role','org'] 
    
    del data['ind','rev_txn','ind'].edge_index 
    del data['org','rev_txn','org'].edge_index 
    del data['ind','rev_txn','org'].edge_index 
    del data['org','rev_txn','ind'].edge_index 
    del data['ind','rev_txn','ext'].edge_index 
    del data['org','rev_txn','ext'].edge_index 
    del data['ext','rev_txn','org'].edge_index 
    del data['ext','rev_txn','ind'].edge_index 
    del data['org','rev_role','ind'].edge_index 
    del data['ind','rev_txn','ind'].edge_attr 
    del data['org','rev_txn','org'].edge_attr 
    del data['ind','rev_txn','org'].edge_attr 
    del data['org','rev_txn','ind'].edge_attr 
    del data['ind','rev_txn','ext'].edge_attr 
    del data['org','rev_txn','ext'].edge_attr 
    del data['ext','rev_txn','org'].edge_attr 
    del data['ext','rev_txn','ind'].edge_attr 
    del data['org','rev_role','ind'].edge_attr 
    del data['ind','rev_txn','ind']
    del data['org','rev_txn','org']
    del data['ind','rev_txn','org']
    del data['org','rev_txn','ind']
    del data['ind','rev_txn','ext']
    del data['org','rev_txn','ext']
    del data['ext','rev_txn','org']
    del data['ext','rev_txn','ind']
    del data['org','rev_role','ind']
    del data['ind']
    del data['org']
    del data['ext']
    del data['map']
    del data['cv_mask']
    del data
################################################################################################

################################################################################################
# Compute the number of parameters in a model
def get_num_params(model):
    total_dim  = 0 
    for param in model.parameters():
        my_shape = param.data.shape
        dim = 1
        for i in range(len(my_shape)): dim = dim*max(my_shape[i],1)
        total_dim+= dim
    return total_dim
################################################################################################