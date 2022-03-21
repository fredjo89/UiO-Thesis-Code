# UiO-Thesis-Code
Contains Python-code created in my work with the thesis: Finding Money Launderers Using Heterogeneous Graph Neural Networks.

The notebook train_test contains the script to train the models using cross validation, and test the resulting models on the test-set. 

In the folder "models" are the implemented models explored. 

In the folder "node_features" are the notebooks used to create the additional graph-features for the nodes. 

The python-file graph_preprocessing contains functions to preprocessing the input-data. These are used in the beginning of the train_test-script. 

The python-file helper_functions contains some additional helper-functions used in the train_test-script, such as computing ROC-curves.
