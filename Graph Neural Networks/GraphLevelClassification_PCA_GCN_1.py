import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
import os
import scipy.sparse as sp
import tensorflow as tf
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, Dropout, ReLU, Sigmoid
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def set_seed(seed):
    random.seed(seed)  # Seed for the random module
    np.random.seed(seed)  # Seed for NumPy
    torch.manual_seed(seed)  # Seed for PyTorch
    torch.cuda.manual_seed(seed)  # Seed for current GPU
    torch.cuda.manual_seed_all(seed)  # Seed for all GPUs (if you have more than one)

# Set the desired seed
set_seed(42)

file_str="tadv_5min_wave_group_window_15mins_4"

data=np.load(file_str+".npz")
wave_data_train=data["wave_data_train"]
wave_data_test=data["wave_data_test"]
label_train=data["label_train"]
label_test=data["label_test"]

wave_data_train = wave_data_train.reshape(wave_data_train.shape[0], wave_data_train.shape[1])
wave_data_test = wave_data_test.reshape(wave_data_test.shape[0], wave_data_test.shape[1])
num_classes=2

wave_data_train_standardized = (wave_data_train- np.mean(wave_data_train, axis=0)) / np.std(wave_data_train, axis=0)
wave_data_test_standardized = (wave_data_test- np.mean(wave_data_test, axis=0)) / np.std(wave_data_test, axis=0)

pca = PCA()
pca.fit(wave_data_train_standardized)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components needed to explain 90% and 95% of the variance
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

n_components_req = n_components_90
pca_new = PCA(n_components=n_components_req)
wave_data_train_reconstructed = pca_new.fit_transform(wave_data_train_standardized).reshape(wave_data_train.shape[0], n_components_req, 1)

wave_data_test_reconstructed = pca_new.transform(wave_data_test_standardized).reshape(wave_data_test.shape[0], n_components_req, 1)

num_nodes = wave_data_train_reconstructed.shape[1]
num_node_features = wave_data_train_reconstructed.shape[2]
num_neighbours = 30  ### Choice for the number of neighbours in the graph structure
batch_size = 32  ### Batch size to be used for creating dataloaders

print("----------------The graph structure will be in the form of multiple graphs------------------".upper())
print("--Each graph will have multiple nodes where each node is an observation of the time series--".upper())
print("----------------------------The feature length of each node is 1----------------------------".upper())
print('\n')
print('-----------------------The graph properties are given below---------------------\n'.upper())
print(f"The number of nodes in each graph is {num_nodes}")
print(f"The number of features for each node is {num_node_features}")
print(f"The number of training set examples is {wave_data_train.shape[0]}")
print(f"The number of test set examples is {wave_data_test.shape[0]}")
print(f"The number of neighbours chosen for each node is {num_neighbours}")

def create_adjacency_matrix(num_nodes, directionality=True):
    D_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            D_matrix[i,j] = abs(i-j)

    k = num_neighbours
    A = np.zeros_like(D_matrix)
    for i in range(len(A)):
        neighbours = np.argsort(D_matrix[i])[:k]
        if directionality:
            for n in neighbours:
                if n>i:
                    A[i, n] = 0
                else:
                    A[i, n] = 1 
        else:
            A[i, neighbours] = 1 

    return A

directionality = False
A = create_adjacency_matrix(num_nodes, directionality)
sparse_matrix = sp.coo_matrix(A)
indices = np.column_stack((sparse_matrix.nonzero()))
values = sparse_matrix.data
dense_shape = sparse_matrix.shape
sparse_A = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

print("---------Details required for constructing the edge matrices-----------\n".upper())
print(f"Are the edges directed? {directionality}")
print(f"The number of edges in each graph is {len(values)}")

edge_file_path = f"edge_data_15_4_pca_num_neighbors_{num_neighbours}.npz"

if os.path.exists(edge_file_path):
    edge_data=np.load(edge_file_path)
    edge_data_train=edge_data["edge_data_train"]
    edge_data_test=edge_data["edge_data_test"]

else:
    edge_features_train = np.zeros((wave_data_train.shape[0], len(indices)))
    edge_features_test = np.zeros((wave_data_test.shape[0], len(indices)))

    for i in range(edge_features_train.shape[0]):
        for j in range(len(indices)):
            index = indices[j]
            edge_features_train[i,j] = abs(index[1]-index[0])

    for i in range(edge_features_test.shape[0]):
        for j in range(len(indices)):
            index = indices[j]
            edge_features_test[i,j] = abs(index[1]-index[0])

    train_min = np.min(edge_features_train)
    train_max = np.max(edge_features_train)

    test_min = np.min(edge_features_test)
    test_max = np.max(edge_features_test)

    edge_features_train = (edge_features_train - train_min) / (train_max - train_min)
    edge_features_test = (edge_features_test - test_min) / (test_max - test_min)

    np.savez(edge_file_path,edge_data_train=edge_features_train, edge_data_test=edge_features_test)

    edge_data=np.load(edge_file_path)
    edge_data_train=edge_data["edge_data_train"]
    edge_data_test=edge_data["edge_data_test"]

print(f"\nThe shape of the edge weights for training is: {edge_data_train.shape}")
print(f"The shape of the edge weights for testing is: {edge_data_test.shape}")

def create_and_save_graphs(edge_tensor, node_tensor, labels, indices, path):

    graphs = []  
    edge_data = torch.tensor([indices[:,0], indices[:,1]])
        
    num_graphs = len(node_tensor)
    for k in range(num_graphs):
        x = node_tensor[k]
        edge_weights = edge_tensor[k]
        y = labels[k]

        graph = Data(x=x, edge_index=edge_data, edge_weight=edge_weights, y=y)
        graphs.append(graph)

    torch.save(graphs, path)

    return graphs

### Creating the training, validation and the test datasets
train_val_split = 0.7

### The paths to the saved datasets
train_data_path = 'train_set_pca_graph_classification.pth'
val_data_path = 'val_set_pca_graph_classification.pth'
test_data_path = 'test_set_pca_graph_classification.pth'

if os.path.exists(train_data_path) and os.path.exists(val_data_path) and os.path.exists(test_data_path):
    train_dataset = torch.load(train_data_path)
    val_dataset = torch.load(val_data_path)
    test_dataset = torch.load(test_data_path)

else:
    train_dataset = create_and_save_graphs(edge_data_train[:int(train_val_split * len(edge_data_train)),:], wave_data_train_reconstructed[:int(train_val_split * len(wave_data_train_reconstructed)),:], label_train[:int(train_val_split * len(label_train))], indices, train_data_path)
    val_dataset = create_and_save_graphs(edge_data_train[int(train_val_split * len(edge_data_train)):,:], wave_data_train_reconstructed[int(train_val_split * len(wave_data_train_reconstructed)):,:], label_train[int(train_val_split * len(label_train)):], indices, val_data_path)
    test_dataset = create_and_save_graphs(edge_data_test, wave_data_test_reconstructed, label_test, indices, test_data_path)

print("\n ----------------contains the dataset information for the training, validation and test data--------------------\n".upper())
print(f"The training dataset has {len(train_dataset)} graphs".upper())
print(f"The validation dataset has {len(val_dataset)} graphs".upper())
print(f"The testing dataset has {len(test_dataset)} graphs".upper())

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

## Printing the info of a batch 
for batch in train_dataloader:
    print("Batch Node Features (x) shape:")
    print(batch.x[0].shape)  # Batched node features
    print("Batch Labels (y) shape:")
    print(batch.y.shape)  # Batched node features
    print("\nBatch Edge Indices (edge_index) shape:")
    print(batch.edge_index.to(torch.int64).dtype)  # Batched edge indices
    print("\nBatch Edge Weights (edge_weight) shape:")
    print(batch.edge_weight[0].shape)  # Batched edge weights
    print("\nBatch Information (batch) shape:")
    print(batch.batch.shape)  # Batch info: indicates graph membership of each node
    print("\nBatch Size:", len(batch.x[0]), "nodes")
    print("Number of Graphs in this Batch:", batch.batch.max().item() + 1)
    # print((torch.tensor(batch.x)).reshape(len(batch.x) * len(batch.x[0]),1).shape)
    # print((torch.tensor(batch.edge_weight)).reshape(len(batch.x) * len(batch.edge_weight[0])).shape)
    break  # Exit after the first batch

class GraphConvolutionalNetwork(torch.nn.Module):
    """
    Creates a gnn model based on global pooling of embeddings - for graph level classifications
    """
    def __init__(self, dim_pre_MLP, dim_post_MLP, dim_graphLin, num_pre_layers, num_post_layers, dropout_prob, num_graph_layers, training=True):
        
        self.dim_pre_MLP = dim_pre_MLP
        self.dim_post_MLP = dim_post_MLP
        self.dim_graphLin = dim_graphLin
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers
        self.num_graph_layers = num_graph_layers
        self.dropout_prob = dropout_prob
        self.training = training

        super(GraphConvolutionalNetwork, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        ### Each MLP module has two linear layers, each followed by activation functions of ReLU
        self.MLP = nn.ModuleList()
        for i in range(self.num_pre_layers):
            if i==0:
                mlp_layer = Sequential(Linear(num_node_features, self.dim_pre_MLP), ReLU()) 
                self.MLP.append(mlp_layer)
            else:
                mlp_layer = Sequential(Linear(self.dim_pre_MLP , self.dim_pre_MLP), ReLU()) 
                self.MLP.append(mlp_layer)
                    
        ### The MLP layers is followed by graph convolutional layers
        self.graphLayers = nn.ModuleList()
        for i in range(self.num_graph_layers):
            if i==0:
                gconv_layer = GCNConv(self.dim_pre_MLP, self.dim_graphLin)
                self.graphLayers.append(gconv_layer)
            else:
                gconv_layer = GCNConv(self.dim_graphLin, self.dim_graphLin)
                self.graphLayers.append(gconv_layer)
        
        ### The graph convolutional layers are followed by post processing layers
        ### Each MLP module has two linear layers, each followed by activation functions of ReLU
        self.postGCLayers = nn.ModuleList()
        for i in range(self.num_post_layers):
            if i==0 and i!=self.num_post_layers-1:
                if self.training:
                    mlp_layer = Sequential(Linear(self.dim_graphLin, self.dim_post_MLP), Dropout(self.dropout_prob), ReLU())
                else:
                    mlp_layer = Sequential(Linear(self.dim_graphLin, self.dim_post_MLP), ReLU())
                self.postGCLayers.append(mlp_layer)

            elif i==0 and i==self.num_post_layers-1:
                mlp_layer = Linear(self.dim_graphLin, 1)
                self.postGCLayers.append(mlp_layer)

            elif i>0 and i<self.num_post_layers-1:
                if self.training:
                    mlp_layer = Sequential(Linear(self.dim_post_MLP, self.dim_post_MLP), Dropout(self.dropout_prob), ReLU())
                else:
                    mlp_layer = Sequential(Linear(self.dim_post_MLP, self.dim_post_MLP), ReLU())
                self.postGCLayers.append(mlp_layer)
                
            else:
                mlp_layer = Linear(self.dim_post_MLP, 1)
                self.postGCLayers.append(mlp_layer)

    def forward(self, x, edge_indices, edge_weights, batch):
        ## Pre-processing layers (MLP modules)
        # print(batch)
        # print(f"The number of unique batches in the batch tensor is {len(torch.unique(batch))}")
        for i in range(self.num_pre_layers):
            x = self.MLP[i](x)

        # print(f"The shape of the input at the end of the pre-processing MLP layers is {x.shape}")
        # print(f"The shape of the the weights fed to the model is {edge_weights.shape}")
        ## Node embeddings
        for k in range(self.num_graph_layers):
            x = self.graphLayers[k](x, edge_indices, edge_weights)
            x = self.relu(x)
            # print(f"The shape of the embedding at the end of the graph layer {k} is {x.shape}")

        ## Graph-level pooling
        x = global_mean_pool(x, batch)
        # print(f"The shape of the embedding after graph level global pooling is {h[0].shape}")

        ### Post-processing and classification
        if self.num_post_layers == 1:
            x = self.sigmoid(self.postGCLayers[i](x))
        else:
            for i in range(self.num_post_layers):
                if i < self.num_post_layers-1:
                    x = self.postGCLayers[i](x)
                    # print(f"The shape of the input at the end of the post-processing MLP layer {i} is {x.shape}")
                else:
                    x = self.sigmoid(self.postGCLayers[i](x))
                    # print(f"The shape of the input at the end of the post-processing MLP layer {i} is {x.shape}")
        # print(f"The shape of the input at the end of the post-processing MLP layer {i} is {h_concat.shape}")

        # print(f"The shape of the input at the end of the post-processing MLP layers is {h_concat.shape}")
        # print(f"The output is:{h_concat}")

        return x

class GraphConvolutionNetworkWithSkipConnections(torch.nn.Module):
    """
    Creates a gnn model based on global pooling of embeddings - for graph level classifications
    """
    def __init__(self, dim_pre_MLP, dim_post_MLP, dim_graphLin, num_pre_layers, num_post_layers, dropout_prob, num_graph_layers, training=True):
        
        self.dim_pre_MLP = dim_pre_MLP
        self.dim_post_MLP = dim_post_MLP
        self.dim_graphLin = dim_graphLin 
        self.num_pre_layers = num_pre_layers  ## Now, it gives the number of dense layers
        self.num_post_layers = num_post_layers  ## Now, it gives the number of dense layers
        self.num_graph_layers = num_graph_layers
        self.dropout_prob = dropout_prob
        self.training = training

        super(GraphConvolutionNetworkWithSkipConnections, self).__init__()
        self.MLP = nn.ModuleList()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for i in range(self.num_pre_layers):
            if i==0:
                linear_layer = Linear(num_node_features, self.dim_pre_MLP)
                self.MLP.append(linear_layer)
            else:
                linear_layer = Linear(self.dim_pre_MLP + num_node_features, self.dim_pre_MLP)
                self.MLP.append(linear_layer)
                    
        ### The MLP layers is followed by graph convolutional layers
        self.graphLayers = nn.ModuleList()
        for i in range(self.num_graph_layers):
            if i==0:
                gconv_layer = GCNConv(self.dim_pre_MLP, self.dim_graphLin)
                self.graphLayers.append(gconv_layer)
            else:
                gconv_layer = GCNConv(self.dim_graphLin, self.dim_graphLin)
                self.graphLayers.append(gconv_layer)
        
        
        ### The graph convolutional layers are followed by post processing layers
        ### Each MLP module has two linear layers, each followed by activation functions of ReLU
        self.postGCLayers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(self.num_post_layers):
            if i==0 and i!=self.num_post_layers-1:
                linear_layer = Linear(self.dim_graphLin, self.dim_post_MLP)
                self.postGCLayers.append(linear_layer)
                if self.training:
                    self.dropout_layers.append(Dropout(self.dropout_prob))
            
            elif i==0 and i==self.num_post_layers-1:
                linear_layer = Linear(self.dim_graphLin, 1)
                self.postGCLayers.append(linear_layer)

            elif i>0 and i<self.num_post_layers-1:
                linear_layer = Linear(self.dim_post_MLP + self.dim_post_MLP, self.dim_post_MLP)
                self.postGCLayers.append(linear_layer)
                if self.training:
                    self.dropout_layers.append(Dropout(self.dropout_prob))
                
            else:
                linear_layer = Linear(self.dim_post_MLP + self.dim_post_MLP, 1)
                self.postGCLayers.append(linear_layer)

    def forward(self, x, edge_indices, edge_weights, batch):
        ## Pre-processing layers (MLP modules)
        # print(batch)
        # print(f"The number of unique batches in the batch tensor is {len(torch.unique(batch))}")
        if self.num_pre_layers == 1:
            x = self.relu(self.MLP[i](x))
        else:
            for i in range(self.num_pre_layers):
                if i < self.num_pre_layers-1:
                    # print(f"The shape of the input at the start of the pre-processing layer{i} is {x.shape}")
                    x_out = self.relu(self.MLP[i](x))
                    # print(f"The shape of the input at the end of the pre-processing layer{i} is {x_out.shape}")
                    x = torch.cat([x_out,x], dim=-1)
                    # print(f"The shape of the input after concatenation at the end of the pre-processing layer{i} is {x.shape}")
                else:
                    x = self.relu(self.MLP[i](x))
                    # print(f"The shape of the input at the end of the pre-processing layer{i} is {x.shape}")

        # print(f"The shape of the input at the end of the pre-processing MLP layers is {x.shape}")
        # print(f"The shape of the the weights fed to the model is {edge_weights.shape}")
        ## Node embeddings
        for k in range(self.num_graph_layers):
            x = self.graphLayers[k](x, edge_indices, edge_weights)
            # print(f"The shape of the embedding at the end of the graph layer {k} is {x.shape}")
            x = self.relu(x)

        ## Graph-level pooling
        x = global_mean_pool(x, batch)

        ### Post-processing and classification
        if self.num_post_layers == 1:
            x = self.sigmoid(self.postGCLayers[i](x))
        else:
            for i in range(self.num_post_layers):
                if i < self.num_post_layers-1:
                    x = self.postGCLayers[i](x)
                    if self.training:
                        x = self.dropout_layers[i](x)
                    x_out = self.relu(x)
                    x = torch.cat([x_out,x], dim=-1)

                else:
                    x = self.sigmoid(self.postGCLayers[i](x))
            # print(f"The shape of the input at the end of the post-processing MLP layer {i} is {h_concat.shape}")

        # print(f"The shape of the input at the end of the post-processing MLP layers is {h_concat.shape}")
        # print(f"The output is:{h_concat}")

        return x

### Hyperparameters to be tested
training = True
dim_pre_MLP = 32
dim_post_MLP = 32
dim_graphLin = 32
num_pre_layers = 2
num_post_layers = 2
dropout_prob = 0.1
num_graph_layers = 3
patience_new = 15
learning_rate = 5e-4

dict = {"dim_pre_MLP": dim_pre_MLP,
        "dim_post_MLP": dim_post_MLP,
        "dim_graphLin": dim_graphLin,
        "num_pre_layers": num_pre_layers,
        "num_post_layers": num_post_layers,
        "dropout_prob": dropout_prob,
        "num_graph_layers": num_graph_layers,
        "patience": patience_new,
        "batch size": batch_size,
        "num neighbours": num_neighbours,
        "learning_rate": learning_rate}

for key, val in dict.items():
    print(f"{key} : {val}")

file_str = f"GraphClassification_PCA_GCN_pre_MLP_dim_{dim_pre_MLP}"
curves_filename = os.getcwd()+'/training_history_'+'/'+ file_str +'.jpg'
model_filename = os.getcwd()+'/model_saves_'+'/'+ file_str +'.pt'
accuracy_filename = os.getcwd()+'/accuracy_saves_'+'/'+ file_str+'.txt'
time_filename = os.getcwd()+'/time_saves_'+'/'+ file_str+'.txt'

def compute_accuracy(preds, labels):
    # print(f"Output:{preds}")
    # print(f"Labels:{labels}")
    predicted_classes = (preds > 0.5).float()
    # print(f"Predicted classes:{predicted_classes}")
    num_correct = (predicted_classes == labels).float().sum()
    # print(f"Number of correctly predicted results: {num_correct}")
    # print(f"Total number of items: {labels.shape[0]}")
    accuracy = num_correct / labels.shape[0]
    return accuracy.item()

from torch.optim.lr_scheduler import ReduceLROnPlateau

def trainVal(model, dataloader):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= 0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    num_epochs = 200

    ##Early stopping parameters
    best_val_loss = float('inf')
    patience = patience_new
    counter = 0
    best_model_path = model_filename

    ## Keep track of the losses and accuracies over epochs
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    
    for epoch in range(num_epochs+1):
        
        model.train() ## Training mode
        
        train_acc = 0
        train_loss = 0

        for batch in dataloader:
            edge_indices = batch.edge_index.to(torch.int64)
            node_features = (torch.tensor(batch.x)).reshape(len(batch.x) * len(batch.x[0]),1).float() 
            edge_weights = (torch.tensor(batch.edge_weight)).reshape(len(batch.x) * len(batch.edge_weight[0])).float()
            labels = batch.y.reshape(len(batch.x) ,1).float()
            batches = batch.batch
            num_unique_batches = len(torch.unique(batches))
            batches %= num_unique_batches

            ## Forward pass
            optimizer.zero_grad()
            output = model(node_features, edge_indices, edge_weights, batches)
            # print(f"Output shape is:{output.shape}")
            loss = criterion(output, labels)

            ## Backward propagation and optimization 
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()

            ## Keep track of the losses and accuracies
            train_loss += loss.item()
            train_acc += compute_accuracy(output, labels)

        # print(f"Dataloader length:{len(dataloader)}")
        # print(f"Train loss: {train_loss}")
        # print(f"Train accuracy: {train_acc}")
        
        epoch_loss = train_loss / len(dataloader)
        epoch_accuracy = train_acc / len(dataloader)

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        ## validation
        val_loss = 0.0
        val_acc = 0.0
        
        model.eval()  ## evaluation state
        with torch.no_grad():
            for batch in val_dataloader:
                edge_indices = batch.edge_index.to(torch.int64)
                node_features = (torch.tensor(batch.x)).reshape(len(batch.x) * len(batch.x[0]),1).float() 
                edge_weights = (torch.tensor(batch.edge_weight)).reshape(len(batch.x) * len(batch.edge_weight[0])).float()
                labels = batch.y.reshape(len(batch.x) ,1).float()
                batches = batch.batch
                num_unique_batches = len(torch.unique(batches))
                batches %= num_unique_batches

                # edge_indices = data[0]
                # node_features = data[1]
                # edge_weights = data[2]
                # labels = data[3]
                # batches = data[4]
        
                output = model(node_features, edge_indices, edge_weights, batches)
                val_loss += criterion(output, labels).item()
                val_acc += compute_accuracy(output, labels)

        validation_loss = val_loss/len(val_dataloader)
        validation_acc = val_acc/len(val_dataloader)

        scheduler.step(val_loss)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_acc)

        ## print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {epoch_loss:.2f}'
              f'| Train Acc: {epoch_accuracy*100:4.3f}% '
              f'| Val Loss: {validation_loss:.2f} '
              f'| Val Acc: {validation_acc*100: 4.3f}%')

        ### Check for the validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f'Saving the model with val loss: {val_loss:.4f}')
        else:
            counter +=1

        ### Early stopping condition
        if counter >= patience:
            print("Early stopping triggered.")
            break

    ## Put the losses and accuracies in a dictionary which can be returned
    losses = {"Train loss": training_losses,
              "Train accuracy": training_accuracies,
              "Validation loss": validation_losses,
              "Validation accuracy": validation_accuracies}

    return losses

@torch.no_grad()
def test(model):
    criterion = torch.nn.BCELoss()
    model.eval()
    test_loss = 0
    test_acc = 0

    for batch in test_dataloader:
        edge_indices = batch.edge_index.to(torch.int64)
        node_features = (torch.tensor(batch.x)).reshape(len(batch.x) * len(batch.x[0]),1).float() 
        edge_weights = (torch.tensor(batch.edge_weight)).reshape(len(batch.x) * len(batch.edge_weight[0])).float()
        labels = batch.y.reshape(len(batch.x) ,1).float()
        batches = batch.batch
        num_unique_batches = len(torch.unique(batches))
        batches %= num_unique_batches
        
        output = model(node_features, edge_indices, edge_weights, batches)
        test_loss += criterion(output, labels).item()
        test_acc += compute_accuracy(output, labels)

    testing_loss = test_loss / len(test_dataloader)
    testing_acc = test_acc / len(test_dataloader)

    return testing_loss, testing_acc

# start_time = time.time()

# ### Training and evaluation of the model
# model = GraphConvolutionalNetwork(dim_pre_MLP, dim_post_MLP, dim_graphLin, num_pre_layers, num_post_layers, dropout_prob, num_graph_layers, training)
# train_val_metrics = trainVal(model, train_dataloader)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time is {elapsed_time / 60:3.2f} minutes.") 

start_time = time.time()

### Training and evaluation of the model
model = GraphConvolutionNetworkWithSkipConnections(dim_pre_MLP, dim_post_MLP, dim_graphLin, num_pre_layers, num_post_layers, dropout_prob, num_graph_layers, training)
train_val_metrics = trainVal(model, train_dataloader)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time is {elapsed_time / 60:3.2f} minutes.") 

elapsed_time = np.array([round(elapsed_time, 4)])
np.savetxt(time_filename, elapsed_time, fmt='%.4f')

plt.rcParams["font.family"]="serif"
epochs = np.arange(0, len(train_val_metrics["Train loss"]))

fig,ax = plt.subplots(1,2, figsize=[8,3], dpi=300)
ax[0].plot(epochs, train_val_metrics["Train loss"], 'red', linestyle = '-', linewidth=1.0, marker = 's', mfc = 'k', markersize = 3, label = 'Training loss')
ax[0].plot(epochs, train_val_metrics["Validation loss"], 'blue', linestyle = '-', linewidth=1.0, marker = 's', mfc = 'k', markersize = 3, label = 'Validation loss')
ax[0].set_title('Loss curves')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Losses")
ax[0].legend()

ax[1].plot(epochs, train_val_metrics["Train accuracy"], 'red', linestyle = '-', linewidth=1.0, marker = 's', mfc = 'k', markersize = 3, label = 'Training accuracy')
ax[1].plot(epochs, train_val_metrics["Validation accuracy"], 'blue', linestyle = '-', linewidth=1.0, marker = 's', mfc = 'k', markersize = 3, label = 'Validation accuracy')
ax[1].set_title('Accuracy curves')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.tight_layout()
plt.savefig(curves_filename,dpi=199)
plt.close()

model = GraphConvolutionNetworkWithSkipConnections(dim_pre_MLP, dim_post_MLP, dim_graphLin, num_pre_layers, num_post_layers, dropout_prob, num_graph_layers, training=False)
model.load_state_dict(torch.load(model_filename))
model.eval()

test_metrics = test(model)
formatted_accuracy = "{:.4f}".format(test_metrics[1])
formatted_loss = "{:.4f}".format(test_metrics[0])

print(f"Test loss: {formatted_loss}")
print(f"Test accuracy: {formatted_accuracy}")

test_loss = test_metrics[0]
test_accuracy = np.array([round(test_metrics[1], 4)])

np.savetxt(accuracy_filename, test_accuracy, fmt='%.4f')