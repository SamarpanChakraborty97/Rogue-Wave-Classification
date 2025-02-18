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
from torch.nn import Linear, Sequential, BatchNorm1d, Dropout, ReLU, Sigmoid, ELU
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, scatter
import time
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from torch_geometric.utils import degree

def set_seed(seed):
    random.seed(seed)  # Seed for the random module
    np.random.seed(seed)  # Seed for NumPy
    torch.manual_seed(seed)  # Seed for PyTorch
    torch.cuda.manual_seed(seed)  # Seed for current GPU
    torch.cuda.manual_seed_all(seed)  # Seed for all GPUs (if you have more than one)

# Set the desired seed
set_seed(42)

file_str="RWs_H_g_2_tadv_5min_localized_area"
file_str_test = "RWs_H_g_2_tadv_5min_localized_area_test"

data = np.load(file_str+".npz")
data_test = np.load(file_str_test+".npz")

num_examples = 2000

wave_data_train=data["wave_data_train"][:num_examples]
wave_data_test=data_test["wave_data_test"]
label_train=data["label_train"][:num_examples]
label_test=data_test["label_test"]
buoy_mask_train=data["buoy_mask_train"][:num_examples]
buoy_mask_test=data_test["buoy_mask_test"]

num_classes=2

num_nodes = wave_data_train.shape[0] +  wave_data_test.shape[0]
num_node_features = wave_data_train.shape[1]

print("----------------The graph structure will be in the form of a single graph------------------".upper())
print("--The graph will have multiple nodes where each node is a time series sample--".upper())
print("----------------------------The feature length of each node is the length of the time series----------------------------".upper())
print('\n')
print('-----------------------The graph properties are given below---------------------\n'.upper())
print(f"The number of nodes in the graph is {num_nodes}")
print(f"The number of features for each node is {num_node_features}")
print(f"The number of training set examples is {wave_data_train.shape[0]}")
print(f"The number of test set examples is {wave_data_test.shape[0]}")

wave_data_train_df= pd.DataFrame(data=wave_data_train.reshape(wave_data_train.shape[0], wave_data_train.shape[1]))
wave_data_test_df= pd.DataFrame(data=wave_data_test.reshape(wave_data_test.shape[0], wave_data_test.shape[1]))
combined_wave_data_nodes = pd.concat([wave_data_train_df, wave_data_test_df], axis=0)

labels = np.concatenate((label_train, label_test), axis=0)
buoy_masks = np.concatenate((buoy_mask_train, buoy_mask_test), axis=0)

train_masks = np.zeros(len(labels))
for i in range(len(train_masks)):
    if i<len(label_train):
        train_masks[i] = 1
    else:
        train_masks[i] = 0
combined_wave_data_nodes['labels'] = labels
combined_wave_data_nodes['buoy_masks'] = buoy_masks
combined_wave_data_nodes['train_masks'] = train_masks

combined_wave_info = np.array(combined_wave_data_nodes)
permuted_indices = np.random.permutation(combined_wave_info.shape[0])
combined_wave_info = combined_wave_info[permuted_indices]

distances = [[0.00, 27.86,43.21, 84.34, 95.40],
                 [-27.86, 0.00, 16.92, 68.93, 68.23],
                 [-43.21, 16.92, 0.00, 55.76, 52.20],
                 [-84.34, -68.93, -55.76, 0.00, 58.41],
                 [-95.40, -68.23, -52.20, -58.41, 0.00]]

max_dist = max(max(distances))
norm_dist = []
for i in range(len(distances)):
    buoy_list = []
    for j in range(len(distances[0])):
        if distances[i][j] >= 0:
            buoy_list.append(1 - (distances[i][j] / max_dist))
        else:
            buoy_list.append(0)
    norm_dist.append(buoy_list)

num_samples = combined_wave_info.shape[0]

edge_file_path = f"edge_data_local_original_node_classification_{num_examples}.npz"

if os.path.exists(edge_file_path):
    edge_data=np.load(edge_file_path)
    edge_indices_start=edge_data["edge_start"]
    edge_indices_end=edge_data["edge_end"]
    edge_weights = edge_data["edge_weights"]

else:
    ### For the creation of edges between the different nodes, we are looking at the correlation between the different time series samples
    ### The edges are created if the correlation exceeds a certain threshold
    
    edge_indices_start = []
    edge_indices_end = []
    edge_weights_list = []
    for i in range(num_samples):
        buoy_i = combined_wave_info[i,-2].astype(int)
        num_immediate_neighbours = 0
        num_other_neighbours = 0
        skip_inner_loop = False
        for j in range(num_samples):
            buoy_j = combined_wave_info[j,-2].astype(int)
            if norm_dist[buoy_i][buoy_j] > 0:
                if buoy_i == buoy_j:
                    num_immediate_neighbours +=1
                    edge_indices_start.append(i)
                    edge_indices_end.append(j)
                    edge_weights_list.append(norm_dist[buoy_i][buoy_j])
                    if num_immediate_neighbours >= 400:
                        skip_inner_loop = True
                        break
                else:
                    num_other_neighbours +=1
                    edge_indices_start.append(i)
                    edge_indices_end.append(j)
                    edge_weights_list.append(norm_dist[buoy_i][buoy_j])
                    if num_other_neighbours >= 500:
                        skip_inner_loop = True
                        break
            if skip_inner_loop:
                continue
                        
    np.savez(edge_file_path,edge_start=edge_indices_start, edge_end=edge_indices_end, edge_weights=edge_weights_list)

    edge_data=np.load(edge_file_path)
    edge_indices_start=edge_data["edge_start"]
    edge_indices_end=edge_data["edge_end"]
    edge_weights = edge_data["edge_weights"]

edge_indices = [edge_indices_start, edge_indices_end]

x = torch.tensor(combined_wave_info[:,:-3], dtype=torch.float)
y = torch.tensor(combined_wave_info[:,-3], dtype=torch.float)
edge_index = torch.tensor(edge_indices, dtype = torch.long)
edge_weight = torch.tensor(edge_weights, dtype=torch.float)
                          
graph_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_val_indices = np.where(combined_wave_info[:,-1] == 1.0)[0]
train_val_ratio = 0.7
train_indices = train_val_indices[:int(train_val_ratio * len(train_val_indices))]
val_indices = train_val_indices[int(train_val_ratio * len(train_val_indices)):]
test_indices = np.where(combined_wave_info[:,-1] == 0.0)[0]

train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

graph_data.train_mask = train_mask
graph_data.val_mask = val_mask
graph_data.test_mask = test_mask

print(f'\nGraph:'.upper())
print('------')
print(f"Total number of edges: {len(graph_data.edge_index[0])}".upper())
print(f'Training nodes: {sum(graph_data.train_mask).item()}'.upper())
print(f'Evaluation nodes: {sum(graph_data.val_mask).item()}'.upper())
print(f'Test nodes: {sum(graph_data.test_mask).item()}'.upper())

class LSTM_Graph(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_lstm_layers, dropout):
        super(LSTM_Graph, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        self.lstm_layers = nn.ModuleList()
        self.layerNorm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        for i in range(self.num_lstm_layers):
            if i==0:
                lstm_layer = nn.LSTM(self.input_dim, self.lstm_hidden_dim, batch_first=True)
                self.lstm_layers.append(lstm_layer)
                self.layerNorm_layers.append(nn.LayerNorm(self.lstm_hidden_dim))
            else:
                lstm_layer = nn.LSTM(self.lstm_hidden_dim, self.lstm_hidden_dim, batch_first=True)
                self.lstm_layers.append(lstm_layer)
                self.layerNorm_layers.append(nn.LayerNorm(self.lstm_hidden_dim))

    def forward(self, x):
        ## Initialize the hidden and cell states for each LSTM layer
        h_n, c_n = [None] * self.num_lstm_layers, [None] * self.num_lstm_layers

        for i in range(self.num_lstm_layers):
            x, (h_n[i], _) = self.lstm_layers[i](x)
            
            # Apply layer normalization to the output of the current LSTM layer
            x = self.layerNorm_layers[i](x)
            
            # Apply dropout if not the last layer
            if i < self.num_lstm_layers - 1:
                x = self.dropout(x)

        return h_n[-1]

class GCN_Graph(nn.Module):
    def __init__(self, num_gcn_layers, gcn_idim, gcn_hdim, dropout):
        super(GCN_Graph, self).__init__()

        self.num_gcn_layers = num_gcn_layers
        self.gcn_idim = gcn_idim
        self.gcn_hdim = gcn_hdim
        
        self.dropout = Dropout(dropout)
        self.elu = ELU()

        self.gcn_layers = nn.ModuleList()

        for i in range(self.num_gcn_layers):
            if i==0:
                gcn_layer = GCNConv(self.gcn_idim, self.gcn_hdim)
                self.gcn_layers.append(gcn_layer)
            else:
                gcn_layer = GCNConv(self.gcn_hdim, self.gcn_hdim)
                self.gcn_layers.append(gcn_layer)

    def forward(self, x, edge_index, edge_weight):

        for k in range(self.num_gcn_layers):
            gcn_out = self.elu(self.gcn_layers[k](x, edge_index, edge_weight))
            if k < self.num_gcn_layers-1:
                gcn_out = self.dropout(gcn_out)

        return gcn_out

import torch.nn.functional as F
class SpatioTemporalGCN(torch.nn.Module):
    """
    Creates a gnn model based on global pooling of embeddings - for graph level classifications
    """
    def __init__(self, num_gcn_layers, gcn_hdim, dropout_prob, num_lstm_layers, lstm_hdim, training=True):
        
        self.num_gcn_layers = num_gcn_layers
        self.gcn_hdim = gcn_hdim
        self.dropout_prob = dropout_prob
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hdim = lstm_hdim
        self.training = training

        super(SpatioTemporalGCN, self).__init__()

        self.temporal_module = LSTM_Graph(1, self.lstm_hdim, self.num_lstm_layers, self.dropout_prob)
        self.spatial_module = GCN_Graph(self.num_gcn_layers, self.lstm_hdim, self.gcn_hdim, self.dropout_prob)
        self.fc = Linear(self.lstm_hdim+1, 2)

    def forward(self, x, edge_index, edge_weight):
        # print(f"The shape of the input at the start of the spatio-temporal module is {x.shape}")
        x_lstm = x.unsqueeze(2)
        # print(f"The shape of the input at the start of the temporal module is {x_lstm.shape}")
        
        batch_size, seqeunce_len, _ = x_lstm.size()
        # lstm_outputs = []
        # for i in range(batch_size):
        #     lstm_outputs.append(self.temporal_module(x_lstm[i]))
        # print(lstm_outputs[0].shape)
        # print(len(lstm_outputs))
        # lstm_out = torch.stack(lstm_outputs, dim=0)
        
        
        lstm_out = torch.stack([self.temporal_module(x_lstm[i]) for i in range(batch_size)], dim=0)
        # print(f"The shape of the input at the end of the temporal module is {lstm_out.shape}")

        gcn_in = lstm_out.view(-1, lstm_out.shape[2])
        # print(f"The shape of the input at the start of the spatial module is {gcn_in.shape}")

        gcn_out = self.spatial_module(gcn_in, edge_index, edge_weight)
        # print(f"The shape of the input at the end of the spatial module is {gcn_out.shape}")

        concatenated_features = torch.cat([gcn_in, torch.mean(gcn_out, dim=1, keepdim=True)], dim=1)
        # concatenated_features = torch.cat([gcn_in, gcn_out], dim=1)
        # print(f"The shape of the concatenated features is {concatenated_features.shape}")

        out = self.fc(concatenated_features)
        # print(f"The shape of the output is {out.shape}")

        return F.log_softmax(out, dim=1)

num_gcn_layers = 2
gcn_hdim = 32
dropout_prob = 0.1
num_lstm_layers = 2
lstm_hdim = 32

patience_new = 10

file_str = f"NodeClassification_Local_Originallength_GCN_num_gcn_layers_{num_gcn_layers}"
curves_filename = os.getcwd()+'/training_history_'+'/'+ file_str +'.jpg'
model_filename = os.getcwd()+'/model_saves_'+'/'+ file_str +'.pt'
accuracy_filename = os.getcwd()+'/accuracy_saves_'+'/'+ file_str+'.txt'
time_filename = os.getcwd()+'/time_saves_'+'/'+ file_str+'.txt'

def compute_accuracy(preds, labels):
    # print(f"Output:{preds}")
    # print(f"Labels:{labels}")
    _, pred_classes = torch.max(preds, dim=1)
    # print(f"Predicted classes:{pred_classes}")
    num_correct = (pred_classes == labels).float().sum()
    # print(f"Number of correctly predicted results: {num_correct}")
    # print(f"Total number of items: {labels.shape[0]}")
    accuracy = num_correct / labels.shape[0]
    # print(accuracy)
    return accuracy.item()

from torch.optim.lr_scheduler import ReduceLROnPlateau

def trainVal(model):
    criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay= 5e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    num_epochs = 70

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
        optimizer.zero_grad()
        output = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        # print(f"Output shape is:{output.shape}")
        # loss = criterion(output[graph_data.train_mask], graph_data.y[graph_data.train_mask].reshape(-1,1))
        loss = criterion(output[graph_data.train_mask], graph_data.y[graph_data.train_mask].long())

        ## Backward propagation and optimization 
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optimizer.step()

        ## Keep track of the losses and accuracies
        train_loss = loss.item()
        # train_acc = compute_accuracy(output[graph_data.train_mask], graph_data.y[graph_data.train_mask].reshape(-1,1))
        train_acc = compute_accuracy(output[graph_data.train_mask], graph_data.y[graph_data.train_mask].long())

        ## Keep track of the validation losses and accuracies
        # val_loss = criterion(output[graph_data.val_mask], graph_data.y[graph_data.val_mask].reshape(-1,1)).item()
        val_loss = criterion(output[graph_data.val_mask], graph_data.y[graph_data.val_mask].long()).item()
        # val_acc = compute_accuracy(output[graph_data.val_mask], graph_data.y[graph_data.val_mask].reshape(-1,1))
        val_acc = compute_accuracy(output[graph_data.val_mask], graph_data.y[graph_data.val_mask].long())

        # scheduler.step(val_loss)

        training_losses.append(train_loss)
        training_accuracies.append(train_acc)

        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        ## print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.2f}'
              f'| Train Acc: {train_acc*100:4.3f}% '
              f'| Val Loss: {val_loss:.2f} '
              f'| Val Acc: {val_acc*100: 4.3f}%')

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
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
  
    output = model(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
    # test_loss = criterion(output[graph_data.test_mask], graph_data.y[graph_data.test_mask].reshape(-1,1)).item()
    test_loss = criterion(output[graph_data.test_mask], graph_data.y[graph_data.test_mask].long()).item()
    # test_acc = compute_accuracy(output[graph_data.test_mask], graph_data.y[graph_data.test_mask].reshape(-1,1))
    test_acc = compute_accuracy(output[graph_data.test_mask], graph_data.y[graph_data.test_mask].long())

    testing_loss = test_loss
    testing_acc = test_acc

    return testing_loss, testing_acc

start_time = time.time()

training=True
### Training and evaluation of the model
model = SpatioTemporalGCN(num_gcn_layers, gcn_hdim, dropout_prob, num_lstm_layers, lstm_hdim, training=True)
train_val_metrics = trainVal(model)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time is {elapsed_time / 60:3.2f} minutes.") 

elapsed_time = np.array([round(elapsed_time, 4)])
np.savetxt(time_filename, elapsed_time, fmt='%.4f')

plt.rcParams["font.family"]="serif"
epochs = np.arange(0, len(train_val_metrics["Train loss"]))

fig,ax = plt.subplots(1,2, figsize=[8,3], dpi=300)
ax[0].plot(epochs, train_val_metrics["Train loss"], 'red', linestyle = '-', linewidth=0.25, marker = 's', mfc = 'k', markersize = 0.5, label = 'Training loss')
ax[0].plot(epochs, train_val_metrics["Validation loss"], 'blue', linestyle = '-', linewidth=0.25, marker = 's', mfc = 'k', markersize = 0.5, label = 'Validation loss')
ax[0].set_title('Loss curves')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Losses")
ax[0].legend()

ax[1].plot(epochs, train_val_metrics["Train accuracy"], 'red', linestyle = '-', linewidth=0.25, marker = 's', mfc = 'k', markersize = 0.5, label = 'Training accuracy')
ax[1].plot(epochs, train_val_metrics["Validation accuracy"], 'blue', linestyle = '-', linewidth=0.25, marker = 's', mfc = 'k', markersize = 0.5, label = 'Validation accuracy')
ax[1].set_title('Accuracy curves')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.tight_layout()
plt.savefig(curves_filename,dpi=199)
plt.close()

model = SpatioTemporalGCN(num_gcn_layers, gcn_hdim, dropout_prob, num_lstm_layers, lstm_hdim, training=False)
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
