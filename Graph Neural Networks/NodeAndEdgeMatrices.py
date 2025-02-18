import numpy as np
import scipy.sparse as sp
import os
import tensorflow as tf

wg = 4
window = 15
file_str=f"tadv_5min_wave_group_window_{window}mins_{wg}"

save_name_edges = os.getcwd() + f"/edge_data_{window}_{wg}_original"

data=np.load(file_str+".npz")
wave_data_train=data["wave_data_train"]
wave_data_test=data["wave_data_test"]
label_train=data["label_train"]
label_test=data["label_test"]
num_classes=2

print(wave_data_train.shape)
print(wave_data_test.shape)

print(label_train.shape)
print(label_test.shape)

learning_rate = 1e-3  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 10  # Patience for early stopping
batch_size = 32  # Batch size

from scipy.stats import skew, kurtosis

sequence_length = wave_data_train.shape[1]
obs_per_sec = 1.28
ten_sec_windows_obs = round(10*obs_per_sec)

num_nodes = wave_data_train.shape[0]

def create_adjacency_matrix(num_nodes, directionality=True):
    D_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            D_matrix[i,j] = abs(i-j)

    k = 70
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

A = create_adjacency_matrix(num_nodes, directionality=False)

sparse_matrix = sp.coo_matrix(A)
indices = np.column_stack((sparse_matrix.nonzero()))
values = sparse_matrix.data
dense_shape = sparse_matrix.shape
sparse_A = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

edge_features_train = np.zeros((wave_data_train.shape[0], len(indices),1))
edge_features_test = np.zeros((wave_data_test.shape[0], len(indices),1))

for i in range(edge_features_train.shape[0]):
    for j in range(len(indices)):
        index = indices[j]
        arr1 = wave_data_train[i,index[0]*sliding_window_length:(index[0]+1)*sliding_window_length,:].reshape(-1)
        arr2 = wave_data_train[i,index[1]*sliding_window_length:(index[1]+1)*sliding_window_length,:].reshape(-1)
        edge_features_train[i,j,:] = np.corrcoef(arr1, arr2)[0,1]

for i in range(edge_features_test.shape[0]):
    for j in range(len(indices)):
        index = indices[j]
        arr1 = wave_data_test[i,index[0]*sliding_window_length:(index[0]+1)*sliding_window_length,:].reshape(-1)
        arr2 = wave_data_test[i,index[1]*sliding_window_length:(index[1]+1)*sliding_window_length,:].reshape(-1)
        edge_features_test[i,j,:] = np.corrcoef(arr1, arr2)[0,1]

np.savez(save_name_edges,edge_data_train=edge_features_train, edge_data_test=edge_features_test)

file_str=save_name_edges
data=np.load(file_str+".npz")
edge_data_train=data["edge_data_train"]
edge_data_test=data["edge_data_test"]

print(edge_data_train.shape)
print(edge_data_test.shape)