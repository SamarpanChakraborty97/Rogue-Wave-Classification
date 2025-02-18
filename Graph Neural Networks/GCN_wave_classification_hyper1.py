#!/usr/bin/env python
# coding: utf-8

# ### Importing required modules - using spektral library for the GNN

# In[1]:


import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, sparse_categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW

from spektral.data import Dataset, DisjointLoader, Graph, MixedLoader
from spektral.layers import GCSConv, GlobalAvgPool, GCNConv, GlobalSumPool
from spektral.transforms.normalize_adj import NormalizeAdj
import os
import networkx as nx


# ### The required wave data

# In[2]:


file_str="tadv_5min_wave_group_4"
GCN_save_name= os.getcwd() + "/best_GCN_different_group_4_"+file_str +".h5"

wg = 4
window = 15

save_name= os.getcwd() + f"/transformed_data_{window}_{wg}"
save_name_edges = os.getcwd() + f"/edge_data_{window}_{wg}"

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


# ### Configuration for the model

# In[3]:


learning_rate = 1e-3  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 10  # Patience for early stopping
batch_size = 32  # Batch size


# In[4]:


# print(np.min(wave_data_train))
# print(np.max(wave_data_train))
# print(np.mean(wave_data_train))
# print(np.std(wave_data_train))

# print('\n')
# print(np.min(wave_data_test))
# print(np.max(wave_data_test))
# print(np.mean(wave_data_test))
# print(np.std(wave_data_test))


# #### Breaking down the time sequence into sliding windows. Calculating the node features and the edge features from this.

# In[5]:


sequence_length = wave_data_train.shape[1]
obs_per_sec = 1.28
ten_sec_windows_obs = round(4*obs_per_sec)

num_sliding_windows = int(sequence_length / ten_sec_windows_obs)
num_nodes = num_sliding_windows
sliding_window_length = round(wave_data_train.shape[1] / num_nodes)

transformed_wave_data_train = np.zeros((wave_data_train.shape[0], num_nodes,1))
transformed_wave_data_test = np.zeros((wave_data_test.shape[0], num_nodes,1))

for i in range(wave_data_train.shape[0]):
    for j in range(num_nodes):
        transformed_wave_data_train[i,j,:] = np.mean(wave_data_train[i,j*sliding_window_length:(j+1)*sliding_window_length,:])

for i in range(wave_data_test.shape[0]):
    for j in range(num_nodes):
        transformed_wave_data_test[i,j,:] = np.mean(wave_data_test[i,j*sliding_window_length:(j+1)*sliding_window_length,:])

mean_train = np.mean(transformed_wave_data_train)
std_train = np.std(transformed_wave_data_train)
transformed_wave_data_train = (transformed_wave_data_train) / (4*std_train)

mean_test = np.mean(transformed_wave_data_test)
std_test = np.std(transformed_wave_data_test)
transformed_wave_data_test = (transformed_wave_data_test) / (4*std_test)


# In[6]:


# mean_train = np.mean(transformed_wave_data_train)
# std_train = np.std(transformed_wave_data_train)
# print(f'std_train:{std_train}')
# print(f'mean_train:{mean_train}')
# print('\n')
# transformed_wave_data_train = (transformed_wave_data_train) / (4*std_train)
# print(f'min:{np.min(transformed_wave_data_train)}')
# print(f'max:{np.max(transformed_wave_data_train)}')
# print('\n')

# mean_test = np.mean(transformed_wave_data_test)
# std_test = np.std(transformed_wave_data_test)
# print(f'std_test:{std_test}')
# print(f'mean_test:{mean_test}')
# print('\n')
# transformed_wave_data_test = (transformed_wave_data_test) / (4*std_test)
# print(f'min:{np.min(transformed_wave_data_test)}')
# print(f'max:{np.max(transformed_wave_data_test)}')


# #### Investigating a random window 

# In[7]:


# ### Creating the train and test data graphs

# In[8]:


class TimeSeriesGraphDataset(Dataset):
    def __init__(self, X, Y, **kwargs):
        self.X = X
        self.num_samples = X.shape[0]
        self.num_nodes = X.shape[1]
        print(self.num_nodes)
        print(self.num_samples)
        self.Y = Y
        self.A = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.A.shape[0]):
                for j in range(i+1):
                    self.A[i,j] = (1/(i+1-j))
                    
        super().__init__(**kwargs)
        

    def read(self):
        graphs = []
        for k in range(self.num_samples):
            x = self.X[k,:]    
            a = self.A
            y = self.Y[k]

            graph = Graph(x=x, a=a, y=y)
            graphs.append(graph)
        
        return graphs

data_trainVal = TimeSeriesGraphDataset(transformed_wave_data_train, label_train, transforms=NormalizeAdj())
data_test = TimeSeriesGraphDataset(transformed_wave_data_test, label_test, transforms=NormalizeAdj())

# ### Creating the dataloaders

# In[ ]:


# Train/valid/test split
idxs = np.random.permutation(len(data_trainVal))
split_va = int(0.8 * len(data_trainVal))
idx_tr, idx_va = np.split(idxs, [split_va])

data_tr = data_trainVal[idx_tr]
data_va = data_trainVal[idx_va]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_test, batch_size=batch_size)


# In[ ]:


batch = loader_va.__next__()
inputs, target = batch
x, a, i = inputs
x.shape


# In[ ]:


num_nodes


# ### Defining the time series classification model

# In[ ]:


################################################################################
# Build model
################################################################################
class TimeSeriesGNN(Model):
    def __init__(self, n_hidden, n_preout, n_classes, **kwargs):     
        self.n_hidden = n_hidden
        self.n_preout = n_preout
        self.n_classes = n_classes

        self.conv1 = GCSConv(self.n_hidden, activation="relu")
        self.conv2 = GCSConv(self.n_hidden, activation="relu")
        self.conv3 = GCSConv(self.n_hidden, activation="relu")
        self.global_pool = GlobalSumPool()
        self.fc1 = Dense(self.n_preout)
        self.dense = Dense(self.n_classes, activation = 'sigmoid')
        super().__init__(**kwargs)

    def build(self, input_shape):
        # No need for complex handling here since the layers are created in __init__()
        # Just ensuring the model input shape is recognized
        super().build(input_shape)
        print(f"Model built with input shape: {input_shape}")

    def call(self, inputs):
        x, a, i = inputs
        i = tf.cast(i,tf.int32)
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.fc1(output)
        output = self.dense(output)
        return output

# Instantiate the model
n_hidden = 32
n_preout = 16
n_classes = 2

acc_save_name = os.getcwd() + "/accuracy_saves_/" + f"acc_gcs_batch_size_{batch_size}.txt"

model = TimeSeriesGNN(n_hidden, n_preout, n_classes)
# model.build(input_shape=(None, num_nodes, 1))
# model.build(input_shape=[(None, num_nodes * batch_size, 1), (None, num_nodes*batch_size, num_nodes*batch_size)])  # [Node features, adjacency matrix]
# model.summary()

optimizer = Adam(learning_rate=learning_rate)
loss_fn = SparseCategoricalCrossentropy()

# Compile the model
model.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy,
            metrics=["sparse_categorical_accuracy"],
            optimizer=optimizer)


# ### Compiling and training the model

# In[ ]:


def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(sparse_categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(sparse_categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


# In[ ]:


import time

start_time = time.time()

epoch = step = 0
best_val_loss = np.inf
best_val_acc = -200
best_weights = None
patience = es_patience
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time is: {elapsed_time/60} minutes.")


model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
np.savetxt(acc_save_name,np.asarray([test_acc]), fmt='%1.4f')