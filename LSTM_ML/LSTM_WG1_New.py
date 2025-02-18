#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import IPython
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from tensorflow import keras
import seaborn as sns

import random
import tensorflow as tf
import matplotlib.pyplot as plt


# In[5]:


file_str="tadv_5min_wave_group_window_15mins_1"


# In[6]:


data=np.load(file_str+".npz")

for vars in data:
    print(vars)

wave_data_train=data["wave_data_train"]
wave_data_test=data["wave_data_test"]
label_train=data["label_train"]
label_test=data["label_test"]
num_classes=2


# In[7]:


print(wave_data_train.shape)
print(wave_data_test.shape)


# ## LSTM model

# In[8]:


batch_size=32
patience = 20
num_hidden_units = 40
normalization_type = keras.layers.LayerNormalization
dropout = 0.05
num_stacks = 4


# In[13]:


from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay

tf.random.set_seed(30)

model_LSTM = Sequential()
for i in range(num_stacks):
    if i==0:
        model_LSTM.add(LSTM(num_hidden_units, input_shape = wave_data_train.shape[1:], return_sequences=True))
        model_LSTM.add(normalization_type())
        model_LSTM.add(Dropout(dropout))
        
    elif (i>0) and (i < num_stacks-1):
        model_LSTM.add(LSTM(num_hidden_units, return_sequences=True))
        model_LSTM.add(normalization_type())
        model_LSTM.add(Dropout(dropout))
    
    else:
        model_LSTM.add(LSTM(num_hidden_units))
        model_LSTM.add(normalization_type())
        model_LSTM.add(Dropout(dropout))

model_LSTM.add(Dense(num_classes, activation = 'sigmoid'))

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
model_LSTM.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
            optimizer=optimizer)

model_LSTM.summary()


# In[15]:


save_name = "accuracy_new_lstm_wg1"
LSTM_save_name = os.getcwd() + "/model_saves/" + save_name + ".h5"
#LSTM_save_name = os.getcwd() + "/model_saves/" + save_name + ".checkpoint.model.keras"
plot_save_name = os.getcwd() + "/training_history_/" + save_name + ".jpg"
acc_save_name = os.getcwd() + "/accuracy_saves_/" + save_name + ".txt"

import time
start_time = time.time()
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=300,  # number of batches after which the learning rate decays
    decay_rate=0.9,
    staircase=False)

def scheduler(epoch, lr):
    return float(lr_schedule(epoch * batch_size))  # Assuming 100 steps per epoch (example)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

es = EarlyStopping(monitor='val_loss', patience=patience)
mc = ModelCheckpoint(LSTM_save_name, save_best_only=True, monitor = "val_loss")

history = model_LSTM.fit(wave_data_train, label_train, epochs=500, batch_size=batch_size, validation_split=0.2, callbacks=[es, mc, lr_scheduler], verbose=0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time is: {elapsed_time/60} minutes.")


# In[ ]:


model_LSTM = keras.models.load_model(LSTM_save_name)

test_loss, test_acc = model_LSTM.evaluate(wave_data_test, label_test, verbose = 0)
print("Test accuracy", test_acc)
print("Test loss", test_loss)


# In[ ]:


metric = "accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")

plt.savefig(plot_save_name,dpi=199)
plt.show()
plt.close()
np.savetxt(acc_save_name,np.asarray([test_acc]), fmt='%1.4f') 

