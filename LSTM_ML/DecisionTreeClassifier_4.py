#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import IPython
import IPython.display
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="serif"

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np
from sklearn import metrics
import seaborn as sns

file_str="RWs_H_g_2_tadv_10min_rw_smallWindow_0.5"
file_str_test="RWs_H_g_2_tadv_10min_rw_smallWindow_0.5"

dt_save_name= os.getcwd() + "/model_saves_dt" + "/" +  "/best_model_" + file_str + ".pkl"
metrics_save_name = os.getcwd() + "/metric_saves_dt" + "/" + file_str + ".txt"

data=np.load(file_str+".npz")
data_test=np.load(file_str_test+".npz")

for vars in data:
    print(vars)

wave_data_train=data["wave_data_train"]
wave_data_test=data_test["wave_data_test"]
label_train=data["label_train"]
label_test=data_test["label_test"]

print(wave_data_train.shape)
print(wave_data_test.shape)

x_train = wave_data_train.reshape((wave_data_train.shape[0], wave_data_train.shape[1] * wave_data_train.shape[2]))
x_test = wave_data_test.reshape((wave_data_test.shape[0], wave_data_test.shape[1] * wave_data_test.shape[2]))

clf = DecisionTreeClassifier(random_state = 0)
clf.fit(x_train, label_train)

with open(dt_save_name,'wb') as f:
    pickle.dump(clf,f)
    
with open(dt_save_name,'rb') as f:
    clf = pickle.load(f)
    
label_pred = clf.predict(x_test)

confusion_matrix = metrics.confusion_matrix(label_test, label_pred)
print('Confusion matrix')
print(confusion_matrix)
print('---------------')
print('Accuracy:', metrics.accuracy_score(label_test, label_pred))
print('Precision:', metrics.precision_score(label_test, label_pred))
print('Recall:', metrics.recall_score(label_test, label_pred))
print('F1 Score:', metrics.f1_score(label_test, label_pred))

lines = ['Confusion matrix\n', f"{confusion_matrix}\n", "---------------\n",          f" 'Accuracy:', {metrics.accuracy_score(label_test, label_pred)}",         f" 'Precision:', {metrics.precision_score(label_test, label_pred)}",         f" 'Recall:', {metrics.recall_score(label_test, label_pred)}",         f" 'F1 Score:', {metrics.f1_score(label_test, label_pred)}"]
with open(metrics_save_name, "w") as f:
    f.writelines(lines)
    
group_names = ['Correctly predicted','Incorrectly predicted',                'Incorrectly predicted','Correctly predicted']
group_counts = ["{0:0.0f}".format(value) for value in
                confusion_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                        confusion_matrix.flatten()/np.sum(confusion_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
yaxislabels = ['Rogue waves absent','Rogue waves present']
xaxislabels = ['Predicted as absent','Predicted as present']
plt.figure(figsize=[6,6])
s = sns.heatmap(confusion_matrix, annot=labels, yticklabels=yaxislabels, xticklabels=xaxislabels, fmt='', cmap='Blues')
s.set_xlabel("Predicted label", fontsize = 10)
s.set_ylabel("True label", fontsize=10)
filename=os.getcwd()+'/confusion_matrices_dt'+'/'+file_str+'.jpg'
plt.savefig(filename,dpi=199)
