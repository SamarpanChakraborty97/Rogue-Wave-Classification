#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import IPython
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

relative_rw = [0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8]
test_accuracies = []
for i in range(len(relative_rw)):
    file_str = f"RWs_H_g_2_tadv_3min_rw_smallWindow_{relative_rw[i]}"
    file_str_test = "RWs_H_g_2_tadv_3min_rw_smallWindow_0.5"

    svm_save_name= os.getcwd() + "/model_saves" + "/" +  "/best_model_" + file_str + ".pkl"
    metrics_save_name = os.getcwd() + "/metric_saves" + "/" + file_str + ".txt"

    data_test = np.load(file_str_test+".npz")

    for vars in data_test:
        print(vars)
    
    wave_data_train=data_test["wave_data_train"]
    wave_data_test=data_test["wave_data_test"]
    label_train=data_test["label_train"]
    label_test=data_test["label_test"]

    print(wave_data_train.shape)
    print(wave_data_test.shape)

    x_test = wave_data_test.reshape((wave_data_test.shape[0], wave_data_test.shape[1] * wave_data_test.shape[2]))

    clf = svm.SVC(kernel='rbf', random_state = 0, verbose=True)

    with open(svm_save_name,'rb') as f:
        clf = pickle.load(f)
    
    label_pred = clf.predict(x_test)

    confusion_matrix = metrics.confusion_matrix(label_test, label_pred)
    print('Confusion matrix')
    print(confusion_matrix)
    print('---------------')
    print('Precision:', metrics.precision_score(label_test, label_pred))
    print('Recall:', metrics.recall_score(label_test, label_pred))
    print('F1 Score:', metrics.f1_score(label_test, label_pred))

    lines = ['Confusion matrix\n', f"{confusion_matrix}\n", "---------------\n",          f" Precision:, {metrics.precision_score(label_test, label_pred)}\n",         f" Recall:, {metrics.recall_score(label_test, label_pred)}\n",         f" F1 Score:, {metrics.f1_score(label_test, label_pred)}\n"]
    
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
    filename=os.getcwd()+'/confusion_matrices_svm'+'/'+file_str+'.jpg'
    plt.savefig(filename,dpi=199)

