# Wave-Forecasting

## Content
This repository contains the codes for performing rogue wave forecasting using machine learning and deep learning techniques. Specifically, given a time window of wave data, the aim was to predict whether there will be a rogue wave over the horizon of the next few minutes. For this classification problem, LSTM, Decsion Trees, Support Vector Machine Classfier and Graph Neural Networks have been utilized.
The data has been acquired from CDIP buoys, both off-shore and near-shore, around the coast of United States. The buoys are located in both deep and shallow water. For the training process of the different models, equal number of windows for rogue wave and non-rogue wave scenarios have been used. Different metrics including accuracy, precision, recall and F1 score were finally utilized to test the performance of the trained models on the unseen test datasets.
Different buoy location scenarios were also considered. Wave data from localized buoy networks in both near shore and offshore locations have been used for case studies on real-time rogue wave prediction.
For the graph neural networks, two different rogue wave classification situations have been examined. 
- In the first case, **graph isomorphism networks coupled with Multi-Layer Perceptrons have been utilized to carry out graph-level classification** of different rogue wave groups categorized by their magnitudes. More than 71% accuracy has been obtained for the largest rogue waves.
- In the second case, graph convolution networks coupled with LSTM are being utilized for performing spatio-temporal classification on a localized buoy network. Given a time window from a buoy not being used for training, the task of the traained model will be to correctly predict the occurrence of a rogue wave by training it with the wave data from the other buoys in the buoy network. The edges for this graph are constructed based on the inter-buoy distances as well as the similarity metrics between the wave data. This model is then used for **node-level classification**. As of now, 60% accuracy has been obtained on this classification scenario. 

## License
This software is made public for research use only. It may be modified and redistributed under the terms of the MIT License.

