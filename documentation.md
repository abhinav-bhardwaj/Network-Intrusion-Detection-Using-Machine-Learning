# Documentation
## Overview
Loosely based on research paper **A Novel Statistical Analysis and Autoencoder Driven Intelligent Intrusion Detection Approach** 
[https://doi.org/10.1016/j.neucom.2019.11.016](https://doi.org/10.1016/j.neucom.2019.11.016)

## Datasets
-   **bin_data.csv**  - CSV Dataset file for Binary Classification
-   **multi_data.csv**  - CSV Dataset file for Multi-class Classification
-   **KDDTrain+.txt**  - Original Dataset downloaded

The NSL-KDD dataset from the Canadian Institute for Cybersecurity (updated version of the original KDD Cup 1999 Data (KDD99) [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)

## Machine Learning Models

 - Linear Support Vector Machine
 - Quadratic Support Vector Machine
 - K-Nearest-Neighbor
 - Linear Discriminant Analysis
 - Quadratic Discriminant Analysis
 - Multi Layer Perceptron
 - Long Short-Term Memory
 - Auto Encoder

## Data Preprocessing
- Dataset had 43 attributes, attribute **'difficulty_level'** was dropped.

- ### Data Normalization
	- 38 Numeric Columns of DataFrame is scaled using **Standard Scaler**. 
	
- ### One-hot-encoding
	- Categorical Columns **'protocol_type'**, **'service'**, **'flag'** are one-hot-encoded using **pd.get_dummies()**.
	- **'categorical'** Dataframe had 84 attributes after one-hot-encoding.
	
- ### Binary Classification
	- A copy of DataFrame is created for Binary Classification.
	- Attack label (**'label'** attribute) is classified into two categories **'normal'** and **'abnormal'**.
	- **'label'** is encoded using **LabelEncoder()**, encoded labels are saved in **'intrusion'**.
	- **'label'** is one-hot-encoded.
	
- ### Multi-class Classification
	- A copy of DataFrame is created for Multi-class Classification. 
	- Attack label (**'label'** attribute) is classified into five categories **'normal'**, **'U2R'**, **'R2L'**, **'Probe'**, **'Dos'**.
	- **'label'** is encoded using **LabelEncoder()**, encoded labels are saved in **'intrusion'**.
	- **'label is one-hot-encoded'**.
	
- ### Feature Extraction
	- No. of attributes of **'bin_data'**  - 45
	- No. of attributes of **'multi_data'** - 48
	- The attributes of **'bin_data'** and **'multi_data'** are selected using **'Pearson Correlation Coefficient'**.
	- The attributes with more than 0.5 correlation coefficient with the target attribute **'intrusion'** were selected.
	- 9 attributes **'count'**, **'srv_serror_rate'**, **'serror_rate'**, **'dst_host_serror_rate'**, **'dst_host_srv_serror_rate'**, **'logged_in'**, **'dst_host_same_srv_rate'**, **'dst_host_srv_count'**, **'same_srv_rate'**.
	- No. of attributes of **'bin_data'** after feature selection and joining **'categorical'** DataFrame - 97
	- No. of attributes of **'multi_data'** after feature selection and joining **'categorical'** DataFrame - 100

## Splitting the dataset
- Splitting the dataset into 1:4 Ratio for Testing and Training.
- 93 attributes were selected out of 97 attributes, to exclude the target attribute (encoded, one-hot-encoded, original) for Binary Classification
- **'intrusion'** attribute was selected as the target attribute.
- 93 attributes were selected out of 100 attributes, to exclude the target attribute (encoded, one-hot-encoded, original) for Multi-class Classification.
 
## Linear Support Vector Machine
- Binary Classification Accuracy - **96.69 %**
- Multi-class Classification Accuracy - **95.24 %**
- Kernel Type used - **Linear**
- `SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)`
 
## Quadratic Support Vector Machine
- Binary Classification Accuracy - **95.71 %**
- Multi-class Classification Accuracy - **92.86 %**
- Kernel Type used - **Poly**
- `SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)`

## K-Nearest-Neighbor
- Binary Classification Accuracy - **98.55 %**
- Multi-class Classification Accuracy - **98.29 %**
- No. of neighbors - **5**
- Weights - **Uniform**
- `KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=5, p=2,weights='uniform')`

## Linear Discriminant Analysis
- Binary Classification Accuracy - **96.70 %**
- Multi-class Classification Accuracy - **93.19 %**
- Solver used - **svd (singular value decomposition)**
- `LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)`
  
## Quadratic Discriminant Analysis
 - Binary Classification Accuracy - **68.79 %**
 - Multi-class Classification Accuracy - **44.96 %**
 - `QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)`
 
## Multi Layer Perceptron 
- Binary Classification Accuracy - **97.79 %**
	- **Input layer** with **93** input dimensions
	- **1 Hidden layer** with **50 Neurons** and **relu** activation function
	- Output layer with **1 neuron** and **sigmoid** activation function
	- Loss - **binary_crossentropy**
	- Optimizer - **adam** 
	- Batch size - **5000**
	- Epochs - **100**
- Multi-class Classification Accuracy - **96.92 %**
	- **Input layer** with **93** input dimensions
	- **1 Hidden layer** with **50 Neurons** and **relu** activation function
	- Output layer with **5 neurons** and **softmax** activation function
	- Loss - **categorical_crossentropy**
	- Optimizer - **adam** 
	- Batch size - **5000**
	- Epochs - **100**

## Long Short-Term Memory
- Binary Classification Accuracy - **83.05 %**
- **Input layer** with **93** input dimensions
- **LSTM** layer with **50 encoding cells**
- Output layer with **1 neuron** and **sigmoid** activation function
- Loss - **binary_crossentropy**
- Optimizer - **adam**
- Batch Size - **5000**
- Epochs - **100**

## Autoencoder
- Binary Classification Accuracy - **92.26 %**
- Multi-class Classification Accuracy - **91.22 %**
- **Input layer**
- **Encoding layer** with **50 encoding cells**
- **Output layer** and **Decoding Layer** with **softmax** activation function
- Loss - **mean_squared_error**
- Optimizer - **adam**
- Batch Size - **500**
- Epochs - **100**

## Citations
-   Cosimo Ieracitano, Ahsan Adeel, Francesco Carlo Morabito, Amir Hussain, A Novel Statistical Analysis and Autoencoder Driven Intelligent Intrusion Detection Approach, Neurocomputing (2019), doi:  [https://doi.org/10.1016/j.neucom.2019.11.016](https://doi.org/10.1016/j.neucom.2019.11.016)
    
-   The NSL-KDD dataset from the Canadian Institute for Cybersecurity (updated version of the original KDD Cup 1999 Data (KDD99)  [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)

> Written with [StackEdit](https://stackedit.io/).
