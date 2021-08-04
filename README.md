# Network-Intrusion-Detection-Using-Deep-Learning
Loosely based on research paper **A Novel Statistical Analysis and Autoencoder Driven Intelligent Intrusion Detection Approach** 
https://doi.org/10.1016/j.neucom.2019.11.016

## Blog of this Project
- ## [Network Intrusion Detection using Deep Learning on Medium.com](https://medium.com/geekculture/network-intrusion-detection-using-deep-learning-bcc91e9b999d?source=friends_link&sk=2b84dd61f3e76d63af0a14daf6f89f43)

## Repository Structure
Network-Intrusion-Detection-Using-Machine-Learning
  - Datasets
    - **bin_data.csv** - CSV Dataset file for Binary Classification
    - **multi_data.csv** - CSV Dataset file for Multi-class Classification
    - **KDDTrain+.txt** - Original Dataset downloaded

  - Labels
    - **le1_classes.npy** - Numpy file for ndarray containing Binary Labels
    - **le2_classes.npy** - Numpy file for ndarray containing Multi-class Labels

  - Models
    - **ae_binary.json** - Trained Auto Encoder Model JSON File for Binary Classification
    - **ae_multi.json** - Trained Auto Encoder Model JSON File for Multi-class Classification
    - **knn_binary.pkl** - Trained K-Nearest-Neighbor Model Pickle File for Binary Classification
    - **knn_multi.pkl** - Trained K-Nearest-Neighbor Model Pickle File for Multi-class Classification
    - **lda_binary.pkl** - Trained Linear Discriminant Analysis Model Pickle File for Binary Classification
    - **lda_multi.pkl** - Trained Linear Discriminant Analysis Model Pickle File for Multi-class Classification
    - **lst_binary.json** - Trained Long Short-Term Memory Model JSON File for Binary Classification
    - **lsvm_binary.pkl** - Trained Linear Support Vector Machine Model Pickle File for Binary Classification
    - **lsvm_multi.pkl** - Trained Linear Support Vector Machine Model Pickle File for Multi-class Classification
    - **mlp_binary.json** - Trained Multi Layer Perceptron Model JSON File for Binary Classification
    - **mlp_multi.json** - Trained Multi Layer Perceptron Model JSON File for Multi-class Classification
    - **qda_binary.pkl** - Trained Quadratic Discriminant Analysis Model Pickle File for Binary Classification
    - **qda_multi.pkl** - Trained Quadratic Discriminant Analysis Model Pickle File for Multi-class Classification
    - **qsvm_binary.pkl** - Trained Quadratic SUpport Vector Machine Model Pickle File for Binary Classification
    - **qsvm_multi.pkl** - Trained Quadratic Support Vector Machine Model Pickle File for Multi-class Classification
    
  - Weights
    - **ae_binary.h5** - Model weights of Auto Encoder Model for Binary Classification
    - **ae_multi.h5** - Model weights of Auto Encoder Model for Multi-class Classification
    - **lst_binary.h5** - Model weights of Long Short-Term Memory Model for Binary Classification
    - **mlp_binary.h5** - Model weights of Multi Layer Perceptron Model for Binary Classification
    - **mlp_multi.h5** - Model weights of Multi Layer Perceptron Model for Multi-class Classification

  - Plots
    - **Pie_chart_binary.png** - Pie chart of Binary Classification
    - **Pie_chart_multi.png** - Pie chart of Multi-class Classification
    - **ae_binary.png** - Auto Encoder Model Summary for Binary Classification
    - **ae_binary_accuracy.png** - Auto Encoder Accuracy Plot for Binary Classification
    - **ae_binary_loss.png** - Auto Encoder Loss Plot for Binary Classification
    - **ae_multi.png** - Auto Encoder Model Summary for Multi-class Classification
    - **ae_multi_accuracy.png** - Auto Encoder Accuracy Plot for Multi-class Classification
    - **ae_multi_loss.png** - Auto Encoder Loss Plot for Multi-class Classification
    - **lstm_binary.png** - Long Short-Term Memory Model Summary for Binary Classification
    - **lstm_binary_accuracy.png** - Long Short-Term Memory Accuracy Plot for Binary Classification
    - **lstm_binary_loss.png** - Long Short-Term Memory Loss Plot for Binary Classification
    - **mlp_binary.png** - Multi Layer Perceptron Model Summary for Binary Classification
    - **mlp_binary_accuracy.png** - Multi Layer Perceptron Accuracy Plot for Binary Classification
    - **mlp_binary_loss.png** - Multi Layer Perceptron Loss Plot for Binary Classification
    - **mlp_multi.png** - Multi Layer Perceptron Model Summary for Multi-class Classification
    - **mlp_multi_accuracy.png** - Multi Layer Perceptron Accuracy Plot for Multi-class Classification
    - **mlp_multi_loss.png** - Multi Layer Perceptron Loss Plot for Multi-class Classification
    
  - **Classifiers_NSL-KDD.ipynb** - Machine Learning Classifiers IPYNB file 

  - **Data_Preprocessing_NSL-KDD.ipynb** - Data Preprocessing IPYNB File 

  - **Intrusion_Detection.ipynb** - Combined IPYNB File

## Dataset
The NSL-KDD dataset from the Canadian Institute for Cybersecurity (updated version of the original KDD Cup 1999 Data (KDD99)
https://www.unb.ca/cic/datasets/nsl.html

## Prerequisites
 - Keras 
 - Sklearn 
 - Pandas 
 - Numpy
 - Matplotlib
 - Pickle

## Running the Notebook
The notebook can be run on 
 - Google Colaboratory
 - Jupyter Notebook
 
## Instructions
 - To run the code, user must have the required Dataset on their system or programming environment.
 - Upload the notebook and dataset on Jupyter Notebook or Google
   Colaboratory.
 - Click on the file with .ipynb extension to open the notebook. To run
   complete code at once press Ctrl + F9
 - To run any specific segment of code, select that code cell and press
   Shift+Enter or Ctrl+Shift+Enter

>**Caution - The code should be executed in the given order for best results without encountering any errors.**

## Citation

 - Cosimo Ieracitano, Ahsan Adeel, Francesco Carlo Morabito, Amir
   Hussain, A Novel Statistical Analysis and Autoencoder Driven
   Intelligent Intrusion Detection Approach, Neurocomputing (2019), doi:
   https://doi.org/10.1016/j.neucom.2019.11.016
   
 - The NSL-KDD dataset from the Canadian Institute for Cybersecurity (updated version of the original KDD Cup 1999 Data (KDD99)
  https://www.unb.ca/cic/datasets/nsl.html
