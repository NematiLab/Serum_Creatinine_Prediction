# Serum_Creatinine_Prediction
This repo includes codes for the paper "Development and Validation of a Deep Learning Algorithm for the Prediction of Serum Creatinine in Critically Ill Patients"


### Multi-Head Attention Model for Creatinine Prediction

# Purpose:
This code implements a multi-head attention-based model for predicting Serum Creatinine levels in patients. The model is trained and evaluated using a k-fold cross-validation strategy.

# Data:
The data used for training and testing is loaded from NumPy files (.npy) located in the "D:/Kidney_Project/Model_input_data/kFold_CV/fold{fold}/" directory. The data is divided into folds, with each fold containing training and testing sets.

# Model:
The model architecture consists of:

Input layer: Takes time-series data with shape (num_hours, num_features).
Multi-head attention layer: Employs multiple attention heads to capture complex temporal relationships.
Layer normalization: Normalizes the output of the attention layer.
Global average pooling: Aggregates temporal information.
Dense layers: Fully connected layers for feature extraction and transformation.
Dropout layers: Prevent overfitting.
Output layer: Single neuron for regression, predicting Serum Creatinine levels.

# Training:
The model is trained using mean squared error loss and Adam optimizer. Early stopping and model checkpointing are used to monitor validation loss and save the best model.

# Evaluation:
The trained model is evaluated on the testing set using mean absolute error (MAE) and root mean squared error (RMSE). Bland-Altman plots are generated to visualize the agreement between predicted and actual Serum Creatinine values.

Additional Analysis:
The code also performs analysis on unstable creatinine cases, calculating MAE and RMSE for these specific instances.

# File Structure:

kFold_CV folders: Contain training and testing data for each fold.
train_Daily_data_input_Xnorm.npy: Normalized input features for training.
test_Daily_data_input_Xnorm.npy: Normalized input features for testing.
train_Daily_creatinine_6am_Y.npy: Creatinine target values for training.
test_Daily_creatinine_6am_Y.npy: Creatinine target values for testing.
train_patients_data.csv: Patient information for the training set.
test_patients_data.csv: Patient information for the testing set.

# Key Parameters:

num_features: Number of input features (241).
num_hours: Number of time steps in the input data (24).
num_epochs: Number of training epochs (140).
batch_size: Batch size for training (800).
learning_rate: Learning rate for the Adam optimizer (0.001).
# Output:
The code saves the best trained model for each fold in the "Final_model_kFolds" directory. It also generates Bland-Altman plots and prints MAE and RMSE values for both the overall and unstable cases.

