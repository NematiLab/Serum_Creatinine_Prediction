# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:45:00 2024

@author: gghanbari
"""

import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import matplotlib.pyplot as plt

np.random.seed(12345*4)
tf.random.set_seed(42)

num_features = 241
num_hours = 24
                
num_epochs = 140
batch_size = 800

mae_train =[]
rmse_train =[]
mae_test=[]
rmse_test=[]

unst_mae_train =[]
unst_rmse_train =[]
unst_mae_test=[]
unst_rmse_test=[]

for fold in range(1,11):
    
    print(fold)
    BaseDir=f'D:/Kidney_Project/Model_input_data/kFold_CV/fold{fold}/' 
    
    train_X= np.load(BaseDir + 'train_Daily_data_input_Xnorm.npy')
    test_X= np.load(BaseDir + 'test_Daily_data_input_Xnorm.npy')
    train_Y= np.load(BaseDir + 'train_Daily_creatinine_6am_Y.npy')
    test_Y= np.load(BaseDir + 'test_Daily_creatinine_6am_Y.npy')
    train_patients_data = pd.read_csv(BaseDir + 'train_patients_data.csv', index_col=None)
    test_patients_data = pd.read_csv(BaseDir + 'test_patients_data.csv')
    
    
    nan_indices_train = np.isnan(train_Y)
    nan_indices_test = np.isnan(test_Y)
    
    train_X_cleaned = train_X[~nan_indices_train]
    train_Y_cleaned = train_Y[~nan_indices_train]
    
    # Normalizing the data
    max_ytrain= np.max(train_Y_cleaned)
    normalized_ytrain= train_Y_cleaned / max_ytrain
    
    test_X_cleaned = test_X[~nan_indices_test]
    test_Y_cleaned = test_Y[~nan_indices_test]
    
    # Normalizing the data
    normalized_ytest= test_Y_cleaned / max_ytrain
    
    # ------- Model structure --------
    input_layer = Input(shape=(num_hours, num_features))

    attention_output = MultiHeadAttention(num_heads=2, key_dim=128)(input_layer, input_layer)
    layer_norm = LayerNormalization(epsilon=1e-6)(attention_output + input_layer)

    # Flatten the output of the final attention layer
    flatten_layer = tf.keras.layers.GlobalAveragePooling1D()(layer_norm)

    dense_layer1 = Dense(128, activation='relu', kernel_regularizer=l2(0.0003))(flatten_layer)
    dropout_layer1 = Dropout(0.1)(dense_layer1)

    dense_layer2 = Dense(64, activation='relu', kernel_regularizer=l2(0.0003))(dropout_layer1)
    dropout_layer2 = Dropout(0.1)(dense_layer2)
    # Output layer for regression
    output_layer = Dense(1)(dropout_layer2)

    # ------------ Create the model---------
    model = Model(inputs=input_layer, outputs=output_layer)

    learning_rate = 0.001
    optimizer1=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer1, metrics=['mae'])
    
    checkpoint_path = f'C:/Users/gghanbari/Downloads/Kidney_project_codes_v0/\
    multi-head_attention_1D_checkPoint/kFold_CV_new1/2Head_128key_800batch_fold{fold}.hdf5'
    
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True,  # Save only the best model
        mode='min', verbose=1)
    
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    
    history = model.fit(train_X_cleaned, normalized_ytrain, epochs=num_epochs, 
                         batch_size=batch_size, callbacks=[early_stopping, checkpoint_callback], validation_data=(test_X_cleaned, normalized_ytest))
    
    
    best_model = Model(inputs=input_layer, outputs=output_layer)  # Recreating the model
    
    best_model.load_weights(checkpoint_path) 
    
    best_model.compile(loss='mean_squared_error', optimizer=optimizer1, metrics=['mae'])
    
    filepath = f'C:/Users/gghanbari/Downloads/Kidney_project_codes_v0/Final_model_kFolds/Final_model_fold{fold}.keras'
    
    tf.keras.saving.save_model(best_model, filepath, overwrite=True, save_format='keras')
    
    # print("Created model and loaded weights from file")
    
    train_predictions = best_model.predict(train_X_cleaned)
    test_predictions = best_model.predict(test_X_cleaned)
    
    train_predictions = train_predictions.reshape(train_predictions.shape[0],)
    test_predictions = test_predictions.reshape(test_predictions.shape[0],)
    
    # Convert to original shape, nonnormalized
    train_predictions_org= train_predictions*max_ytrain
    test_predictions_org= test_predictions*max_ytrain
    
    
    #   =========== Perform Bland-Altman analysis  ==========
    
    # Calculate the difference between data1 and data2
    diff = train_Y_cleaned - train_predictions_org
    mean_values = (train_Y_cleaned + train_predictions_org)/2
    
    plt.figure(figsize=(8, 6))
    points_size = 1
    plt.scatter(mean_values, diff, s=points_size, c='b', marker='o')
    plt.axhline(np.mean(diff), color='r', linestyle='--', label='Mean Difference')
    # Calculate the Upper and Lower LOA
    upper_loa = np.mean(diff) + 1.96 * np.std(diff, ddof=1)
    lower_loa = np.mean(diff) - 1.96 * np.std(diff, ddof=1)
    
    plt.axhline(upper_loa, color='g', linestyle='--', label='+ 1.96 SD')
    plt.axhline(lower_loa, color='g', linestyle='--', label='- 1.96 SD')
    
    plt.xlim(0,13)
    plt.ylim(-4,11)
    plt.xlabel('Mean of true SCr and predictions')
    plt.ylabel('Difference between true SCr values and predictions')
    plt.legend()
    plt.title('Bland-Altman Plot for multi-head Attention Model, Train set')
    plt.grid(True)
    resolution_value = 600
    #plt.savefig('C:/Users/gghanbari/Downloads/Attention_plots/Bland-Altman_mean_x_axis_0.1.png', format="png", dpi=resolution_value)#format='eps')
    plt.show()
    
     #  test set
    diff = test_Y_cleaned - test_predictions_org
    mean_values = (test_Y_cleaned + test_predictions_org)/2
    # Create a Bland-Altman plot with data1 on the x-axis
    plt.figure(figsize=(8, 6))
    points_size = 1
    
    plt.scatter(mean_values, diff, s=points_size, c='b', marker='o')
    plt.axhline(np.mean(diff), color='r', linestyle='--', label='Mean Difference')
    # Calculate the Upper and Lower LOA
    upper_loa = np.mean(diff) + 1.96 * np.std(diff, ddof=1)
    lower_loa = np.mean(diff) - 1.96 * np.std(diff, ddof=1)
    
    plt.axhline(upper_loa, color='g', linestyle='--', label='+ 1.96 SD')
    plt.axhline(lower_loa, color='g', linestyle='--', label='- 1.96 SD')
    
    plt.xlim(0,13)
    plt.ylim(-4,11)
    plt.xlabel('Mean of true SCr and predictions')
    plt.ylabel('Difference between true SCr values and predictions')
    plt.legend()
    plt.title('Bland-Altman Plot for multi-head Attention Model, test set')
    plt.grid(True)
    resolution_value = 600
    plt.show()
    
    # =================   Physical units of errors ====================:  
    errors_train = train_predictions_org - train_Y_cleaned
    abs_error_tr= np.abs(errors_train)
    
    rmse_train.append(np.sqrt(np.mean(errors_train**2)))
    mae_train.append(np.mean(abs_error_tr))
    
    errors_test= test_predictions_org - test_Y_cleaned
    abs_error_te =np.abs(errors_test)
    
    rmse_test.append(np.sqrt(np.mean(errors_test**2)))
    mae_test.append(np.mean(abs_error_te))
    
    # print(f'\nMAE:\nTrain error: {mae_train:.2f}  \nTest: {mae_test:.4f}')
    # print(f'\nRMSE:\nTrain : {rmse_train:.2f}   \nTest: {rmse_test:.4f}')
    
    # ============== Unstable cases =============================
    
    unstCr_train_X = np.load(BaseDir + 'unstSCr_train_X.npy')
    unstCr_train_Y = np.load(BaseDir + 'unstSCr_train_Y.npy')
    unstCr_test_X = np.load(BaseDir + 'unstSCr_test_X.npy')
    unstCr_test_Y = np.load(BaseDir + 'unstSCr_test_Y.npy')
    
    unst_train_predictions = best_model.predict(unstCr_train_X)
    unst_test_predictions = best_model.predict(unstCr_test_X)
    
    unst_train_predictions = unst_train_predictions.reshape(unst_train_predictions.shape[0],)
    unst_test_predictions = unst_test_predictions.reshape(unst_test_predictions.shape[0],)
    
    # Convert to original shape, nonnormalized
    unst_train_predictions_org= unst_train_predictions*max_ytrain
    
    unst_test_predictions_org= unst_test_predictions*max_ytrain
    
    # ===== MAE and RMSE for unstable days ======    
  
    unst_errors_train = unstCr_train_Y - unst_train_predictions_org
    unst_abs_error_train= np.abs(unst_errors_train)
    
    unst_rmse_train.append(np.sqrt(np.mean(unst_errors_train**2)))
    unst_mae_train.append(np.mean(unst_abs_error_train))
    
    unst_errors_test = unstCr_test_Y - unst_test_predictions_org
    unst_abs_error_test= np.abs(unst_errors_test)
    
    unst_rmse_test.append(np.sqrt(np.mean(unst_errors_test**2)))
    unst_mae_test.append(np.mean(unst_abs_error_test))
    
    # print(f'\nTrain unstable:\nMAE: {unst_mae_train:.2f}  \nRMSE: {unst_rmse_train:.2f}')
    # print(f'\nTest unstable:\nMAE: {unst_mae_test:.2f}   \nRMSE: {unst_rmse_test:.2f}')
