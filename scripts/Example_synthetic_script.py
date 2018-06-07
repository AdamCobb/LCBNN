# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"

os.system('export CUDA_HOME=/opt/cuda-8.0')
os.system('export LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cuda-8.0/extras/CUPTI/lib64')

import sys
sys.path.insert(0, '../code/')
sys.path.insert(0, '../data/')
import lcbnn_multiclass as lcbnn
import synthetic
from utils import plot_confusion_matrix
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import accuracy_score
import pickle

# Keras
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.metrics import binary_accuracy
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Layer, Lambda
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adadelta, adam

#########################
####### Settings ########
#########################
save_name = 'exp_1_R_30'
N_seeds = 10 #0
batch_size = 50
epochs = 200
noise = 0.2
num_classes = 3
N_per_class = 50
R_per_class = 30
N_val = 50
N_test = 100
N_train = num_classes * N_per_class

# ## Set up utility
M = 2.00001
loss_mat = np.array([[0,1,2],[0.8,0,0.7],[0.9,0.6,0]])
print("         TRUTH\n" )
print('P  ',loss_mat[0])
print('R  ',loss_mat[1])
print('E  ',loss_mat[2])

# Set up weight for weighted cross entropy
class_weight = {0 : 1,
    1: 2,
    2: 3}


# ## General network structure

Early_Stop_acc = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
# units
units = 20

dropout=0.2
# Regularise
tau = 1.0
lengthscale = .01
reg = lengthscale**2 * (1 - dropout) / (2. * N_train * tau)

def network():
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=x_train.shape[1:],W_regularizer=l2(reg)))
    model.add(Lambda(lambda x: K.dropout(x,level=dropout)))
    model.add(Dense(num_classes, activation='softmax',W_regularizer=l2(reg)))
    return model



########################
#######   LOOP  ########
########################

results_array = np.zeros((N_seeds,3,2))
for seed in range(N_seeds):

    np.random.seed(seed=seed)
    random.seed(seed)
    # ## Import data
    N_train = num_classes*N_per_class
    x_train, y_train, x_train_ordered, y_train_ordered = synthetic.create_synthetic_train(num_classes=num_classes, 
                                                                                N_per_class=N_per_class, 
                                                                                input_noise=noise, R_per_class=R_per_class,
                                                                                         seed=seed)

    x_val, y_val = synthetic.create_synthetic_test(num_classes=num_classes, N_per_class=N_val, input_noise=noise,seed=seed)
    x_test, y_test = synthetic.create_synthetic_test(num_classes=num_classes, N_per_class=N_test, input_noise=noise,seed=seed)


    # ### Standard
    weights_file_std = '../models/synth/synth_weights_std.h5'
    model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_std, monitor='val_loss', save_best_only=True,
                                       save_weights_only=True, mode='auto',verbose=0)
    std_model = network()
    std_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    history_std = std_model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=[x_val,y_val],
                      callbacks=[model_checkpoint])
    std_model.load_weights(weights_file_std)


    # ### Weighted
    weights_file_we = '../models/synth/synth_weights_we.h5'
    model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_we, monitor='val_loss', save_best_only=True,
                                       save_weights_only=True, mode='auto',verbose=0)
    we_model = network()
    we_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    history_we = we_model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=0,
                      validation_data=[x_val,y_val],
                      class_weight = class_weight,
                      callbacks=[model_checkpoint])
    std_model.load_weights(weights_file_we)


    # ### Loss-calibrated
    weights_file_lc = '../models/synth/synth_weights_lc.h5'
    model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_lc, monitor='val_loss', save_best_only=True,
                                       save_weights_only=True, mode='auto',verbose=0)
    M_tf = tf.constant(M, dtype=tf.float32)
    basic_model = network()
    # Initilise loss
    loss = lcbnn.loss_K(loss_mat)
    H_x = Input(shape=(num_classes,),name='H_x')
    y_true = Input(shape=(num_classes,),name='y_true')
    x = Input(shape=(x_train.shape[1:]),name='x')
    y_pred = basic_model(x)
    lc_model = Model([x,H_x],y_pred)
    lc_model.compile(loss = lcbnn.cal_loss(loss,M,H_x),optimizer='adam')

    #Initiailisations:
    y_pred_samples = np.expand_dims(y_train,0)
    y_pred_samples_val = np.expand_dims(y_val,0)
    H_x = lcbnn.optimal_h(y_pred_samples,loss_mat) #np.random.randint(0,2,(np.shape(y_train)))
    H_x_val = lcbnn.optimal_h(y_pred_samples_val,loss_mat)

    history_lc = []
    for epoch in range(epochs):
        h_lc = lc_model.fit([x_train,H_x],[y_train],
                     batch_size=batch_size,
                     nb_epoch=1,
                     verbose=0,
                     validation_data=[[x_val,H_x_val],y_val],
                     callbacks=[model_checkpoint])
        T = 10
        yt_hat = np.array([lc_model.predict([x_train,H_x]) for _ in range(T)])
        H_x = lcbnn.optimal_h(yt_hat,loss_mat)
        history_lc.append(h_lc)
        if epoch % 20 ==1:
            print('Epoch: ',epoch)        
        
    lc_model.load_weights(weights_file_lc)


    # ## Results

    T = 100
    H_x_test = np.zeros_like(y_test)

    #######
    # STD #
    #######
    yt_hat_std = np.array([std_model.predict([x_test]) for _ in range(T)])
    MC_pred_std = np.mean(yt_hat_std, 0)
    H_x_test_std = lcbnn.optimal_h(yt_hat_std,loss_mat) 
    acc_std = accuracy_score(np.argmax(y_test,1),np.argmax(H_x_test_std,1))
    loss_std = np.mean(lcbnn.loss_np(y_test,H_x_test_std,loss_mat))
    print('Standard:\n')
    print('Accuracy on optimal decision: %0.2f' % acc_std)
    print('Expected loss: %0.2f' % loss_std)

    #######
    # WEI #
    #######
    yt_hat_we = np.array([we_model.predict([x_test]) for _ in range(T)])
    MC_pred_we = np.mean(yt_hat_we, 0)
    H_x_test_we = lcbnn.optimal_h(yt_hat_we,loss_mat) 
    acc_we = accuracy_score(np.argmax(y_test,1),np.argmax(H_x_test_we,1))
    loss_we = np.mean(lcbnn.loss_np(y_test,H_x_test_we,loss_mat))

    print('\nWeighted:\n')
    print('Accuracy on optimal decision: %0.2f' % acc_we)
    print('Expected loss: %0.2f' % loss_we)

    #######
    # L-C #
    #######
    yt_hat_lc = np.array([lc_model.predict([x_test, H_x_test]) for _ in range(T)])
    MC_pred_lc = np.mean(yt_hat_lc, 0)
    H_x_test_lc = lcbnn.optimal_h(yt_hat_lc,loss_mat) 
    acc_lc = accuracy_score(np.argmax(y_test,1),np.argmax(H_x_test_lc,1))
    loss_lc = np.mean(lcbnn.loss_np(y_test,H_x_test_lc,loss_mat))
    print('\nLoss-cal:\n')
    print('Accuracy on optimal decision: %0.2f' % acc_lc)
    print('Expected loss: %0.2f' % loss_lc)

    results_array[seed,0] = [acc_std,loss_std]
    results_array[seed,1] = [acc_we,loss_we]
    results_array[seed,2] = [acc_lc,loss_lc]
    pickle.dump( results_array, open( "results_synthetic/"+save_name+".p", "wb" ) )


config_file = {'N_seeds':N_seeds, 'batch_size':batch_size, 'epochs':epochs, 'noise':noise,
               'N_per_class':N_per_class, 'R_per_class':R_per_class, 'N_val':N_val, 'N_test':N_test,
               'Bound':M, 'loss_mat':loss_mat, 'class_weight':class_weight,
               'network':{'units':units, 'dropout':dropout, 'tau':tau, 'lengthscale':lengthscale}
               }
pickle.dump( results_array, open( "results_synthetic/"+save_name+"_config.p", "wb" ) )



