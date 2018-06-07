
# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
save_name = 'exp_3_R_05_opp_weights'
# Settings
N_seeds = 10
batch_size = 50
epochs = 50
num_classes = 10
n_samples = 5000
I = np.eye(num_classes)
# ## Set up utility
digits = [3,8]
digit_weight = 0.7
L = np.copy(I)
L[np.where(I==0)] = 1
L[[3,8]] = digit_weight # Select more important rows with lower loss in prediction
L[np.where(I==1)] = 0
loss_mat = L
M = 1.0001
print("         TRUTH\n" )
string = 'PRED'
for i in range(num_classes):
    if i < 4:
        print(string[i],'  ',str(i),': ',loss_mat[i])
    else:
        print('    ',str(i),': ',loss_mat[i])
        
# Set up weight for weighted cross entropy
class_weight = np.ones((num_classes))
class_weight[digits] = digit_weight #1./digit_weight

N = int(n_samples/2)
dropout=0.2
tau = 1.0
lengthscale = .01
reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)
corrupt_prop =0.1

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

########################
#######   LOOP  ########
########################


units_list = [20,30,40,50,60,70,80,90,100]
N_units = len(units_list)
results_array = np.zeros((N_seeds,N_units,3,2))

u = 0
for units in units_list:
    print('Units: ', units)
    for seed in range(N_seeds):
        print('Seed: ',seed)
        np.random.seed(seed=seed)
        random.seed(seed)
        # Data
        x_train_S, y_train_S, x_test, y_test = synthetic.load_mnist(n_samples, square=False, conv=False)
        x_val = x_train_S[N:]
        y_val = y_train_S[N:]
        x_train = x_train_S[:N]
        y_train = y_train_S[:N]
        # Add output noise to important class
        R = int(corrupt_prop*N)
        r_ind = random.sample(range(0,N),R)
        r_class = np.random.randint(0,num_classes,R)
        y_train[r_ind] = I[r_class]
        ## General network structure
        Early_Stop_acc = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

        def network():
            model = Sequential()
            model.add(Dense(units, activation='relu', input_shape=x_train.shape[1:],W_regularizer=l2(reg)))
            model.add(Lambda(lambda x: K.dropout(x,level=dropout)))
            model.add(Dense(num_classes, activation='softmax',W_regularizer=l2(reg)))
            return model

        # ### Standard
        weights_file_std = '../models/mnist/mnist_weights_std_low.h5'
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
                          callbacks=[Early_Stop_acc,model_checkpoint])
        std_model.load_weights(weights_file_std) 

        # ### Weighted
        weights_file_we = '../models/mnist/mnist_weights_we_low.h5'
        model_checkpoint =  keras.callbacks.ModelCheckpoint(weights_file_we, monitor='val_loss', save_best_only=True,
                                           save_weights_only=True, mode='auto')
        we_model = network()
        we_model.compile(loss=weighted_categorical_crossentropy(class_weight),
                          optimizer='adam',
                          metrics=['accuracy'])
        history_we = we_model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=0,
                          validation_data=[x_val,y_val],
                          class_weight = class_weight, 
                          callbacks=[Early_Stop_acc,model_checkpoint])
        we_model.load_weights(weights_file_we) 

        # ### Loss-calibrated
        weights_file_lc = '../models/mnist/mnist_weights_lc_low.h5'
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

        T = 10
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
        print('Accuracy on optimal decision: ', acc_std)
        print('Expected loss: ', loss_std)

        #######
        # WEI #
        #######
        yt_hat_we = np.array([we_model.predict([x_test]) for _ in range(T)])
        MC_pred_we = np.mean(yt_hat_we, 0)
        H_x_test_we = lcbnn.optimal_h(yt_hat_we,loss_mat) 
        acc_we = accuracy_score(np.argmax(y_test,1),np.argmax(H_x_test_we,1))
        loss_we = np.mean(lcbnn.loss_np(y_test,H_x_test_we,loss_mat))

        print('\nWeighted:\n')
        print('Accuracy on optimal decision: ', acc_we)
        print('Expected loss: ', loss_we)

        #######
        # L-C #
        #######
        yt_hat_lc = np.array([lc_model.predict([x_test, H_x_test]) for _ in range(T)])
        MC_pred_lc = np.mean(yt_hat_lc, 0)
        H_x_test_lc = lcbnn.optimal_h(yt_hat_lc,loss_mat) 
        acc_lc = accuracy_score(np.argmax(y_test,1),np.argmax(H_x_test_lc,1))
        loss_lc = np.mean(lcbnn.loss_np(y_test,H_x_test_lc,loss_mat))
        print('\nLoss-cal:\n')
        print('Accuracy on optimal decision: ', acc_lc)
        print('Expected loss: ', loss_lc)

        results_array[seed,u,0] = [acc_std,loss_std]
        results_array[seed,u,1] = [acc_we,loss_we]
        results_array[seed,u,2] = [acc_lc,loss_lc]
        pickle.dump( results_array, open( "results_mnist/"+save_name+".p", "wb" ) )

    u+=1

config_file = {'N_seeds':N_seeds, 'batch_size':batch_size, 'epochs':epochs, 'n_samples':n_samples, 'digits':digits,
               'digit_weight':digit_weight, 'loss_mat':loss_mat, 'Bound':M, 'class_weight':class_weight, 'label_noise':corrupt_prop, 
               'units_list':units_list, 'dropout':dropout, 'tau':tau, 'lengthscale':lengthscale, 'reg':reg}
       
pickle.dump( config_file, open( "results_mnist/"+save_name+"_config.p", "wb" ) )
