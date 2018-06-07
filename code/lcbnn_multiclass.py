from keras import backend as K
import numpy as np
import tensorflow as tf


import time
##
# Loss functions
##

def loss_np(y_true=[],y_pred=None,loss_mat=None):
	"""
	numpy function to calculate loss from the loss matrix:
	Inputs:
		y_true: true values (N,D)
		y_pred: predicted values (N,D)
		loss_mat: matrix of loss values for selecting outputs (D,D)
		net_loss: True -> same as loss_K, False -> used in optimal decision
	"""
	N,D = np.shape(y_pred)
	if len(y_true) != 0:
		A = np.expand_dims(np.matmul(y_pred,loss_mat),1)
		B = np.expand_dims(y_true.T,0)
		B = np.matmul(A,B.T)
		L = B.reshape((N,))
	else: # For inferring opimal H you treat y_pred as the true values
		A = np.expand_dims(np.matmul(loss_mat,y_pred.T),0)
		R_d = np.zeros_like(y_pred)
		for d in range(D):
			Z = np.zeros_like(y_pred)
			Z[:,d] = 1
			Z = np.expand_dims(Z,1)
			B = np.matmul(Z,A.T)#     Matrix mul for D=12, N = 60,000 is 5000 x slower than for loop
			R_d[:,d] = B.reshape((N,)) 
			L = R_d 
	return L


def loss_K(loss_mat, Segmentation = False):
	"""
	TF function to calculate loss from the loss matrix:
	Inputs:
		y_true: true values (N,D)
		y_pred: predicted values (N,D)
		loss_mat: matrix of loss values for selecting outputs (D,D)
	"""
	
	loss_mat = tf.constant(loss_mat,dtype=tf.float32)
	
	if Segmentation:
		def loss(y_true,y_pred):
			shape = tf.shape(y_true)
			y_true = tf.reshape(y_true,(-1,shape[2])) # Turn each pixel into a data point
			y_pred = tf.reshape(y_pred,(-1,shape[2]))
			A = tf.expand_dims(tf.matmul(y_pred, loss_mat,name='matmul1'),1) # Select rows with pred
			B = tf.expand_dims(y_true,-1)
			L = tf.reshape(tf.matmul(A, B,name='matmul2'),(-1,1))
			return tf.reshape(L,(-1,shape[1]))
	else:
		def loss(y_true,y_pred):
			A = tf.matmul(y_pred, loss_mat) # Select rows with pred
			B = tf.matmul(A, tf.transpose(y_true))
			L = tf.diag_part(B)
			return L
	return loss


def cal_loss(loss,M,H_x,Segmentation=False):
	"""
	Loss function for lcbnn
	Inputs:
		loss: TF function loss(y_true,y_pred)
		M: Bound on loss (to convert to utility)
		H_x: Optimal label for each x
	Outputs:
		dyn_loss: dynamic loss function for network
	"""
	def dyn_loss(y_true,y_pred):
		# L = K.mean(K.categorical_crossentropy(y_true, y_pred) - K.mean(K.log(M-loss(y_pred,H_x)),axis=-1),axis=0)
		L = K.mean(K.categorical_crossentropy(y_true, y_pred) - K.log(M-loss(y_pred,H_x)),axis=0)
		if Segmentation == True:
			L = K.mean(K.categorical_crossentropy(y_true, y_pred) - K.log(M-loss(y_pred,H_x)),axis=-1)

		return L
	return dyn_loss

##
# Optimal H
##

def optimal_h(y_pred_samples, loss_mat, return_risk = False):
	"""
	Calculate the optimal_h
	Inputs:
		y_pred_samples: predicted values (N,D)
		loss_mat: matrix of loss values for selecting outputs (D,D)
	"""
	T,N,D = np.shape(y_pred_samples)
	R_t = np.zeros((N,D)) # Risk
	for t in range(T):
		R_t += loss_np(y_pred = y_pred_samples[t], loss_mat = loss_mat)
	I = np.eye(D)
	H_x = I[np.argmin(R_t,axis=-1)]
	if not return_risk:
		return H_x
	else:
		return H_x, (1./float(T)) * R_t

