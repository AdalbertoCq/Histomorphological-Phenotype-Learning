from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.evaluation import *
from models.optimizer import *
from models.loss import *
from models.ops import *

import tensorflow as tf
import numpy as np


display = True

# Feature Extractor Network 20x.
def feature_extractor_20x(inputs, z_dim, regularizer_scale, use, reuse, scope):
	print('Feature Extractor Network 20x:', inputs.shape[-1], 'Dimensions')
	interm = inputs
	if use:
		with tf.variable_scope('feature_extractor_20x_%s' % scope, reuse=reuse):	

			interm = tf.reshape(interm, (-1, z_dim))	
			net = dense(inputs=interm, out_dim=int(z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net = ReLU(net)
			net = dense(inputs=net,    out_dim=int(z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			interm = ReLU(net)
			interm = tf.reshape(interm, (-1, 16, z_dim))
	print()
	return interm

# Feature Extractor Network 10x.
def feature_extractor_10x(inputs, z_dim, regularizer_scale, use, reuse, scope):
	print('Feature Extractor Network 10x:', inputs.shape[-1], 'Dimensions')
	interm = inputs
	if use:
		with tf.variable_scope('feature_extractor_10x_%s' % scope, reuse=reuse):	

			interm = tf.reshape(interm, (-1, z_dim))	
			net = dense(inputs=interm, out_dim=int(z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net = ReLU(net)
			net = dense(inputs=net,    out_dim=int(z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			interm = ReLU(net)
			interm = tf.reshape(interm, (-1, 4, z_dim))
	print()
	return interm

# Feature Extractor Network 5x
def feature_extractor_5x(inputs, z_dim, regularizer_scale, use, reuse, scope):
	print('Feature Extractor Network 5x:', inputs.shape[-1], 'Dimensions')
	interm = inputs
	if use:
		with tf.variable_scope('feature_extractor_5x_%s' % scope, reuse=reuse):		
			net = dense(inputs=inputs, out_dim=int(z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net = ReLU(net)
			net = dense(inputs=net,    out_dim=int(z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			interm = ReLU(net)
		print()
	return interm

# Attention Network 20x.
def attention_20x(inputs, z_dim, att_dim, regularizer_scale, reuse, scope, use_gated=True):
	print('Attention Network 20x:', inputs.shape[-1], 'Dimensions')
	with tf.variable_scope('attention_20x_%s' % scope, reuse=reuse):	

		net1 = tf.reshape(inputs, (-1, z_dim))	
		net1 = dense(inputs=net1, out_dim=att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net1 = tanh(net1)

		if use_gated:
			# GatedAttention.  
			net2 = dense(inputs=inputs, out_dim=att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net2 = sigmoid(net2)
			net = net1*net2
		else:
			net = net1

		# Get weights.
		net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net_reshape = tf.reshape(net, (-1, 16, 1))
		weights = tf.nn.softmax(net_reshape, axis=1)
	print()
	return weights

# Attention Network 10x.
def attention_10x(inputs, z_dim, att_dim, regularizer_scale, reuse, scope, use_gated=True):
	print('Attention Network 10x:', inputs.shape[-1], 'Dimensions')
	with tf.variable_scope('attention_10x_%s' % scope, reuse=reuse):	

		net1 = tf.reshape(inputs, (-1, z_dim))	
		net1 = dense(inputs=net1, out_dim=att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net1 = tanh(net1)

		if use_gated:
			# GatedAttention.  
			net2 = dense(inputs=inputs, out_dim=att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net2 = sigmoid(net2)
			net = net1*net2
		else:
			net = net1

		# Get weights.
		net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net_reshape = tf.reshape(net, (-1, 4, 1))
		weights = tf.nn.softmax(net_reshape, axis=1)
	print()
	return weights

# Aggregate 20x representations.
def aggregate_20x_representations(interm, weights, reuse, scope):
	print('Aggregate Network 20x:', interm.shape[-1], 'Dimensions')
	with tf.variable_scope('aggregate_20x_%s' % scope, reuse=reuse):	
		weighted_rep   = interm*weights
		aggregated_rep = tf.reduce_sum(weighted_rep, axis=1)
	return aggregated_rep

# Aggregate 10x representations.
def aggregate_10x_representations(interm, weights, reuse, scope):
	print('Aggregate Network 10x:', interm.shape[-1], 'Dimensions')
	with tf.variable_scope('aggregate_10x_%s' % scope, reuse=reuse):	
		weighted_rep   = interm*weights
		aggregated_rep = tf.reduce_sum(weighted_rep, axis=1)
	return aggregated_rep

# Feature combination for concatenated vector of 5x/10x/20x
def feature_extractor_comb(inputs, z_dim, regularizer_scale, use, reuse, scope):
	print('Feature Extractor Network All Magnifications:', inputs.shape[-1], 'Dimensions')
	interm = inputs
	if use:
		with tf.variable_scope('feature_extractor_comb_%s' % scope, reuse=reuse):		
			net = dense(inputs=inputs, out_dim=int(z_dim)*3, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net = ReLU(net)
			net = dense(inputs=net,    out_dim=int(z_dim)*3, scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			interm = ReLU(net)
		print()
	return interm

# Attention Network.
def attention(inputs, z_dim, att_dim, regularizer_scale, reuse, scope, use_gated=True):
	print('Attention Network All Magnifications:', inputs.shape[-1], 'Dimensions')
	with tf.variable_scope('attention_%s' % scope, reuse=reuse):	

		#
		net1 = dense(inputs=inputs, out_dim=att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net1 = tanh(net1)

		if use_gated:
			# GatedAttention.  
			net2 = dense(inputs=inputs, out_dim=att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
			net2 = sigmoid(net2)
			net = net1*net2
		else:
			net = net1

		# Get weights.
		net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(regularizer_scale), display=True)
		net = tf.transpose(net)
		weights = tf.nn.softmax(net)
		weights = tf.transpose(weights)
	print()

	return weights

# Patient aggregation for representations and weights. 
def patient_aggregation(interm, weights, reuse, scope):
	print('Patient Aggregation Network:', interm.shape[-1], 'Dimensions')
	with tf.variable_scope('patient_aggregation_%s' % scope, reuse=reuse):		
		# Weight each sample.
		patient_rep = tf.reshape(tf.reduce_sum(weights*interm, axis=0), (-1,1))
		patient_rep = tf.transpose(patient_rep)
	print()			
	return patient_rep

def attention_network(model, represenation_input_5x, represenation_input_10x, represenation_input_20x, regularizer_scale, reuse, name='attention_network'):

	with tf.variable_scope(name, reuse=reuse):    

		# Feature Extractions.
		model.interm_5x  = feature_extractor_5x(inputs=represenation_input_5x,   z_dim=model.z_dim, regularizer_scale=regularizer_scale, use=True, reuse=False, scope=1)
		model.interm_10x = feature_extractor_10x(inputs=represenation_input_10x, z_dim=model.z_dim, regularizer_scale=regularizer_scale, use=True, reuse=False, scope=1)
		model.interm_20x = feature_extractor_20x(inputs=represenation_input_20x, z_dim=model.z_dim, regularizer_scale=regularizer_scale, use=True, reuse=False, scope=1)

		################### Multi-Magnification Attention MIL portion of the model.
		# Attention and aggregation of 20x.
		model.weights_20x   = attention_20x(inputs=model.interm_20x, z_dim=model.z_dim, att_dim=model.att_dim, regularizer_scale=regularizer_scale, use_gated=model.use_gated,  reuse=False, scope=1)
		aggregate_tiles_20x = aggregate_20x_representations(model.interm_20x, model.weights_20x, reuse=False, scope=1)

		# Attention and aggregation of 10x.
		model.weights_10x   = attention_10x(inputs=model.interm_10x, z_dim=model.z_dim, att_dim=model.att_dim, regularizer_scale=regularizer_scale, use_gated=model.use_gated,  reuse=False, scope=1)
		aggregate_tiles_10x = aggregate_10x_representations(model.interm_10x, model.weights_10x, reuse=False, scope=1)

		# Concatenate all magnification representations: 3*z_dim.
		rep_multimag        = tf.concat([model.interm_5x, aggregate_tiles_10x, aggregate_tiles_20x], axis=1)
		rep_multimag        = feature_extractor_comb(inputs=rep_multimag, z_dim=model.z_dim, regularizer_scale=regularizer_scale, use=True, reuse=False, scope=1)
		
		# Attention and Patient Represenation. 
		model.weights       = attention(inputs=rep_multimag, z_dim=model.z_dim, att_dim=model.att_dim, regularizer_scale=regularizer_scale, use_gated=model.use_gated, reuse=False, scope=1)
		patient_rep_ind     = patient_aggregation(interm=rep_multimag, weights=model.weights, reuse=False, scope=1)

	return patient_rep_ind





