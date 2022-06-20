from models.normalization import *
from models.activations import *
from models.ops import *

import tensorflow as tf
import numpy as np


display = True

def discriminator_resnet(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, feature_space_flag=False, name='discriminator', realness=1):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and (net.shape.as_list()[1]==attention):
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
				# net = lambda_network(net, m=attention/2, spectral=spectral, init=init, regularizer=regularizer, scope=layers+layer)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Feature space extraction
		feature_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2,2])
		feature_space = tf.layers.flatten(inputs=feature_space)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		if label is not None: 
			print(label.shape)
			net = dense(inputs=net, out_dim=label.shape[-1], spectral=spectral, init=init, regularizer=regularizer, scope=3)				
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Dense
		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		if label is not None: 
			inner_prod = tf.reduce_sum(net * label, axis=-1, keepdims=True)
			logits = logits_net + inner_prod
			output = sigmoid(logits)
		else:
			logits = logits_net
			output = sigmoid(logits)

	print()
	if feature_space_flag:
		return output, logits, feature_space
	return output, logits


def discriminator_resnet_mask_class(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, name='discriminator', softmax=tf.constant(1.)):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		# Discriminator with conditional projection.
		batch_size, label_dim = label.shape.as_list()
		embedding_size = channels[-1]

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True) 
			net = activation(net)

		# Feature space extraction
		feature_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=1)
		feature_space = tf.layers.flatten(inputs=feature_space)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		net = activation(net)

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=label_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		# class_logits = tf.nn.log_softmax(class_logits)
		# One encoding for label input
		logits = class_logits*label
		logits = tf.reduce_sum(logits, axis=-1)
		output = sigmoid(logits)
		
	print()
	return output, logits, feature_space


def discriminator_resnet_mask_invariant(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, name='discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense Feature Space.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		net = activation(net)

		# Dense Feature Space.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		feature_space = activation(net)

		# Dense.
		net = dense(inputs=feature_space, out_dim=channels[-2], spectral=spectral, init=init, regularizer=regularizer, scope=3)				
		net = activation(net)

		# Dense Classes.
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=4)	
		
		# One encoding for label input
		output = sigmoid(logits)

	print()
	return output, logits, feature_space


def discriminator_resnet_class(images, layers, spectral, activation, reuse, l_dim, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='discriminator'):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		 # Dense.
		feature_space = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		net = activation(feature_space)		

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		output = sigmoid(logits)

		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=4)		
		net = activation(net)	

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=l_dim, spectral=spectral, init=init, regularizer=regularizer, scope=5)		

	print()
	return output, logits, feature_space, class_logits


def discriminator_resnet_class2(images, layers, spectral, activation, reuse, l_dim, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='discriminator'):
	net = images
	# channels = [32, 64, 128, 256, 512, 1024, 2048]
	channels = [32, 64, 128, 256, 512, 1024]

	# New
	layers = layers + 1

	if display:
		print('DISCRIMINATOR INFORMATION:', name)
		print('Total  Channels: ', channels)
		print('Chosen Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
								 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# New
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		feature_space = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		net = activation(feature_space)		

		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=3)		
		output = sigmoid(logits)

		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=4)		
		net = activation(net)	

		# Dense Classes
		class_logits = dense(inputs=net, out_dim=l_dim, spectral=spectral, init=init, regularizer=regularizer, scope=5)			

	print()
	return output, logits, feature_space, class_logits


def discriminator(images, layers, spectral, activation, reuse, normalization=None):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	
	if display:
		print('Discriminator Information.')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()
	with tf.variable_scope('discriminator', reuse=reuse):
		# Padding = 'Same' -> H_new = H_old // Stride

		for layer in range(layers):
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=5, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)
		
		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)
		
		# Dense
		logits = dense(inputs=net, out_dim=1, spectral=spectral, scope=2)				
		output = sigmoid(logits)

	print()
	return output, logits


def discriminator_encoder(enconding, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, name='dis_encoding'):
	net = enconding
	channels = [150, 100, 50, 25, 12]
	# channels = [200, 150, 100, 50, 24]
	if display:
		print('DISCRIMINATOR-ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):
		for layer in range(layers):

			# Residual Dense layer.
			net = residual_block_dense(inputs=net, scope=layer, is_training=True, normalization=normalization, spectral=spectral, activation=activation, init=init, regularizer=regularizer, display=True)

			# Dense layer downsample dim.
			net = dense(inputs=net, out_dim=channels[layer], spectral=spectral, init=init, regularizer=regularizer, scope=layer)				
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)

		# Dense
		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)		
		output = sigmoid(logits_net)

	print()
	return output, logits_net


def discriminator_resnet_contrastive_hrep(h_representations, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='discriminator'):
	batch_size, h_rep_dim = h_representations.get_shape().as_list()
	if display:
		print('H REPRESENTATION DISCRIMINATOR INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	net = h_representations
	with tf.variable_scope(name, reuse=reuse):

		# Dense.
		net = residual_block_dense(inputs=net, scope=1, is_training=is_train, normalization=normalization, spectral=spectral, activation=activation, init=init, regularizer=regularizer, display=True)
	
		net = dense(inputs=net, out_dim=int(h_rep_dim/2), spectral=spectral, init=init, regularizer=regularizer, scope=2)
		net = activation(net)
		net = residual_block_dense(inputs=net, scope=2, is_training=is_train, normalization=normalization, spectral=spectral, activation=activation, init=init, regularizer=regularizer, display=True)
		
		net = dense(inputs=net, out_dim=100, spectral=spectral, init=init, regularizer=regularizer, scope=3)				
		net = activation(net)
		net = residual_block_dense(inputs=net, scope=3, is_training=is_train, normalization=normalization, spectral=spectral, activation=activation, init=init, regularizer=regularizer, display=True)

		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope='logits')				
		output = sigmoid(logits_net)

	print()
	return output, logits_net


def discriminator_resnet_contrastive_whole(images, z_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, 
								 regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Feature space extraction
		conv_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2,2])
		conv_space = tf.layers.flatten(inputs=conv_space)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		contrastive_net = dense(inputs=h, out_dim=int((channels[-1])/2), spectral=spectral, init=init, regularizer=regularizer, scope='contrastive_1')				
		if normalization is not None: contrastive_net = normalization(inputs=contrastive_net, training=is_train)
		contrastive_net = activation(contrastive_net)
		
		z = dense(inputs=contrastive_net, out_dim=128, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

		logits_net = dense(inputs=z, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope='logits')				
		output = sigmoid(logits_net)

	print()
	return output, logits_net, conv_space, h, z


def discriminator_resnet_contrastive(images, z_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_discriminator'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, 
								 regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Feature space extraction
		conv_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2,2])
		conv_space = tf.layers.flatten(inputs=conv_space)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=int((channels[-1])/2), spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)
		
		# Z Representation Layer.
		z = dense(inputs=net, out_dim=128, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				
		net = activation(net)

	# Unused part, legacy.
	with tf.variable_scope('unused', reuse=reuse):

		logits_net = dense(inputs=z, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope='Adversarial')				
		output = sigmoid(logits_net)

	print()
	return output, logits_net, conv_space, h, z


def discriminator_resnet_lambda(images, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', label=None, feature_space_flag=False, name='discriminator', realness=1):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('DISCRIMINATOR INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):

			if net.shape[1]==224:
				# ResBlock.
				net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, 
									 spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			else:
				# ResBlock.
				net = lambda_residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, spectral=spectral, init=init, 
											regularizer=regularizer, activation=activation)
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
		
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		if label is not None: 
			print(label.shape)
			net = dense(inputs=net, out_dim=label.shape[-1], spectral=spectral, init=init, regularizer=regularizer, scope=3)				
			if normalization is not None: net = normalization(inputs=net, training=True)
			net = activation(net)
			
		# Dense
		logits_net = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=2)		
		if label is not None: 
			inner_prod = tf.reduce_sum(net * label, axis=-1, keepdims=True)
			logits = logits_net + inner_prod
			output = sigmoid(logits)
		else:
			logits = logits_net
			output = sigmoid(logits)

	print()
	if feature_space_flag:
		return output, logits, None
	return output, logits


