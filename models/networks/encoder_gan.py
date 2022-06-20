from models.normalization import *
from models.activations import *
from models.ops import *

import tensorflow as tf
import numpy as np


display = True

def encoder_resnet_instnorm(images, latent_dim, layers, spectral, activation, reuse, is_train, init='xavier', regularizer=None, normalization=instance_norm, attention=None, down='downscale', name='encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	_, height, width, _ = images.shape.as_list()
	with tf.variable_scope(name, reuse=reuse):

		layer = 0
		net = convolutional(inputs=net, output_channels=channels[layer], filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope=layer)
		# Style extraction.			
		styles = style_extract(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		for layer in range(layers):
			# ResBlock.
			net, style = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, style_extract_f=True, latent_dim=latent_dim, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, 
								 init=init, regularizer=regularizer, activation=activation)
			styles += style

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention: 
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			layer_channel = layer+1
			if layer == layers - 1:
				layer_channel = -2
			net = convolutional(inputs=net, output_channels=channels[layer_channel], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
			# Style extraction.			
			style = style_extract(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
			styles += style
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		net = activation(net)

		# Dense
		style = dense(inputs=net, out_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)
		styles += style		

	print()
	return styles

def encoder_resnet_incr(images, latent_dim, layers, spectral, activation, reuse, is_train, init='xavier', regularizer=None, normalization=instance_norm, attention=None, down='downscale', name='encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	_, height, width, _ = images.shape.as_list()
	with tf.variable_scope(name, reuse=reuse):

		layer = 0
		net = convolutional(inputs=net, output_channels=channels[layer], filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope=layer)
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, 
								 init=init, regularizer=regularizer, activation=activation)

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention: 
				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			layer_channel = layer+1
			if layer == layers - 1:
				layer_channel = -2
			net = convolutional(inputs=net, output_channels=channels[layer_channel], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		net = activation(net)

		# Dense
		style = dense(inputs=net, out_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)

	print()
	return style


def encoder_resnet(images, z_dim, layers, spectral, activation, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('ENCODER INFORMATION:', name)
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers+1):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=True, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention: net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=True, scope=layer)
			net = activation(net)
			
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# Dense.		
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: net = normalization(inputs=net, training=True)
		net = activation(net)

		# Dense
		w_latent = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)		

	print()
	return w_latent


# def encoder_resnet_incr(images, z_dim, layers, spectral, activation, reuse, is_train, init='xavier', regularizer=None, normalization=None, attention=None, stack_layers=False, concat_img=False, 
# 						down='downscale', name='encoder'):
# 	out_stack_layers = list()
# 	net = images
# 	channels = [32, 64, 128, 256, 512, 1024]
# 	if display:
# 		print('ENCODER INFORMATION:')
# 		print('Channels: ', channels[:layers])
# 		print('Normalization: ', normalization)
# 		print('Activation: ', activation)
# 		print('Attention:  ', attention)
# 		print()

# 	_, height, width, _ = images.shape.as_list()
# 	with tf.variable_scope(name, reuse=reuse):

# 		layer = 0
# 		net = convolutional(inputs=net, output_channels=channels[layer], filter_size=3, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope=layer)

# 		for layer in range(layers):
# 			# ResBlock.
# 			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)

# 			if concat_img and layer != 0:
# 				down_sample = tf.image.resize_images(images=images, size=(int(height/(2**layer)),int(width/(2**layer))), method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
# 				print('down_sample', down_sample.shape)
# 				print('net', net.shape)
# 				net = tf.concat([net, down_sample], axis=-1)
# 				print('net', net.shape)

# 			# Attention layer. 
# 			if attention is not None and net.shape.as_list()[1]==attention: 
# 				net = attention_block(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
# 			if stack_layers:
# 				print('Adding layer output to stack layer output.')
# 				out_stack_layers.append(net)
			
# 			# Down.
# 			layer_channel = layer+1
# 			if layer == layers - 1:
# 				layer_channel = -2
# 			net = convolutional(inputs=net, output_channels=channels[layer_channel], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer+1)
# 			if normalization is not None: net = normalization(inputs=net, training=is_train)
# 			net = activation(net)
		
# 		if stack_layers:
# 				print('Adding layer output to stack layer output.')
# 				out_stack_layers.append(net)

# 		if concat_img and layer != 0:
# 			down_sample = tf.image.resize_images(images=images, size=(int(height/(2**(layer+1))),int(width/(2**(layer+1)))), method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
# 			print('down_sample', down_sample.shape)
# 			print('net', net.shape)
# 			net = tf.concat([net, down_sample], axis=-1)
# 			print('net', net.shape)

# 		# Flatten.
# 		net = tf.layers.flatten(inputs=net)

# 		# shape = int(np.product(net.shape.as_list()[1:3])/2)
# 		# # # Dense.
# 		# net = dense(inputs=net, out_dim=shape, spectral=spectral, init=init, regularizer=regularizer, scope=1)				
# 		# if normalization is not None: net = normalization(inputs=net, training=True)
# 		# net = activation(net)

# 		# Dense.
# 		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=2)				
# 		if normalization is not None: net = normalization(inputs=net, training=is_train)
# 		net = activation(net)

# 		# Dense
# 		w_latent = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=3)	

# 	print()
# 	if stack_layers:
# 		return w_latent, out_stack_layers
# 	return w_latent


def decoder_nuance(n_input, z_dim, reuse, is_train, spectral, activation, normalization, init='xavier', regularizer=None, name='decoder_nuance_prediction'):
	
	if display:
		print('DECODER NETWORK INFORMATION:', name)
		print('Normalization: ', normalization)
		print('Activation:    ', activation)
		print()

	net = n_input
	with tf.variable_scope(name, reuse=reuse):
		net = residual_block_dense(inputs=net, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=1)
		net = dense(inputs=net, out_dim=int(z_dim/2), spectral=spectral, init=init, regularizer=regularizer, scope=2)
		net = activation(net)
		net = residual_block_dense(inputs=net, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, activation=activation, init=init, regularizer=regularizer, scope=3)
		nuance_logits = dense(inputs=net, out_dim=3, spectral=spectral, init=init, regularizer=regularizer, scope=4)
	print()
	return nuance_logits	

