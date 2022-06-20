from models.normalization import *
from models.activations import *
from models.ops import *
import tensorflow as tf
import numpy as np

display = True


def encoder_resnet_contrastive_6(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [64, 128, 256, 512, 1024, 2048]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_2(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = convolutional(inputs=net, output_channels=16, filter_size=7, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='intital_layer', display=True)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, 
								 regularizer=regularizer, activation=activation)

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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_1(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, 
								 regularizer=regularizer, activation=activation)

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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_3(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = convolutional(inputs=net, output_channels=32, filter_size=7, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='intital_layer', display=True)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# ResBlock.
			net = residual_block(inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, 
								 regularizer=regularizer, activation=activation)

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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_4(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = convolutional(inputs=net, output_channels=32, filter_size=7, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='intital_layer', display=True)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_5(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = convolutional(inputs=net, output_channels=32, filter_size=7, stride=2, padding='SAME', conv_type='convolutional', spectral=spectral, init=init, regularizer=regularizer, scope='intital_layer', display=True)

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope='%sa' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope='%sb' % layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			

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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init, regularizer=regularizer, activation=activation)
			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_7(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	representation = list()
	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net, style_resnet = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, 
											   init=init, regularizer=regularizer, activation=activation, latent_dim=z_dim, style_extract_f=True)
			representation.append(style_resnet)

			# Attention layer. 
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			# Style extraction.			
			style_conv = style_extract_2(inputs=net, latent_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			representation.append(style_conv)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Feature space extraction
		conv_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2,2])
		conv_space = tf.layers.flatten(inputs=conv_space)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				
		representation.append(z)

	representation = tf.concat(representation, axis=1)
	print('Representation Layer:', representation.shape)

	print()
	return h, z, representation



def encoder_resnet_contrastive_SimSiam(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
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
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
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
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		h = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: h = normalization(inputs=h, training=is_train)
		net = activation(h)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_SwAV(images, z_dim, prototype_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
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
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)
			
			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)
		
		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		# if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=int(channels[-1]/2), spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		# if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')				

		# SwAV paper: Xnt is mapped to a vector representation by a non-linear mapping. Later projected ot a unit sphere.
		z_norm = tf.math.l2_normalize(z, axis=1, name='projection')

		prototype = dense(inputs=z_norm, out_dim=prototype_dim, use_bias=False, spectral=False, init=init, regularizer=regularizer, scope='prototypes')

	print()
	return h, z, z_norm, prototype


def byol_predictor(z_rep, z_dim, h_dim, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='encoder_predictor'):
	net = z_rep
	if display:
		print('PREDICTOR ENCODER INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)				
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Q Prediction.
		q_pred = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='q_pred')				

	print()
	return q_pred



def relational_module(aggregated_representations, h_dim, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='relational_module'):
	net = aggregated_representations
	if display:
		print('RELATIONAL REASONING MODULE INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):
		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)				
		if normalization is not None: 
			net = normalization(inputs=net, training=is_train)
		net = activation(net)
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=2)				

	return logits


