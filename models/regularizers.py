import tensorflow as tf
import numpy as np

def orthogonal_reg(scale):

	def ortho_reg(w):

		if len(w.shape.as_list()) > 2:
			filter_size, filter_size, input_channels, output_channels = w.shape.as_list()
			w_reshape = tf.reshape(w, (-1, output_channels))
			dim = output_channels
		else:
			output_dim, input_dim = w.shape.as_list()
			dim = input_dim
			w_reshape = w

		identity = tf.eye(dim)

		wt_w = tf.matmul(a=w_reshape, b=w_reshape, transpose_a=True)
		term = tf.multiply(wt_w, (tf.ones_like(identity)-identity))

		reg = 2*tf.nn.l2_loss(term)

		return scale*reg

	return ortho_reg

def perceptual_path_length(model, path_length_decay=0.01, path_length_weight=2.0):
	path_length_noise = tf.random_normal(tf.shape(model.fake_images))/np.sqrt(model.image_height*model.image_width)
	path_length_grad = tf.gradients(tf.reduce_sum(model.fake_images * path_length_noise), model.w_latent)

	# Shape: (?, 200, 6)
	path_length_grad_s2 = tf.square(path_length_grad)
	# Shape: (1, ?, 200, 6)
	path_length_1 = tf.reduce_sum(path_length_grad_s2, axis=2)
	# Shape: (1, ?, 6)
	path_length = tf.sqrt(tf.reduce_mean(path_length_1, axis=1))
	# Shape: (1, 6)

	with tf.control_dependencies(None):
		path_length_mean_var = tf.Variable(name='path_length_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
	path_length_mean = path_length_mean_var + path_length_decay*(tf.reduce_mean(path_length)-path_length_mean_var)
	path_length_update = tf.assign(path_length_mean_var, path_length_mean)

	with tf.control_dependencies([path_length_update]):
		path_length_penalty = tf.square(path_length-path_length_mean)

	# print('path_length_mean:', path_length_mean.shape)
	regularization_term = tf.reduce_mean(path_length_penalty)*path_length_weight
	# print('Final variable:', regularization_term.shape)

	return regularization_term


def l2_reg(scale):
	return tf.contrib.layers.l2_regularizer(scale)


def l1_reg(scale):
	return tf.contrib.layers.l1_regularizer(scale)