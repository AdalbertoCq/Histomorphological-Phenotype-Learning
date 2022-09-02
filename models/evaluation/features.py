from models.evaluation.latent_space import *
from data_manipulation.utils import *
from models.utils import *

import tensorflow as tf
import numpy as np
import matplotlib
import random
import shutil
import h5py
import os


# Gather real samples from train and test sets for FID and other scores.
def real_samples(data, data_output_path, num_samples=10000, save_img=False):
	path = os.path.join(data_output_path, 'results')
	path = os.path.join(path, 'real')
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % (data.patch_h, data.patch_w, data.n_channels)
	path = os.path.join(path, res)
	if not os.path.isdir(path):
		os.makedirs(path)

	batch_size = data.batch_size
	images_shape =  [num_samples] + [data.patch_h, data.patch_w, data.n_channels]

	hdf5_sets_path = list()
	dataset_sets_path = [data.hdf5_train, data.hdf5_validation, data.hdf5_test]
	dataset_sets = [data.training, data.validation, data.test]
	for i_set, set_path in enumerate(dataset_sets_path):
		set_data = dataset_sets[i_set]
		if set_data is None:
			continue
		type_set = set_path.split('_')[-1]
		type_set = type_set.split('.')[0]	
		img_path = os.path.join(path, 'img_%s' % type_set)
		if not os.path.isdir(img_path):
			os.makedirs(img_path)

		hdf5_path_current = os.path.join(path, 'hdf5_%s_%s_images_%s_real.h5' % (data.dataset, data.marker, type_set))
		hdf5_sets_path.append(hdf5_path_current)

		if os.path.isfile(hdf5_path_current):
			print('H5 File Image %s already created.' % type_set)
			print('\tFile:', hdf5_path_current)
		else:
			print('H5 File Image %s.' % type_set)
			print('\tFile:', hdf5_path_current)

			hdf5_img_real_file = h5py.File(hdf5_path_current, mode='w')
			img_storage = hdf5_img_real_file.create_dataset(name='images', shape=images_shape, dtype=np.float32)
			label_flag = False
			if len(set_data.labels) > 0:
				print('Labels present, carring them over...')
				label_flag = True
				label_shape =  [num_samples] + [set_data.labels.shape[-1]]
				label_storage = hdf5_img_real_file.create_dataset(name='labels', shape=label_shape, dtype=np.float32)

			possible_samples = len(set_data.images)
			random_samples = list(range(possible_samples))
			random.shuffle(random_samples)

			ind = 0
			for index in random_samples[:num_samples]:
				img_storage[ind] = set_data.images[index]
				if label_flag: label_storage[ind] = set_data.labels[index]
				if save_img:
					plt.imsave('%s/real_%s_%s.png' % (img_path, type_set, ind), set_data.images[index])
				ind += 1
			print('\tNumber of samples:', ind)

	return hdf5_sets_path


# Extract Inception-V1 features from images in HDF5.
def inception_tf_feature_activations(hdf5s, input_shape, batch_size):
	import tensorflow.contrib.gan as tfgan

	images_input = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='images')
	images = 2*images_input
	images -= 1
	images = tf.image.resize_bilinear(images, [299, 299])
	out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')

	hdf5s_features = list()
	with tf.Session() as sess:
		for hdf5_path in hdf5s:
			# Name handling.
			hdf5_feature_path = hdf5_path.split('.h5')[0] + '_features.h5'
			hdf5s_features.append(hdf5_feature_path)
			if os.path.isfile(hdf5_feature_path):
				print('H5 File Feature already created.')
				print('\tFile:', hdf5_feature_path)
				continue
			hdf5_img_file = h5py.File(hdf5_path, mode='r')
			flag_images = False
			hdf5_features_file = h5py.File(hdf5_feature_path, mode='w')
			for key in list(hdf5_img_file.keys()):
				if 'images' in key:
					flag_images = True
					storage_name = key.replace('images', 'features')
					images_storage = hdf5_img_file[key]
					
					num_samples = images_storage.shape[0]
					batches = int(num_samples/batch_size)
					features_shape = (num_samples, 2048)
					features_storage = hdf5_features_file.create_dataset(name=storage_name, shape=features_shape, dtype=np.float32)

					print('Starting features extraction...')
					print('\tImage File:', hdf5_path)
					print('\t\tImage type:', key)
					ind = 0
					for batch_num in range(batches):
						batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
						if np.amax(batch_images) > 1.0:
							batch_images = batch_images/255.
						activations = sess.run(out_incept_v3, {images_input: batch_images})
						features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
						ind += batch_size
					print('\tFeature File:', hdf5_feature_path)
					print('\tNumber of samples:', ind)
			if not flag_images:
				os.remove(hdf5_features_file)	
	return hdf5s_features


# Generate random samples from a model, it also dumps a sprite image width them.
def generate_samples_epoch(session, model, data, epoch, data_out_path, num_samples=10000, batch_size=50, one_hot_encoder=None, report=False):
	# Handle directories and copies.
	results_path = os.path.join(data_out_path, 'results')
	epoch_path = os.path.join(results_path, 'epoch_%s' % epoch)
	check_epoch_path = os.path.join(epoch_path, 'checkpoints')
	checkpoint_path = os.path.join(results_path, '../checkpoints')
	os.makedirs(epoch_path)
	shutil.copytree(checkpoint_path, check_epoch_path)

	# Setup HDF5 file.
	hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_generated_images.h5' % epoch)	
	hdf5_file = h5py.File(hdf5_path, mode='w')
	latent_shape = [num_samples, model.z_dim]
	if one_hot_encoder is not None:
		lemb_shape = [num_samples] + [model.embedding_size]
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		label_storage = hdf5_file.create_dataset(name='labels', shape=[num_samples, 1], dtype=np.int32)
		lemb_storage = hdf5_file.create_dataset(name='label_emb', shape=lemb_shape, dtype=np.float32)
		latent_shape = [num_samples, model.complete_z_dim]		
	img_storage = hdf5_file.create_dataset(name='images', shape=[num_samples, data.patch_h, data.patch_w, data.n_channels], dtype=np.float32)
	w_storage   = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)

	# Image generation.
	ind = 0
	while ind < num_samples:
		z_batch_1 = np.random.normal(size=(batch_size, model.z_dim))

		# Mapping to latent space W.
		if one_hot_encoder is not None:
			z_label_batch_int = np.random.choice(model.labels_unique[:,0], p=model.categorical, size=(batch_size,1))
			batch_labels = one_hot_encoder.transform(z_label_batch_int)
			feed_dict = {model.z_input_1: z_batch_1, model.z_labels:batch_labels}
			w_latent_batch, l_embedding_batch = session.run([model.w_latent_out, model.label_emb_gen], feed_dict=feed_dict)
		else:
			feed_dict = {model.z_input_1: z_batch_1}
			w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]

		# Generate image from latent space W.
		w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
		feed_dict = {model.w_latent_in:w_latent_in}
		gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

		for i in range(batch_size):
			if ind == num_samples:
				break
			img_storage[ind] = gen_img_batch[i, :, :, :]
			w_storage[ind] = w_latent_batch[i, :]
			if one_hot_encoder is not None:
				z_storage[ind] = z_batch_1[i, :]
				label_storage[ind] = z_label_batch_int[i,:].astype(np.int32)
				lemb_storage[ind] = l_embedding_batch[i, :]
			ind += 1
	if report:
		label_samples = None
		if one_hot_encoder is not None:
			label_samples = label_storage
		try:
			report_progress_latent(epoch=epoch, w_samples=w_storage, img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], label_samples=label_samples)
		except:
			print('Issue printing latent space images. Epoch', epoch)


# Generate sampeles from PathologyGAN, no encoder.
def generate_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50, save_img=False):
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	ds_o = data.training
	if ds_o is None:
		ds_o = data.test
	if ds_o is None:
		ds_o = data.validation
	for batch_images, batch_labels in ds_o:
		break
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.test.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		if 'PathologyGAN' in model.model_name:
			w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:
				
				# Image and latent generation for PathologyGAN.
				if model.model_name == 'BigGAN':
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input:z_latent_batch}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Image and latent generation for StylePathologyGAN.
				else:
					z_latent_batch = np.random.normal(size=(batches, model.z_dim))
					feed_dict = {model.z_input_1: z_latent_batch}
					w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
					w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
					feed_dict = {model.w_latent_in:w_latent_in}
					gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					img_storage[ind] = gen_img_batch[i, :, :, :]
					z_storage[ind] = z_latent_batch[i, :]
					if 'PathologyGAN' in model.model_name:
						w_storage[ind] = w_latent_batch[i, :]
					if save_img:
						if gen_img_batch.shape[-1] == 1:
							plt.imshow(gen_img_batch[i, :, :, 0], cmap='gray')
							plt.savefig('%s/gen_%s.png' % (img_path, ind))
						else:
							plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


# Generate sampeles from PathologyGAN, no encoder.
def generate_samples_from_checkpoint_conditional(model, data, data_out_path, checkpoint, k, num_samples=5000, batches=50, save_img=False):
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s_zdim%s_emb%s_nclust%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim, model.embedding_size, k)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	ds_o = data.training
	if ds_o is None:
		ds_o = data.test
	if ds_o is None:
		ds_o = data.validation
	for batch_images, batch_labels in ds_o:
		break

	from sklearn.preprocessing import OneHotEncoder
	one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
	labels_unique = np.array(list(range(k))).reshape((-1,1))
	one_hot_encoder.fit(labels_unique)
	
	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + data.training.shape[1:]
		latent_shape = [num_samples] + [model.z_dim]
		lemb_shape = [num_samples] + [model.embedding_size]
		w_shape = [num_samples] + [model.z_dim + model.embedding_size]
		hdf5_file = h5py.File(hdf5_path, mode='w')

		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		label_storage = hdf5_file.create_dataset(name='labels', shape=latent_shape, dtype=np.float32)
		lemb_storage = hdf5_file.create_dataset(name='label_emb', shape=lemb_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=w_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0

			while ind < num_samples:
				
				z_latent_batch = np.random.normal(size=(batches, model.z_dim))
				z_label_batch_int = np.random.randint(k, size=(batches,1))
				z_label_batch = one_hot_encoder.transform(z_label_batch_int)
				# feed_dict = {model.z_input_1: z_latent_batch, model.z_labels_1: z_label_batch}
				# w_latent_batch, l_embedding_batch = session.run([model.w_latent_out, model.label_emb_gen_1], feed_dict=feed_dict)
				feed_dict = {model.z_input_1: z_latent_batch, model.z_labels: z_label_batch}
				w_latent_batch, l_embedding_batch = session.run([model.w_latent_out, model.label_emb_gen], feed_dict=feed_dict)
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])
				
				feed_dict = {model.w_latent_in:w_latent_in, model.real_images:batch_images}
				gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					img_storage[ind] = gen_img_batch[i, :, :, :]
					z_storage[ind] = z_latent_batch[i, :]
					label_storage[ind] = z_label_batch_int[i,:]
					lemb_storage[ind] = l_embedding_batch[i, :]
					w_storage[ind] = w_latent_batch[i, :]
					if save_img:
						if gen_img_batch.shape[-1] == 1:
							plt.imshow(gen_img_batch[i, :, :, 0], cmap='gray')
							plt.savefig('%s/gen_%s.png' % (img_path, ind))
						else:
							plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
						
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


# Generate and encode samples from PathologyGAN, with an encoder.
def generate_encode_samples_from_checkpoint(model, data, data_out_path, checkpoint, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s' % (data.patch_h, data.patch_w, data.n_channels)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'generated_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_images_%s.h5' % (data.dataset, data.marker, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))

	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + [data.patch_h, data.patch_w, data.n_channels]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		z_storage = hdf5_file.create_dataset(name='z_latent', shape=latent_shape, dtype=np.float32)
		# Generated images.
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		# Reconstructed generated images.
		img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
		w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:
				
				# W latent.
				z_latent_batch = np.random.normal(size=(batches, model.z_dim))
				feed_dict = {model.z_input_1: z_latent_batch}
				w_latent_batch = session.run([model.w_latent_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W latent space.
				feed_dict = {model.w_latent_in:w_latent_in}
				gen_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Encode generated images into W' latent space.
				feed_dict = {model.real_images_2:gen_img_batch}
				w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W' latent space.
				feed_dict = {model.w_latent_in:w_latent_prime_in}
				gen_img_prime_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					z_storage[ind] = z_latent_batch[i, :]
					# Generated.
					img_storage[ind] = gen_img_batch[i, :, :, :]
					w_storage[ind] = w_latent_batch[i, :]
					# Reconstructed.
					img_prime_storage[ind] = gen_img_prime_batch[i, :, :, :]
					w_prime_storage[ind] = w_latent_prime_batch[i, :]

					# Saving images
					plt.imsave('%s/gen_%s.png' % (img_path, ind), gen_img_batch[i, :, :, :])
					plt.imsave('%s/gen_recon_%s.png' % (img_path, ind), gen_img_prime_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


# Encode real images and regenerate from PathologyGAN, with an encoder.
def real_encode_eval_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, type_set, num_samples=5000, batches=50):
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'real_images')
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isdir(img_path):
		os.makedirs(img_path)

	if not os.path.isfile(real_hdf5):
		print('Real image H5 file does not exist:', real_hdf5)
		exit()
	real_images = read_hdf5(real_hdf5, 'images')

	hdf5_path = os.path.join(path, 'hdf5_%s_%s_real_%s_images_%s.h5' % (data.dataset, data.marker, type_set, model.model_name))
	
	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))

	if not os.path.isfile(hdf5_path):
		# H5 File specifications and creation.
		img_shape = [num_samples] + [data.patch_h, data.patch_w, data.n_channels]
		latent_shape = [num_samples] + [model.z_dim]
		hdf5_file = h5py.File(hdf5_path, mode='w')
		# Real images.
		img_storage = hdf5_file.create_dataset(name='images', shape=img_shape, dtype=np.float32)
		w_storage = hdf5_file.create_dataset(name='w_latent', shape=latent_shape, dtype=np.float32)
		# Reconstructed generated images.
		img_prime_storage = hdf5_file.create_dataset(name='images_prime', shape=img_shape, dtype=np.float32)
		w_prime_storage = hdf5_file.create_dataset(name='w_latent_prime', shape=latent_shape, dtype=np.float32)
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		saver = tf.train.Saver()
		with tf.Session() as session:

			# Initializer and restoring model.
			session.run(tf.global_variables_initializer())
			saver.restore(session, checkpoint)

			ind = 0
			while ind < num_samples:

				# Real images.
				if (ind + batches) < len(real_images):
					real_img_batch = real_images[ind: ind+batches, :, :, :]/255.
				else:
					real_img_batch = real_images[ind:, :, :, :]/255.

				# Encode real images into W latent space.
				feed_dict = {model.real_images_2:real_img_batch}
				w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Generate images from W latent space.
				feed_dict = {model.w_latent_in:w_latent_in}
				recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

				# Encode reconstructed images into W' latent space.
				feed_dict = {model.real_images_2:recon_img_batch}
				w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]
				w_latent_prime_in = np.tile(w_latent_prime_batch[:,:, np.newaxis], [1, 1, model.layers+1])

				# Fill in storage for latent and image.
				for i in range(batches):
					if ind == num_samples:
						break
					# Real Images.
					img_storage[ind] = real_img_batch[i, :, :, :]
					w_storage[ind] = w_latent_batch[i, :]
					
					# Reconstructed images.
					img_prime_storage[ind] = recon_img_batch[i, :, :, :]
					w_prime_storage[ind] = w_latent_prime_batch[i, :]

					# Saving images
					plt.imsave('%s/real_%s.png' % (img_path, ind), real_img_batch[i, :, :, :])
					plt.imsave('%s/real_recon_%s.png' % (img_path, ind), recon_img_batch[i, :, :, :])
					ind += 1
		print(ind, 'Generated Images')
	else:
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path


# Encode real images for prognosis.
def real_encode_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, batches=50, save_img=False):
	os.umask(0o002)
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	# path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'real_images_recon')
	if not os.path.isdir(path):
		os.makedirs(path)

	if not os.path.isfile(real_hdf5):
		print('Real image H5 file does not exist:', real_hdf5)
		exit()

	name_file = real_hdf5.split('/')[-1]
	hdf5_path = os.path.join(path, name_file)

	# Lazy access to one set of images, not used at all, just filling tensorflows complains.
	batch_images = np.ones((data.batch_size, data.patch_h, data.patch_w, data.n_channels))
	
	if not os.path.isfile(hdf5_path):
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		# H5 File specifications and creation.
		with h5py.File(real_hdf5, mode='r') as hdf5_file:
			with h5py.File(hdf5_path, mode='w') as hdf5_file_w:

				for key in hdf5_file.keys():
					print('\t Key: %s' % key)
					key_shape = hdf5_file[key].shape
					dtype = hdf5_file[key].dtype
					num_samples = key_shape[0]
					latent_shape = [num_samples] + [model.z_dim]
					if 'PathologyGAN_plus' in model.model_name or 'SelfPathologyGAN' in model.model_name or 'ConditionalPathologyGAN' in model.model_name:
						latent_shape = [num_samples] + [model.complete_z_dim]

					if 'image' in key or 'img' in key:
						if save_img: 
							img_storage = hdf5_file_w.create_dataset(name='%s_prime' % key, shape=key_shape, dtype=np.float32)
							w_prime_storage = hdf5_file_w.create_dataset(name='%s_w_latent_prime' % key, shape=latent_shape, dtype=np.float32)
						w_storage = hdf5_file_w.create_dataset(name='%s_w_latent' % key, shape=latent_shape, dtype=np.float32)
						
						saver = tf.train.Saver()
						with tf.Session() as session:

							# Initializer and restoring model.
							session.run(tf.global_variables_initializer())
							saver.restore(session, checkpoint)

							print('Number of Real Images:', num_samples)
							print('Starting encoding...')

							ind = 0
							while ind < num_samples:
								
								# Real images.
								if (ind + batches) < num_samples:
									real_img_batch = hdf5_file[key][ind: ind+batches, :, :, :]/255.
									
								else:
									real_img_batch = hdf5_file[key][ind:, :, :, :]/255.

								# Encode real images into W latent space.
								feed_dict = {model.real_images_2:real_img_batch}
								w_latent_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]

								if save_img: 
									w_latent_in = np.tile(w_latent_batch[:,:, np.newaxis], [1, 1, model.layers+1])

									# Generate images from W latent space.
									feed_dict = {model.w_latent_in:w_latent_in}
									recon_img_batch = session.run([model.output_gen], feed_dict=feed_dict)[0]

									# Encode reconstructed images into W latent space.
									feed_dict = {model.real_images_2:recon_img_batch}
									w_latent_prime_batch = session.run([model.w_latent_e_out], feed_dict=feed_dict)[0]

								# Fill in storage for latent and image.
								for i in range(batches):
									if ind == num_samples:
										break

									# Reconstructed images.
									if save_img: 
										img_storage[ind] = recon_img_batch[i, :, :, :]
										w_prime_storage[ind] = w_latent_prime_batch[i, :]
									w_storage[ind] = w_latent_batch[i, :]

									ind += 1

								if ind%10000==0: print('Processed', ind, 'images')
							print(ind, 'Encoded Images')
					else:
						storage = hdf5_file_w.create_dataset(name=key, shape=key_shape, dtype=dtype)
						ind = 0
						while ind < num_samples:
							# Real images.
							if (ind + batches) < num_samples:
								info_batch = hdf5_file[key][ind:ind+batches]
							else:
								info_batch = hdf5_file[key][ind:]

							# Fill in storage for latent and image.
							for i in range(batches):
								if ind == num_samples:
									break

								# Reconstructed images.
								storage[ind] = info_batch[i]
								ind += 1
	else:
		with h5py.File(hdf5_path, mode='r') as hdf5_file:
			for key in hdf5_file.keys():
				num_samples = key_shape.shape[0]
				break
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path, num_samples


# Generate sampeles from PathologyGAN, no encoder.
def discriminator_features_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, clust_percent=1.0, clusters_num=None, batches=50, save_img=False):
	os.umask(0o002)
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	# path = os.path.join(path, data.marker)
	res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
	path = os.path.join(path, res)
	img_path = os.path.join(path, 'real_images_recon')
	if not os.path.isdir(path):
		os.makedirs(path)

	if not os.path.isfile(real_hdf5):
		print('Real image H5 file does not exist:', real_hdf5)
		exit()

	name_file = real_hdf5.split('/')[-1] 
	name_file = name_file.split('.h')[0] + '_discr_features.h5'
	hdf5_path = os.path.join(path, name_file)

	if not os.path.isfile(hdf5_path):
		print('Generated Images path:', img_path)
		print('H5 File path:', hdf5_path)

		# H5 File specifications and creation.
		with h5py.File(real_hdf5, mode='r') as hdf5_file:
			with h5py.File(hdf5_path, mode='w') as hdf5_file_w:

				for key in hdf5_file.keys():
					print('\t Key: %s' % key)
					key_shape = hdf5_file[key].shape
					dtype = hdf5_file[key].dtype
					num_samples = key_shape[0]

					if 'image' in key or 'img' in key:
						if save_img: img_storage = hdf5_file_w.create_dataset(name='images', shape=key_shape, dtype=np.float32)
						feature_storage = hdf5_file_w.create_dataset(name='features', shape=[num_samples] + [model.feature_space_real.shape[1]], dtype=np.float32)
						
						saver = tf.train.Saver()
						with tf.Session() as session:

							# Initializer and restoring model.
							session.run(tf.global_variables_initializer())
							saver.restore(session, checkpoint)

							print('Number of Real Images:', num_samples, 'Feature space size:', model.feature_space_real.shape[1])
							print('Starting encoding...')

							ind = 0
							while ind < num_samples:
								
								# Real images.
								if (ind + batches) < num_samples:
									real_img_batch = hdf5_file[key][ind: ind+batches, :, :, :]/255.
									
								else:
									real_img_batch = hdf5_file[key][ind:, :, :, :]/255.

								# Encode real images into W latent space.
								feed_dict = {model.real_images:real_img_batch}
								features_batch = session.run([model.feature_space_real], feed_dict=feed_dict)[0]

								# Fill in storage for latent and image.
								for i in range(batches):
									if ind == num_samples:
										break

									# Reconstructed images.
									if save_img: img_storage[ind] = real_img_batch[i, :, :, :]*255.
									feature_storage[ind] = features_batch[i, :]

									ind += 1

								if ind%10000==0: print('Processed', ind, 'images')
							print(ind, 'Encoded Images')

						if clusters_num is not None:
							import umap
							from sklearn.cluster import KMeans

							all_indx = list(range(num_samples))
							random.shuffle(all_indx)
							selected_indx = np.array(sorted(all_indx[:int(num_samples*clust_percent)]))
							
							umap_reducer = umap.UMAP(n_components=2, random_state=45)
							umap_fitted = umap_reducer.fit(feature_storage[selected_indx, :])
							embedding_umap_clustering = umap_fitted.transform(feature_storage)

							feature_labels_storage = hdf5_file_w.create_dataset(name='feat_cluster_labels', shape=[num_samples] + [1], dtype=np.float32)
							initialization = 'k-means++'
							kmeans = KMeans(init=initialization, n_clusters=clusters_num, n_init=10).fit(embedding_umap_clustering)
							new_classes = kmeans.predict(embedding_umap_clustering)

							for i in range(num_samples):
								feature_labels_storage[i, :] = new_classes[i]
					else:
						storage = hdf5_file_w.create_dataset(name=key, shape=key_shape, dtype=dtype)
						ind = 0
						while ind < num_samples:
							# Real images.
							if (ind + batches) < num_samples:
								info_batch = hdf5_file[key][ind:ind+batches]
							else:
								info_batch = hdf5_file[key][ind:]

							# Fill in storage for latent and image.
							for i in range(batches):
								if ind == num_samples:
									break

								# Reconstructed images.
								storage[ind] = info_batch[i]
								ind += 1
	else:
		with h5py.File(hdf5_path, mode='r') as hdf5_file:
			for key in hdf5_file.keys():
				num_samples = key_shape.shape[0]
				break
		print('H5 File already created.')
		print('H5 File Generated Samples')
		print('\tFile:', hdf5_path)

	return hdf5_path, num_samples


def real_encode_contrastive_from_checkpoint(model, data, data_out_path, checkpoint, real_hdf5, batches=50, save_img=False):
	# Directory handling.
	path = os.path.join(data_out_path, 'results')
	path = os.path.join(path, model.model_name)
	path = os.path.join(path, data.dataset)
	res = 'h%s_w%s_n%s_zdim%s' % (data.patch_h, data.patch_w, data.n_channels, model.z_dim)
	path = os.path.join(path, res)
	if not os.path.isdir(path):
		os.makedirs(path)
	if not os.path.isfile(real_hdf5):
		print('H5 File not found:', real_hdf5)
		exit()

	# Extracting name for projections.
	name_file = real_hdf5.split('/')[-1]
	hdf5_path = os.path.join(path, name_file)
	# Check if file is already there.
	print('H5 Projections file path:', hdf5_path)
	if not os.path.isfile(hdf5_path):

		# H5 File specifications and creation.
		with h5py.File(real_hdf5, mode='r') as hdf5_file:
			with h5py.File(hdf5_path, mode='w') as hdf5_file_w:

				# Iterate through H5 datasets.
				for key in hdf5_file.keys():
					print('\t Key: %s' % key)
					key_shape = hdf5_file[key].shape
					dtype = hdf5_file[key].dtype
					num_samples = key_shape[0]

					# Processing the image dataset.
					if 'image' in key or 'img' in key:
						h_latent_shape = [num_samples] + [model.h_rep_out.shape[1]]
						z_latent_shape = [num_samples] + [model.z_rep_out.shape[1]]
						h_storage      = hdf5_file_w.create_dataset(name='%s_h_latent' % key, shape=h_latent_shape, dtype=np.float32)
						z_storage      = hdf5_file_w.create_dataset(name='%s_z_latent' % key, shape=z_latent_shape, dtype=np.float32)
						if 'ContrastivePathology_SwAV' in model.model_name:
							z_norm_latent_shape  = [num_samples] + [model.z_norm_out.shape[1]]
							prot_latent_shape    = [num_samples] + [model.prot_out.shape[1]]
							z_norm_storage       = hdf5_file_w.create_dataset(name='%s_z_norm_latent' % key, shape=z_norm_latent_shape, dtype=np.float32)
							prot_storage         = hdf5_file_w.create_dataset(name='%s_prot_latent' % key,   shape=prot_latent_shape,   dtype=np.float32)
						if save_img: img_storage = hdf5_file_w.create_dataset(name='%s' % key, shape=key_shape, dtype=np.float32)
						
						saver = tf.train.Saver()
						with tf.Session() as session:

							# Initializer and restoring model.
							session.run(tf.global_variables_initializer())
							saver.restore(session, checkpoint)

							print('Number of Real Images:', num_samples)
							print('Starting encoding...')

							ind = 0
							while ind < num_samples:					
								# Image batch construction.
								if (ind + batches) < num_samples:
									real_img_batch = hdf5_file[key][ind: ind+batches, :, :, :]/255.
									
								else:
									real_img_batch = hdf5_file[key][ind:, :, :, :]/255.
								# Encode real images into W latent space.
								feed_dict = {model.real_images_2:real_img_batch}
								if 'ContrastivePathology_SwAV' in model.model_name:
									outputs_model = [model.h_rep_out, model.z_rep_out, model.z_norm_out, model.prot_out]
								else:
									outputs_model = [model.h_rep_out, model.z_rep_out]
								outputs = session.run(outputs_model, feed_dict=feed_dict)
								# Save batch samples into storage.
								for i in range(batches):
									if ind == num_samples:
										break
									h_storage[ind] = outputs[0][i, :]
									z_storage[ind] = outputs[1][i, :]
									if 'ContrastivePathology_SwAV' in model.model_name:
										z_norm_storage[ind] = outputs[2][i, :]
										prot_storage[ind]   = outputs[3][i, :]
									if save_img: img_storage[ind] = real_img_batch[i, :, :, :]
									ind += 1
								# Report progress.
								if ind%10000==0: print('Processed', ind, 'images')
							print(ind, 'Encoded Images')
					# Carry on any other dataset.
					else:
						storage = hdf5_file_w.create_dataset(name=key, shape=key_shape, dtype=dtype)
						ind = 0
						while ind < num_samples:
							# Real images.
							if (ind + batches) < num_samples:
								info_batch = hdf5_file[key][ind:ind+batches]
							else:
								info_batch = hdf5_file[key][ind:]

							# Fill in storage for latent and image.
							for i in range(batches):
								if ind == num_samples:
									break

								# Reconstructed images.
								storage[ind] = info_batch[i]
								ind += 1
	# H5 File already created.													
	else:
		# Retrieve number of samples.
		with h5py.File(hdf5_path, mode='r') as hdf5_file:
			for key in hdf5_file.keys():
				num_samples = key_shape.shape[0]
				break
		print('H5 File already created.')

	return hdf5_path, num_samples
