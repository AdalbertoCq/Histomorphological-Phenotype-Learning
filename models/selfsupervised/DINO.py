import tensorflow as tf
import numpy as np
try:
	import wandb
	from models.wandb_utils import *
	wandb_flag = True
except:
	wandb_flag = False
	print('Not using W&B')

# Evaluation and Visualization lib.
from models.evaluation.latent_space import *
from models.evaluation.features import *

# Data/Folder Manipulation lib.
from data_manipulation.utils import *
from models.evaluation import *
from models.tools import *
from models.utils import *

# Network related lib.
from models.networks.encoder_contrastive import *
from models.data_augmentation import *
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.ops import *

# Losses and Optimizers.
from models.optimizer import *
from models.loss import *

class RepresentationsPathology():
	def __init__(self,
				data,                       			# Dataset type, training, validation, and test data.
				z_dim,	                    			# Latent space dimensions.
				beta_1,                      			# Beta 1 value for Adam Optimizer.
				learning_rate_e,             			# Learning rate Encoder.
				teacher_temp=0.02,                       # Softmax temperature for teacher network.
				student_temp=0.1,                       # Softmax temperature for student network.
				center_mom=0.9,							# Momentum for teacher representation center. 
				spectral=True,							# Spectral Normalization for weights.
				layers=5,					 			# Number for layers for Encoder.
				attention=28,                			# Attention Layer dimensions, default after height and width equal 28 to pixels.
				power_iterations=1,          			# Iterations of the power iterative method: Calculation of Eigenvalues, Singular Values.
				init = 'orthogonal',    			    # Weight Initialization: default Orthogonal.
				regularizer_scale=1e-4,      			# Orthogonal regularization.
				beta=0.9995, 							# Beta value for EMA on teacher/student networks.
				warmup_epochs=1,						# Number of epochs where the last layer of the Teacher's head is frozen.
				model_name='RepresentationsPathology'   # Model Name.
				):

		### Input data variables.
		self.image_height   = data.patch_h
		self.image_width    = data.patch_w
		self.image_channels = data.n_channels
		self.batch_size     = data.batch_size

		### Architecture parameters.
		self.attention = attention
		self.layers    = layers
		self.spectral  = spectral
		self.z_dim     = z_dim
		self.init      = init
		self.out_dim   = self.z_dim

		### Hyper-parameters.
		self.power_iterations  = power_iterations
		self.regularizer_scale = regularizer_scale
		self.learning_rate_e   = learning_rate_e
		self.beta_1            = beta_1
		self.beta              = beta 
		self.teacher_temp      = teacher_temp
		self.student_temp      = student_temp
		self.center_mom        = center_mom
		self.warmup_epochs     = warmup_epochs

		### Data augmentation conditions.
		# Spatial transformation.
		self.crop          = True
		self.rotation      = True
		self.flip          = True
		# Color transformation.
		self.color_distort = True
		# Gaussian Blur and Noise.
		self.g_blur        = False
		self.g_noise       = False
		# Sobel Filter.
		self.sobel_filter  = False

		self.num_samples = 1000
		self.num_samples_t = data.training.images.shape[0]
		all_indx = list(range(self.num_samples_t))
		random.shuffle(all_indx)
		self.selected_indx = np.array(sorted(all_indx[:self.num_samples]))

		self.model_name = model_name

		self.wandb_flag = wandb_flag
		self.build_model()

	# Self-supervised inputs.
	def model_inputs(self):

		# Image input for transformation.
		real_images_1 = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images_1')
		real_images_2 = tf.placeholder(dtype=tf.float32, shape=(None, self.image_width, self.image_height, self.image_channels), name='real_images_2')

		# Transformed images.
		transf_real_images_1 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.image_width, self.image_height, self.image_channels), name='transf_real_images_1')
		transf_real_images_2 = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.image_width, self.image_height, self.image_channels), name='transf_real_images_2')

		# Center for teacher representations. 
		center_input = tf.placeholder(dtype=tf.float32, shape=(self.out_dim), name='teacher_center')

		# Learning rates.
		learning_rate_e = tf.placeholder(dtype=tf.float32, name='learning_rate_e')

		return real_images_1, real_images_2, transf_real_images_1, transf_real_images_2, center_input, learning_rate_e

	# Data Augmentation Layer.
	def data_augmentation_layer(self, images, crop, rotation, flip, g_blur, g_noise, color_distort, sobel_filter):
		images_trans = images

		# Spatial transformations.
		if crop:
			images_trans = tf.map_fn(random_crop_and_resize_p075, images_trans)
		if rotation:
			images_trans = tf.map_fn(random_rotate, images_trans)
		if flip:
			images_trans = random_flip(images_trans)

		# Gaussian blur and noise transformations.
		if g_blur:
			images_trans = tf.map_fn(random_blur, images_trans)
		if g_noise:
			images_trans = tf.map_fn(random_gaussian_noise, images_trans)

		# Color distorsions.
		if color_distort:
			images_trans = tf.map_fn(random_color_jitter_1p0, images_trans)
		else:
			images_trans = tf_wrapper_rb_stain(images_trans)

		# Sobel filter.
		if sobel_filter:
			images_trans = tf.map_fn(random_sobel_filter, images_trans)

		# Make sure the image batch is in the right format.
		images_trans = tf.reshape(images_trans, [-1, self.image_height, self.image_width, self.image_channels])
		images_trans = tf.clip_by_value(images_trans, 0., 1.)

		return images_trans

	# Encoder Network.
	def encoder(self, images, is_train, reuse, init, name, label_input=None):
		if '_0' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_1' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_1(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_2' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_2(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_3' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_3(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_4' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_4(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_5' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_5(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_6' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_6(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		elif '_7' in self.model_name:
			conv_space, h, z = encoder_resnet_contrastive_7(images=images, z_dim=self.z_dim, h_dim=1024, layers=self.layers, spectral=self.spectral, activation=ReLU, reuse=reuse, init=init,
															 is_train=is_train, normalization=batch_norm, regularizer=None, attention=self.attention, name=name)
		return conv_space, h, z

	# Dino Head Network.
	def head(self, z_rep, is_train, reuse, init, name):
		q = dino_head(z_rep=z_rep, z_dim=self.out_dim, spectral=self.spectral, h_dim=self.z_dim, activation=ReLU, is_train=is_train, reuse=reuse, init=init, regularizer=None, normalization=batch_norm, name=name)
		return q

	# Loss Function.
	def loss(self, teacher_rep, student_rep, teacher_temp, student_temp, center):
		loss = dino_loss(teacher_rep=teacher_rep, student_rep=student_rep, teacher_temp=teacher_temp, student_temp=student_temp, center=center)
		return loss
		
	# Optimizer.
	def optimization(self):
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			trainable_variables         = tf.trainable_variables()
			encoder_total_variable      = list()
			encoder_online_variables    = [variable for variable in trainable_variables if variable.name.startswith('contrastive_encoder_online')]
			encoder_predictor_variables = [variable for variable in trainable_variables if variable.name.startswith('head_online')]
			encoder_total_variable.extend(encoder_online_variables)
			encoder_total_variable.extend(encoder_predictor_variables)
			train_online_encoder = tf.train.AdamOptimizer(learning_rate=(self.learning_rate_input_e*self.batch_size/256),  beta1=self.beta_1).minimize(self.loss_cross, var_list=encoder_total_variable)
		return train_online_encoder

	# Update teacher network.
	def update_target_encoder(self,):
		# Update Backbone.
		trainable_variables = tf.trainable_variables()
		encoder_online_variables = [variable for variable in trainable_variables if variable.name.startswith('contrastive_encoder_online')]
		encoder_target_variables = [variable for variable in trainable_variables if variable.name.startswith('contrastive_encoder_target')]
		self.ops        = list()
		self.ops_xfinal = list()
		for online_var, target_var in zip(encoder_online_variables, encoder_target_variables):
			online_value = online_var.read_value()
			target_value = target_var.read_value()
			self.ops.append(target_var.assign(self.beta*target_value + (1-self.beta)*online_value))
			self.ops_xfinal.append(target_var.assign(self.beta*target_value + (1-self.beta)*online_value))

		# Update Head.
		encoder_online_variables = [variable for variable in trainable_variables if variable.name.startswith('head_online')]
		encoder_target_variables = [variable for variable in trainable_variables if variable.name.startswith('head_target')]
		for online_var, target_var in zip(encoder_online_variables, encoder_target_variables):
			online_value = online_var.read_value()
			target_value = target_var.read_value()
			self.ops.append(target_var.assign(self.beta*target_value + (1-self.beta)*online_value))
			if 'q_pred' not in target_var.name: 
				self.ops_xfinal.append(target_var.assign(self.beta*target_value + (1-self.beta)*online_value))

	# Build the Self-supervised.
	def build_model(self):

		from tensorflow.python.client import device_lib
		local_device_protos = device_lib.list_local_devices()
		avail_gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']

		################### INPUTS & DATA AUGMENTATION #####################################################################################################################################
		# Inputs.
		self.real_images_1, self.real_images_2, self.transf_real_images_1, self.transf_real_images_2, self.center_input, self.learning_rate_input_e = self.model_inputs()
		# Data augmentation layer.
		self.real_images_1_t1 = self.data_augmentation_layer(images=self.real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
															 color_distort=self.color_distort, sobel_filter=self.sobel_filter)
		self.real_images_1_t2 = self.data_augmentation_layer(images=self.real_images_1, crop=self.crop, rotation=self.rotation, flip=self.flip, g_blur=self.g_blur, g_noise=self.g_noise,
															 color_distort=self.color_distort, sobel_filter=self.sobel_filter)

		################### TRAINING #####################################################################################################################################

		# Online Encoder.
		self.conv_space_ot1, self.h_rep_ot1, self.z_rep_ot1 = self.encoder(images=self.transf_real_images_1, is_train=True, reuse=False, init=self.init, name='contrastive_encoder_online')
		self.conv_space_ot2, self.h_rep_ot2, self.z_rep_ot2 = self.encoder(images=self.transf_real_images_2, is_train=True, reuse=True,  init=self.init, name='contrastive_encoder_online')
		self.p1 = self.head(z_rep=self.z_rep_ot1, is_train=True, reuse=False, init=self.init, name='head_online')
		self.p2 = self.head(z_rep=self.z_rep_ot2, is_train=True, reuse=True,  init=self.init, name='head_online')

		# Target Encoder.
		self.conv_space_tt1, self.h_rep_tt1, self.z_rep_tt1 = self.encoder(images=self.transf_real_images_1, is_train=True, reuse=False, init=self.init, name='contrastive_encoder_target')
		self.conv_space_tt2, self.h_rep_tt2, self.z_rep_tt2 = self.encoder(images=self.transf_real_images_2, is_train=True, reuse=True,  init=self.init, name='contrastive_encoder_target')
		self.q1 = self.head(z_rep=self.z_rep_tt1, is_train=True, reuse=False, init=self.init, name='head_target')
		self.q2 = self.head(z_rep=self.z_rep_tt2, is_train=True, reuse=True,  init=self.init, name='head_target')

		# Mean center. 
		self.mean_batch = tf.reduce_mean(tf.concat([self.q1, self.q2], axis=0), axis=[0])

		################### INFERENCE #####################################################################################################################################
		# Encoder Representations Inference.
		self.conv_space_out, self.h_rep_out, self.z_rep_out = self.encoder(images=self.real_images_2, is_train=False, reuse=True, init=self.init, name='contrastive_encoder_target')

		################### LOSS & OPTIMIZER ##############################################################################################################################
		# Losses.
		self.loss_cross = self.loss(self.q1, self.p2, self.teacher_temp, self.student_temp, self.center_input)*0.5 + \
						  self.loss(self.q2, self.p1, self.teacher_temp, self.student_temp, self.center_input)*0.5

		# Optimizers.
		self.train_encoder  = self.optimization()

	# Project subsample to look at the representation space.
	def project_subsample(self, session, data, epoch, data_out_path, report, batch_size=50):
		# Updated
		if not report:
			return

		# Handle directories and copies.
		results_path = os.path.join(data_out_path, 'results')
		epoch_path = os.path.join(results_path, 'epoch_%s' % epoch)
		check_epoch_path = os.path.join(epoch_path, 'checkpoints')
		checkpoint_path = os.path.join(results_path, '../checkpoints')
		os.makedirs(epoch_path)
		shutil.copytree(checkpoint_path, check_epoch_path)

		# Setup HDF5 file.
		hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_%s_projected_images.h5' % epoch)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		img_storage  = hdf5_file.create_dataset(name='images',           shape=[self.num_samples, data.patch_h, data.patch_w, data.n_channels], dtype=np.float32)
		conv_storage = hdf5_file.create_dataset(name='conv_features',    shape=[self.num_samples] + self.conv_space_ot1.shape.as_list()[1:],     dtype=np.float32)
		h_storage    = hdf5_file.create_dataset(name='h_representation', shape=[self.num_samples] + self.h_rep_ot1.shape.as_list()[1:],          dtype=np.float32)
		z_storage    = hdf5_file.create_dataset(name='z_representation', shape=[self.num_samples] + self.z_rep_ot1.shape.as_list()[1:],          dtype=np.float32)

		ind = 0
		while ind<self.num_samples:
			images_batch = data.training.images[self.selected_indx[ind: ind+batch_size], :, :, :]
			feed_dict = {self.real_images_2:images_batch}
			conv_space_out, h_rep_out, z_rep_out = session.run([self.conv_space_out, self.h_rep_out, self.z_rep_out], feed_dict=feed_dict)

			img_storage[ind: ind+batch_size, :, : ,:]  = images_batch
			conv_storage[ind: ind+batch_size, :] = conv_space_out
			h_storage[ind: ind+batch_size, :]    = h_rep_out
			z_storage[ind: ind+batch_size, :]    = z_rep_out
			ind += batch_size
		try:
			conv_path, label_conv_path = report_progress_latent(epoch=epoch, w_samples=conv_storage, img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='conv_lat', metric='euclidean', n_components=100)
			h_path   , label_h_path    = report_progress_latent(epoch=epoch, w_samples=h_storage,    img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='h_lat',    metric='euclidean', n_components=100)
			z_path   , label_z_path    = report_progress_latent(epoch=epoch, w_samples=z_storage,    img_samples=img_storage, img_path=hdf5_path.split('/hdf5')[0], storage_name='z_lat',    metric='euclidean', n_components=100)
			if self.wandb_flag:
				wandb.log({"Conv Space": wandb.Image(conv_path), "H Space":    wandb.Image(h_path), "Z Space":    wandb.Image(z_path)})
		except Exception as ex:
			print('Issue printing latent space images. Epoch', epoch)
			if hasattr(ex, 'message'):
				print('\t\tException', ex.message)
			else:
				print('\t\tException', ex)
		finally:
			os.remove(hdf5_path)

	# Training function.
	def train(self, epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=None, report=False):

		if self.wandb_flag:
			train_config = save_model_config(self, data)
			run_name = self.model_name + '-' + data.dataset
			wandb.init(project='SSL Pathology', entity='adalbertocquiros', name=run_name, config=train_config)

		run_epochs = 0
		saver = tf.train.Saver()

		# Setups.
		checkpoints, csvs = setup_output(data_out_path=data_out_path, model_name=self.model_name, restore=restore)
		losses = ['Cross Entropy Loss Train']
		if data.validation is not None:
			losses.append('Cross Entropy Loss Validation')
		setup_csvs(csvs=csvs, model=self, losses=losses)
		report_parameters(self, epochs, restore, data_out_path)

		# Session Options.
		# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

		print('Number of trainable parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

		print('Starting run.')
		# Center for teacher representations. 
		center_run = np.zeros((self.out_dim))

		# Training session.
		with tf.Session(config=config) as session:
			if self.wandb_flag: wandb.tensorflow.log(tf.summary.merge_all())
			session.run(tf.global_variables_initializer())

			# Restore previous session.
			if restore:
				check = get_checkpoint(data_out_path)
				saver.restore(session, check)
				print('Restored model: %s' % check)

			# Example of augmentation images.
			batch_images, batch_labels = data.training.next_batch(n_images)
			data.training.reset()
			feed_dict = {self.real_images_1:batch_images, self.real_images_2:batch_images}
			output_layers_transformed = [self.real_images_1_t1, self.real_images_1_t2]
			transformed_images = session.run(output_layers_transformed, feed_dict=feed_dict, options=run_options)
			write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_1.png'), data=transformed_images[0], metadata=False)
			write_sprite_image(filename=os.path.join(data_out_path, 'images/transformed_2.png'), data=transformed_images[1], metadata=False)
			if self.wandb_flag:
				dict_ = {"transformed_1": wandb.Image(os.path.join(data_out_path, 'images/transformed_1.png')), "transformed_2": wandb.Image(os.path.join(data_out_path, 'images/transformed_2.png'))}
				wandb.log(dict_)

			# Define operation to update target network.
			self.update_target_encoder()

			# Epoch Iteration.
			for epoch in range(0, epochs+1):

				# Batch Iteration.
				for batch_images, batch_labels in data.training:

					################################# TRAINING ENCODER #################################################
					feed_dict = {self.real_images_1:batch_images, self.real_images_2:batch_images}
					transformed_images = session.run(output_layers_transformed, feed_dict=feed_dict, options=run_options)
					feed_dict = {self.transf_real_images_1:transformed_images[0], self.transf_real_images_2:transformed_images[1], \
								 self.real_images_1:batch_images, self.real_images_2:batch_images, self.learning_rate_input_e: self.learning_rate_e, self.center_input:center_run}
					outputs = session.run([self.train_encoder, self.mean_batch], feed_dict=feed_dict, options=run_options)
					center_run  = self.center_mom * center_run + (1-self.center_mom) * outputs[1]

					# Train target network: Moving average of online network.
					if epochs > self.warmup_epochs:
						session.run(self.ops)
					else:
						session.run(self.ops_xfinal)

					####################################################################################################
					if run_epochs % print_epochs == 0:
						model_outputs = [self.loss_cross]
						epoch_outputs = session.run(model_outputs, feed_dict=feed_dict, options=run_options)

						# Validation loss. 
						if data.validation is not None:
							for batch_images, batch_labels in data.validation:
								feed_dict = {self.real_images_1:batch_images, self.real_images_2:batch_images}
								output_layers_transformed = [self.real_images_1_t1, self.real_images_1_t2]
								transformed_images = session.run(output_layers_transformed, feed_dict=feed_dict, options=run_options)
								feed_dict = {self.transf_real_images_1:transformed_images[0], self.transf_real_images_2:transformed_images[1], \
											 self.real_images_1:batch_images, self.real_images_2:batch_images, self.learning_rate_input_e: self.learning_rate_e, self.center_input:center_run}
								val_outputs = session.run(model_outputs, feed_dict=feed_dict, options=run_options)
								update_csv(model=self, file=csvs[0], variables=[epoch_outputs[0], val_outputs[0]], epoch=epoch, iteration=run_epochs, losses=losses)
								if self.wandb_flag: wandb.log({'Redundancy Reduction Loss Train': epoch_outputs[0], 'Redundancy Reduction Loss Validation': val_outputs[0],})
								break
						else:
							update_csv(model=self, file=csvs[0], variables=[epoch_outputs[0]], epoch=epoch, iteration=run_epochs, losses=losses)
							if self.wandb_flag: wandb.log({'Redundancy Reduction Loss Train': epoch_outputs[0]})

					run_epochs += 1
					if epoch==0: break

				# Save model.
				saver.save(sess=session, save_path=checkpoints)
				data.training.reset()

				############################### FID TRACKING ##################################################
				# Save checkpoint and generate images for FID every X epochs.
				if (checkpoint_every is not None and epoch % checkpoint_every == 0) or (epochs==epoch):
					self.project_subsample(session=session, data=data, epoch=epoch, data_out_path=data_out_path, report=report)

		if self.wandb_flag:
			wandb.finish()
