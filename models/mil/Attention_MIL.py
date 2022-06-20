import tensorflow as tf
import numpy as np
import random
import shutil
from PIL import Image
from data_manipulation.utils import *
from models.evaluation.features import *
from models.generative.ops import *
from models.generative.utils import *
from models.generative.tools import *
from models.generative.loss import *
from models.generative.regularizers import *
from models.generative.activations import *
from models.generative.normalization import *
from models.generative.evaluation import *
from models.generative.optimizer import *
from models.generative.discriminator import *
from models.generative.generator import *
from models.generative.encoder import *


'''
GAN model combining features from BigGAN, StyleGAN, and Relativistic average iscriminator.
	1. Attention network: SAGAN/BigGAN.
	2. Orthogonal initalization and regularization: SAGAN/BigGAN.
	3. Spectral normalization: SNGAN/SAGAN/BigGAN.
	4. Mapping network: StyleGAN.
	5. Relativistic average discriminator.
'''
class Attention_MIL():
	def __init__(self,
				z_dim,
				att_dim,
				bag_size, 
				learning_rate=0.0005, 
				beta_1=0.9, 
				beta_2=0.999, 
				model_name='Attention_MIL', 
				):
		self.z_dim = z_dim
		self.att_dim = att_dim
		self.bag_size = bag_size
		self.learning_rate = learning_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.model_name = model_name

		self.build_model()
		
	# StyleGAN inputs
	def model_inputs(self):
		represenation_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='represenation_input')
		label_input = tf.placeholder(dtype=tf.float32, shape=(None), name='label_input')

		return represenation_input, label_input

	# Attention Network.
	def feature_extractor(self, inputs, use):
		interm = inputs
		if use:
			net = dense(inputs=inputs, out_dim=int(self.z_dim/2), scope=1, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
			net = ReLU(net)
			net = dense(inputs=net,    out_dim=int(self.z_dim/2), scope=2, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
			interm = ReLU(net)
		
		return interm
	
	# Attention Network.
	def attention(self, inputs):
		net1 = dense(inputs=inputs, out_dim=self.att_dim, scope=3, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
		net1 = tanh(net1)
		net2 = dense(inputs=inputs, out_dim=self.att_dim, scope=4, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
		net2 = sigmoid(net2)
		net = net1*net2
		net = dense(inputs=net, out_dim=1, scope=5, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
		weigths = tf.nn.softmax(net, axis=0)
		return weigths
	
	# Classifier Network.
	def classifier(self, interm, weights):
		z = interm*weights
		z = tf.reshape(tf.reduce_sum(z, axis=0), (1,-1))
		net = dense(inputs=z, out_dim=1, scope=10, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)		
		prob = sigmoid(net)					
		return prob, z

	# Loss function.
	def loss(self, label, prob):
		# Negative log likelihood
		prob = tf.clip_by_value(prob, clip_value_min=1e-5, clip_value_max=1.-1e-5)
		loss = -(label*tf.log(prob) + (1.-label)*tf.log(1.-prob))
		return loss

	# Optimizer.
	def optimization(self):
		trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2).minimize(self.loss)
		return trainer

	# Put together the GAN.
	def build_model(self):

		with tf.device('/gpu:0'):
			# Inputs.
			self.represenation_input, self.label_input = self.model_inputs()

			# Optional NN for latent representations.
			self.interm = self.feature_extractor(inputs=self.represenation_input, use=True)

			# Attention.
			self.weights = self.attention(inputs=self.interm)

			# Classifier.
			self.prob, self.z = self.classifier(interm=self.interm, weights=self.weights)
			self.clas = tf.to_float(tf.math.greater(self.prob, 0.5))
			
			# Losses.
			self.loss = self.loss(label=self.label_input, prob=self.prob)

			# Optimizers.
			self.trainer  = self.optimization()

	def pro(self, labels):
		batch_samples = np.zeros((labels.shape[0], 2))
		for i in range(labels.shape[0]):
			if np.sum(labels[i]) > 0:
				batch_samples[i, 1] = 1
		return batch_samples


	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, dataset_name='crc_histophen', img_size=56, test_size=9, folds=10, label_dim=0):
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)
		
		# Open file for data manipulation. 
		hdf5_content = h5py.File(hdf5_file_path, mode='r')
		latent = hdf5_content['images_w_latent']
		labels = hdf5_content['labels']
		if 'file_name' in hdf5_content:
			file_names = hdf5_content['file_name']
		results = np.zeros((folds, test_size))
		for fold in range(folds):

			fold_output_path = os.path.join(data_out_path, 'fold_%s' % fold)
			os.makedirs(fold_output_path)

			# Setup folder for outputs.
			checkpoints, csvs = setup_output(data_out_path=fold_output_path, model_name=self.model_name, restore=False)
			losses = ['Loss']
			setup_csvs(csvs=csvs, model=self, losses=losses)
			report_parameters(self, epochs=epochs, restore=False, data_out_path=fold_output_path)

			# Samples for training and testing. 
			if 'crc' in dataset_name:
				pat_id_index = 4
				y_index = 5
				x_index = 6
			elif dataset_name == 'nki':
				pat_id_index = 1
				y_index = 2
				x_index = 3
			elif dataset_name == 'vgh':
				pat_id_index = 1
				y_index = 2
				x_index = 3
			else:
				pat_id_index = 4
				y_index = 2
				x_index = 3
			#################################################
			# Patient ID Handling
			if 'crc' in dataset_name:
				sample_indxs = list(np.unique(labels[:, pat_id_index]).astype(np.int32))
			elif dataset_name == 'nki' or dataset_name == 'vgh' or dataset_name == 'bisque':
				sample_indxs = list(np.unique(file_names))
			random.shuffle(sample_indxs)
			test_samples = sample_indxs[:test_size]
			#################################################

			run_epochs = 0    
			saver = tf.train.Saver()
			
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			with tf.Session(config=config) as session:
				session.run(tf.global_variables_initializer())
				
				for epoch in range(1, epochs+1):
					
					e_losses = list()
					for sample_indx in sample_indxs:
						if sample_indx in test_samples: continue
						
						#################################################
						# Label Handling
						if 'crc' in dataset_name:
							indxs = np.argwhere(labels[:, pat_id_index]==sample_indx)[:,0]
							num_epi = np.sum(labels[indxs, label_dim]).astype(np.int32)
						elif dataset_name == 'nki':
							indxs = np.argwhere(file_names[:]==sample_indx)[:,0]
							num_epi = np.sum((labels[indxs, label_dim]<5)*1)
						elif dataset_name == 'bisque':
							indxs = np.argwhere(file_names[:]==sample_indx)[:,0]
							num_epi = np.sum((labels[indxs, label_dim]>0)*1)
						if num_epi == 0:
						    label_batch = 0
						else: 
						    label_batch = 1
						#################################################
						if indxs.shape[0] == 0: 
							print('No samples for', sample_indx)
							continue

						lantents_batch = latent[indxs, :]
						feed_dict = {self.represenation_input:lantents_batch, self.label_input:label_batch}
						_, epoch_loss = session.run([self.trainer, self.loss], feed_dict=feed_dict)
						e_losses.append(epoch_loss)
					update_csv(model=self, file=csvs[0], variables=[np.mean(e_losses)], epoch=epoch, iteration=run_epochs, losses=losses)

					saver.save(sess=session, save_path=checkpoints)

				with open(os.path.join(fold_output_path, 'test_results.txt'), 'w') as file_content_test:
					with open(os.path.join(fold_output_path, 'train_results.txt'), 'w') as file_content_train:
						imgs_path = os.path.join(fold_output_path, 'images')
						test_i = 0
						for sample_indx in sample_indxs:
							if 'crc' in dataset_name:
								indxs = np.argwhere(labels[:, pat_id_index]==sample_indx)[:,0]
								num_epi = np.sum(labels[indxs, label_dim]).astype(np.int32)
							elif dataset_name == 'nki':
								indxs = np.argwhere(file_names[:]==sample_indx)[:,0]
								num_epi = np.sum((labels[indxs, label_dim]<5)*1)
							elif dataset_name == 'bisque':
								indxs = np.argwhere(file_names[:]==sample_indx)[:,0]
								num_epi = np.sum((labels[indxs, label_dim]>0)*1)
							if num_epi == 0:
							    label_batch = 0
							else: 
							    label_batch = 1
							if indxs.shape[0] == 0: 
								print('No samples for', sample_indx)
								continue

							lantents_batch = latent[indxs, :]
							feed_dict = {self.represenation_input:lantents_batch, self.label_input:label_batch}
							prob, clas, weights = session.run([self.prob, self.clas, self.weights], feed_dict=feed_dict)
							ind = np.argsort(weights.reshape((1,-1)))[0,:]
							labP = labels[indxs, label_dim]

							if sample_indx in test_samples: 
								file_content = file_content_test
								results[fold, test_i] = (clas==label_batch)*1
								test_i += 1
							else:
								file_content = file_content_train

							if 'crc' in dataset_name:
								file_img = '/media/adalberto/Disk2/PhD_Workspace/dataset_preprocessing/crc_histophen/Selected_%s/img%s.png' % (img_size, sample_indx)
								if label_dim == 0:
									cell_type = 'epithelial'
								elif label_dim == 1:
									cell_type = 'fibroblast'
								elif label_dim == 2:
									cell_type = 'inflammatory'
								file_img_epi = '/media/adalberto/Disk2/PhD_Workspace/dataset_preprocessing/crc_histophen/Selected_%s/img%s_mask_%s.png' % (img_size, sample_indx, cell_type)
								img_epi = np.array(Image.open(file_img_epi))/255.
							elif dataset_name == 'nki':
								file_img = '/media/adalberto/Disk2/PhD_Workspace/dataset_preprocessing/nki/images/Selected_%s/%s' % (img_size, sample_indx)
							elif dataset_name == 'bisque':
								file_img  = '/media/adalberto/Disk2/PhD_Workspace/dataset_preprocessing/bisque/Selected_%s/%s_ccd.png' % (img_size, sample_indx)
								file_gt   = '/media/adalberto/Disk2/PhD_Workspace/dataset_preprocessing/bisque/Selected_%s/%s_ccd_gt.png' % (img_size, sample_indx)

							img = np.array(Image.open(file_img))/255.

							y_s = labels[indxs, y_index].astype(np.int32)
							x_s = labels[indxs, x_index].astype(np.int32)

							weights = (weights - np.min(weights))/(np.max(weights) - np.min(weights))
							weighted_patches = np.zeros((img.shape))
							for y,x,w in zip(y_s, x_s, weights):
								patch = weighted_patches[y:y+img_size, x:x+img_size, :]
								patch[patch<w] = w		
								weighted_patches[y:y+img_size, x:x+img_size, :] = patch

							weighted_img = np.ones(img.shape) * (1-weighted_patches) + weighted_patches * img

							if 'crc' in dataset_name:
								plt.imsave(os.path.join(imgs_path, 'img%s.png' % sample_indx),                      img)
								plt.imsave(os.path.join(imgs_path, 'img%s_weighted.png' % sample_indx),             weighted_img)
								plt.imsave(os.path.join(imgs_path, 'img%s_mask_%s.png' % (sample_indx, cell_type)), img_epi)
							elif dataset_name == 'nki':
								plt.imsave(os.path.join(imgs_path, str(sample_indx)),                             img)
								plt.imsave(os.path.join(imgs_path, sample_indx.replace('.jpg', '_weighted.jpg')), weighted_img)
							elif dataset_name == 'bisque':
								gt = np.array(Image.open(file_gt))/255.
								plt.imsave(os.path.join(imgs_path,'%s_ccd.png' % str(sample_indx)),           img)
								plt.imsave(os.path.join(imgs_path, '%s_ccd_weighted.png' % str(sample_indx)), weighted_img)
								plt.imsave(os.path.join(imgs_path, '%s_ccd_gt.png' % str(sample_indx)), gt)

							file_content.write(' '.join(['Test sample', str(sample_indx),  'Prob:', str(prob[0]), 'Class:', str(clas[0]), 'Label:', str(label_batch), 'Patches:', str(indxs.shape[0]), '\n']))
							file_content.write(' '.join(['\t', 'Prob:', str(prob[0]), '\n']))
							file_content.write(' '.join(['\t', 'Weig:', str(weights.reshape((1,-1))[0, ind]), '\n']))
							file_content.write(' '.join(['\t', 'LabP:', str(labP[ind]), '\n']))
							file_content.write(' '.join(['\n']))

		with open(os.path.join(data_out_path, 'results.txt'), 'w') as file_content:
			file_content.write(' '.join(['Folds accuracy:', str(np.sum(results, axis=-1)/test_size),          '\n']))
			file_content.write(' '.join(['Final Acc:',      str(np.mean(np.sum(results, axis=-1)/test_size)), '\n']))









	