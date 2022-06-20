# Imports.
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py
try:
	import wandb
	from models.wandb_utils import *
	wandb_flag = True
except:
	wandb_flag = False

# Network related lib.
from models.normalization import *
from models.regularizers import *
from models.activations import *
from models.evaluation import *
from models.optimizer import *
from models.loss import *
from models.ops import *

# Metrics.
from models.evaluation.metrics import *

# Data/Folder Manipulation lib.
from models.evaluation.folds import *
from data_manipulation.utils import *
from models.utils import *


class Attention_MIL():
	def __init__(self,
				z_dim,                       # Latent space dimensionality for projections.
				att_dim,					 # Attention network dimesionality.
				init='xavier',               # Network initializer.
				bag_size=10000,              # Maximum number of instances for a bag prediction.
				learning_rate=0.0005, 		 # Learning rate for Deep Attention MIL framework.
				beta_1=0.9,                  # Beta 1 value for Adam optimizer.
				beta_2=0.999,                # Beta 2 value for Adam optimizer.
				use_gated=True,              # Use gated attention.
				model_name='Attention_MIL',  # Name of the Deep Attention MIL run.
				gan_model='SSL_model',       # Name of the representation learning model.
				):

		# Model architecture parameters.
		self.z_dim         = z_dim
		self.att_dim       = att_dim
		self.bag_size      = bag_size
		self.init          = init
		self.use_gated     = use_gated

		# Hyperparameters for training.
		self.beta_1            = beta_1
		self.beta_2            = beta_2
		self.learning_rate     = learning_rate
		self.regularizer_scale = self.learning_rate*1e+3

		# Naming.
		self.model_name    = model_name
		self.gan_model     = gan_model
		
		# Number of clases for prediction.
		self.mult_class    = 2
		self.labels_unique = np.array(range(self.mult_class)).reshape((-1,1))

		# One Hot Encoder.
		self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
		self.one_hot_encoder.fit(self.labels_unique)

		# Weights & Biases visualization
		self.wandb_flag = wandb_flag

		# Instantiate model.
		self.build_model()
		
	# Model Inputs.
	def model_inputs(self):
		represenation_input_20x = tf.placeholder(dtype=tf.float32, shape=(None, 16, self.z_dim), name='represenation_input_20x')
		represenation_input_10x = tf.placeholder(dtype=tf.float32, shape=(None, 4, self.z_dim), name='represenation_input_10x')
		represenation_input_5x  = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='represenation_input_5x')
		label_input = tf.placeholder(dtype=tf.float32, shape=(1, self.mult_class), name='label_input')
		return represenation_input_20x, represenation_input_10x, represenation_input_5x, label_input

	# Feature Extractor Network 20x.
	def feature_extractor_20x(self, inputs, use, reuse, scope):
		print('Feature Extractor Network 20x:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_20x_%s' % scope, reuse=reuse):	

				interm = tf.reshape(interm, (-1, self.z_dim))	
				net = dense(inputs=interm, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				# net = ReLU(net)
				# net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
				interm = tf.reshape(interm, (-1, 16, self.z_dim))
		print()
		return interm

	# Feature Extractor Network 10x.
	def feature_extractor_10x(self, inputs, use, reuse, scope):
		print('Feature Extractor Network 10x:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_10x_%s' % scope, reuse=reuse):	

				interm = tf.reshape(interm, (-1, self.z_dim))	
				net = dense(inputs=interm, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				# net = ReLU(net)
				# net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
				interm = tf.reshape(interm, (-1, 4, self.z_dim))
		print()
		return interm

	# Feature Extractor Network 5x
	def feature_extractor_5x(self, inputs, use, reuse, scope):
		print('Feature Extractor Network 5x:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_5x_%s' % scope, reuse=reuse):		
				net = dense(inputs=inputs, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				# net = ReLU(net)
				# net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
			print()
		return interm
	
	# Attention Network 20x.
	def attention_20x(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network 20x:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_20x_%s' % scope, reuse=reuse):	

			net1 = tf.reshape(inputs, (-1, self.z_dim))	
			net1 = dense(inputs=net1, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = tf.reshape(inputs, (-1, self.z_dim))	
				net2 = dense(inputs=net2, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net2 = sigmoid(net2)
				net = tf.multiply(net1,net2)
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net_reshape = tf.reshape(net, (-1, 16, 1))
			weights = tf.nn.softmax(net_reshape, axis=1)
		print()
		return weights
	
	# Attention Network 10x.
	def attention_10x(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network 10x:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_10x_%s' % scope, reuse=reuse):	

			net1 = tf.reshape(inputs, (-1, self.z_dim))	
			net1 = dense(inputs=net1, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = tf.reshape(inputs, (-1, self.z_dim))	
				net2 = dense(inputs=net2, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net2 = sigmoid(net2)
				net = tf.multiply(net1,net2)
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net_reshape = tf.reshape(net, (-1, 4, 1))
			weights = tf.nn.softmax(net_reshape, axis=1)
		print()
		return weights

	def aggregate_20x_representations(self, interm, weights, reuse, scope):
		print('Aggregate Network 20x:', interm.shape[-1], 'Dimensions')
		with tf.variable_scope('aggregate_20x_%s' % scope, reuse=reuse):	
			weighted_rep   = interm*weights
			aggregated_rep = tf.reduce_sum(weighted_rep, axis=1)
		return aggregated_rep

	def aggregate_10x_representations(self, interm, weights, reuse, scope):
		print('Aggregate Network 10x:', interm.shape[-1], 'Dimensions')
		with tf.variable_scope('aggregate_10x_%s' % scope, reuse=reuse):	
			weighted_rep   = interm*weights
			aggregated_rep = tf.reduce_sum(weighted_rep, axis=1)
		return aggregated_rep


	def feature_extractor_comb(self, inputs, use, reuse, scope):
		print('Feature Extractor Network All Magnifications:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_comb_%s' % scope, reuse=reuse):		
				net = dense(inputs=inputs, out_dim=int(self.z_dim)*3, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=int(self.z_dim)*3, scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
			print()
		return interm

	# Attention Network.
	def attention(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network All Magnifications:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_%s' % scope, reuse=reuse):	

			net1 = dense(inputs=inputs, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = dense(inputs=inputs, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net2 = sigmoid(net2)
				net = tf.multiply(net1,net2)
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
			net = tf.transpose(net)
			weights = tf.nn.softmax(net)
			weights = tf.transpose(weights)
		print()

		return weights

	# Shared network for patient representations.
	def shared_network(self, inputs, use, reuse, scope):
		print('Share Network for Patient Represenation:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('shared_rep_%s' % scope, reuse=reuse):		
				net = dense(inputs=inputs, out_dim=inputs.shape[-1], scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=inputs.shape[-1], scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
			print()
		return interm
	
	# Classifier Network.
	def classifier(self, interm, weights, reuse, scope):
		print('Classifier Network:', interm.shape[-1], 'Dimensions')
		with tf.variable_scope('classifier_%s' % scope, reuse=reuse):		

			# Weight each sample.
			z = tf.reshape(tf.reduce_sum(weights*interm, axis=0), (-1,1))
			z = tf.transpose(z)

			# Consider Sigmoid here.
			logits = dense(inputs=z, out_dim=1, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=None, display=True)	
			prob   = sigmoid(logits)		

		print()			
		return prob, logits, z

	# Loss function.
	def loss(self, label, logits, prob):
		# Replace this with log likelihood.
		label = tf.argmax(label, axis=-1)
		label = tf.cast(label, tf.float32)
		label = tf.reshape(label, (1,1))
		prob = tf.clip_by_value(prob, clip_value_min=1e-5, clip_value_max=1.-1e-5)
		loss = -(label*tf.log(prob) + (1.-label)*tf.log(1.-prob))
		return loss

	# Optimizer.
	def optimization(self):
		# trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2).minimize(self.loss)
		# return trainer
		# Optimizer Initialization
		opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2)                                                                                                   

		# Gather trainable variables, create variables for accumulated gradients, assign zero value to accumulated gradients. 
		tvs = tf.trainable_variables()
		accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]                                        
		zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

		# Compute gradients w.r.t loss. 
		gvs = opt.compute_gradients(self.loss, tvs)
		accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

		# Applied accumulated gradients. 
		train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])  

		return zero_ops, accum_ops, train_step

	# Put together the model.
	def build_model(self):
		with tf.device('/gpu:0'):
			# Inputs.
			self.represenation_input_20x, self.represenation_input_10x, self.represenation_input_5x, self.label_input = self.model_inputs()
			# Feature Extractions.
			self.interm_5x  = self.feature_extractor_5x(inputs=self.represenation_input_5x,   use=True, reuse=False, scope=1)
			self.interm_10x = self.feature_extractor_10x(inputs=self.represenation_input_10x, use=True, reuse=False, scope=1)
			self.interm_20x = self.feature_extractor_20x(inputs=self.represenation_input_20x, use=True, reuse=False, scope=1)

			# Attention and aggregation of 20x.
			self.weights_20x         = self.attention_20x(inputs=self.interm_20x, use_gated=self.use_gated,  reuse=False, scope=1)
			self.aggregate_tiles_20x = self.aggregate_20x_representations(self.interm_20x, self.weights_20x, reuse=False, scope=1)

			# Attention and aggregation of 10x.
			self.weights_10x         = self.attention_10x(inputs=self.interm_10x, use_gated=self.use_gated,  reuse=False, scope=1)
			self.aggregate_tiles_10x = self.aggregate_10x_representations(self.interm_10x, self.weights_10x, reuse=False, scope=1)

			# Concatenate all magnification representations: 3*z_dim.
			self.rep_multimag        = tf.concat([self.interm_5x, self.aggregate_tiles_10x, self.aggregate_tiles_20x], axis=1)
			self.interm_multimag     = self.feature_extractor_comb(inputs=self.rep_multimag, use=True, reuse=False, scope=1)
			
			# Attention Network.
			self.weights = self.attention(inputs=self.interm_multimag, use_gated=self.use_gated, reuse=False, scope=1)
			# Classifier Network.
			self.prob, self.logits, self.z = self.classifier(interm=self.interm_multimag, weights=self.weights, reuse=False, scope=1)			
			# Loss and Optimizer.
			self.loss = self.loss(label=self.label_input, logits=self.logits, prob=self.prob)
			# self.trainer  = self.optimization()
			self.zero_ops, self.accum_ops, self.train_step = self.optimization()

	# Dirty change to handle cancer subtype.
	def process_label(self, patterns):
		if not isinstance(patterns, str):
			patterns = patterns[0]
		proc_labels = 0
		if 'Stage' in patterns:
			if self.subtype_process == patterns:
				proc_labels = 1
		else:
			if self.subtype_process in patterns:
				proc_labels = 1
		return proc_labels

	def keep_to_performance(self, run_metrics, top_fold_metrics, valid=False):
		train_metrics, valid_metrics, test_metrics     = run_metrics
		_,             valid_top,     test_top         = top_fold_metrics
		
		valid_top_accuracy,  _,  _,  valid_top_auc,  _,  _,  _,  _,  _ = valid_top
		valid_test_accuracy,  _,  _, valid_test_auc, _,  _,  _,  _,  _ = valid_metrics
		test_top_accuracy,  _,  _,  test_top_auc,  _,  _,  _,  _,  _ = test_top
		test_test_accuracy,  _,  _, test_test_auc, _,  _,  _,  _,  _ = test_metrics

		valid_top_accuracy  = valid_top_accuracy[0]
		valid_test_accuracy = valid_test_accuracy[0]
		valid_top_auc       = valid_top_auc[0]
		valid_test_auc      = valid_test_auc[0]
		test_top_accuracy   = test_top_accuracy[0]
		test_test_accuracy  = test_test_accuracy[0]
		test_top_auc        = test_top_auc[0]
		test_test_auc       = test_test_auc[0]

		# When doing train NYU / test TCGA only evaluate on test set. 
		if valid:
			if valid_test_auc > valid_top_auc:
				return run_metrics, True
			elif valid_test_auc == valid_top_auc:
				if valid_test_accuracy > valid_top_accuracy:
					return run_metrics, True
				elif valid_test_accuracy < valid_top_accuracy:
					return top_fold_metrics, False

		if test_test_auc > test_top_auc:
			return run_metrics, True
		elif test_test_auc == test_top_auc:
			if test_test_accuracy > test_top_accuracy:
				return run_metrics, True

		return top_fold_metrics, False

	# Save metrics per epoch.
	def keep_epoch_performance(self, data_out_path, epoch, train_metrics, valid_metrics, test_metrics, train_metrics_add=None, valid_metrics_add=None, test_metrics_add=None):
		# Handle directories and copies.
		results_path = os.path.join(data_out_path, 'results')
		epoch_path = os.path.join(results_path, 'epoch_%s' % epoch)
		check_epoch_path = os.path.join(epoch_path, 'checkpoints')
		checkpoint_path = os.path.join(results_path, '../checkpoints')
		os.makedirs(epoch_path)
		shutil.copytree(checkpoint_path, check_epoch_path)

		store_data(data=train_metrics, file_path=os.path.join(epoch_path,'train_metrics.pkl'))
		store_data(data=valid_metrics, file_path=os.path.join(epoch_path,'valid_metrics.pkl'))
		store_data(data=test_metrics, file_path=os.path.join(epoch_path,'test_metrics.pkl'))

		if test_metrics_add is not None:
			store_data(data=train_metrics_add, file_path=os.path.join(epoch_path,'train_metrics_add.pkl'))
			store_data(data=valid_metrics_add, file_path=os.path.join(epoch_path,'valid_metrics_add.pkl'))
			store_data(data=test_metrics_add, file_path=os.path.join(epoch_path,'test_metrics_add.pkl'))

	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, save_weights_flag=False, folds=5, hdf5_file_path_add=None, h_latent=True):
		num_folds = folds
		additional_loss = False
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)

		if 'LUAD' in self.model_name:
			val_split=True
		else:
			val_split=False
		
		# Get data.
		valid_flag = False
		total_latent_20x, total_latent_10x, total_latent_5x, total_orig_indices_20x, total_orig_indices_10x, total_orig_indices_5x, total_patterns, total_slides, total_tiles, total_institutions = gather_content_multi_magnification(hdf5_file_path, set_type=None, h_latent=h_latent)
		if hdf5_file_path_add is not None:
			total_latent_20x_2, total_latent_10x_2, total_latent_5x_2, total_orig_indices_20x_2, total_orig_indices_10x_2, total_orig_indices_5x_2, total_patterns_2, total_slides_2, total_tiles_2, total_institutions_2 = gather_content_multi_magnification(hdf5_file_path_add, set_type=None, h_latent=h_latent)
			valid_flag      = True
			additional_loss = True

		for subtype in self.labels_prediction:
			self.subtype_process = subtype
			subtype_data_out_path = os.path.join(data_out_path, subtype)

			top_performance_metrics = list()
			top_performance_metrics_add = list()

			# If possible use a static split to compare results.
			if 'NYU' in hdf5_file_path:
				train_slides    = np.unique(total_slides[:])
				random.shuffle(train_slides)
				valid_slides    = list()
				test_slides     = np.unique(total_slides[:])
				folds = [(train_slides, valid_slides, test_slides)]*num_folds
				valid_flag = False
			else:
				if 'Institutions' in self.model_name:
					pickle_file = '%s/utilities/files/folds_%s_Institutions_split.pkl' % ('/'.join(data_out_path.split('/')[:-5]), subtype)
					folds = split_folds_institutions(self, total_slides, total_patterns, total_institutions, val_split=val_split, num_folds=num_folds, random_shuffles=20, file_path=pickle_file)
				else:
					pickle_file = '%s/utilities/files/folds_%s_split.pkl' % ('/'.join(data_out_path.split('/')[:-5]), subtype)
					folds = split_folds(self, total_slides, total_patterns, num_folds=num_folds, val_split=val_split, file_path=pickle_file)


			for fold, sets in enumerate(folds):
				if self.wandb_flag:
					train_config = save_model_config_att(self)
					run_name = self.model_name + '-' + self.gan_model + '_fold' + str(fold)
					wandb.init(project='Attention MIL Pathology', entity='adalbertocquiros', name=run_name, config=train_config)

				train_slides, valid_slides, test_slides = sets
				random.shuffle(train_slides)

				top_fold_metrics     = [[[0, 0, 0]]*9]*3
				top_fold_metrics_add = [[[0, 0, 0]]*9]*3

				# Setup folder for outputs.
				losses      = ['Loss', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Train AUC', 'Validation AUC', 'Test AUC', 'Train Recall', 'Validation Recall', 'Test Recall', 'Train Precision', 'Validation Precision', 'Test Precision']
				fold_losses = ['Fold', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Train AUC', 'Validation AUC', 'Test AUC', 'Train Recall', 'Validation Recall', 'Test Recall', 'Train Precision', 'Validation Precision', 'Test Precision']
				fold_output_path = os.path.join(subtype_data_out_path, 'fold_%s' % fold)
				os.makedirs(fold_output_path)
				checkpoints, csvs = setup_output(data_out_path=fold_output_path, model_name=self.model_name, restore=False, additional_loss=additional_loss)
				setup_csvs(csvs=[csvs[0]], model=self, losses=losses)
				if hdf5_file_path_add is not None:
					setup_csvs(csvs=[csvs[1]], model=self, losses=losses)
				report_parameters(self, epochs=epochs, restore=False, data_out_path=fold_output_path)

				run_epochs = 0    
				saver = tf.train.Saver()
				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				with tf.Session(config=config) as session:
					session.run(tf.global_variables_initializer())
					
					for epoch in range(1, epochs+1):
						iter_grad = 0
						e_losses = list()
						for sample_indx in list(train_slides):
							if sample_indx == '': 
								print('Empty sample')
								continue

							# Gather index
							indxs = np.argwhere(total_slides[:]==sample_indx)[:,0]
							start_ind = sorted(indxs)[0]
							num_tiles_5x = indxs.shape[0]

							if num_tiles_5x < 100: 
								print('[INFO] Only train on slides with more than 100 samples:', sample_indx, num_tiles_5x)
								continue
							
							# Slide labels.
							label_instances = total_patterns[start_ind]
							label_batch = self.process_label(label_instances[0])
							label_batch = self.one_hot_encoder.transform([[label_batch]])
							
							# Slide latents for 20x and 5x.
							lantents_5x_batch  = total_latent_5x[start_ind:start_ind+num_tiles_5x]
							lantents_10x_batch = total_latent_10x[start_ind:start_ind+num_tiles_5x]
							lantents_20x_batch = total_latent_20x[start_ind:start_ind+num_tiles_5x]
							if lantents_20x_batch.shape[1] == 4:
								lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, self.z_dim))

							# Train iteration.
							feed_dict = {self.represenation_input_20x:lantents_20x_batch, self.represenation_input_10x:lantents_10x_batch, self.represenation_input_5x:lantents_5x_batch, self.label_input:label_batch}
							# _, epoch_loss = session.run([self.trainer, self.loss], feed_dict=feed_dict)
							_, epoch_loss = session.run([self.accum_ops, self.loss], feed_dict=feed_dict)
							iter_grad += 1

							if iter_grad == 32:
								session.run(self.train_step)
								iter_grad = 0
								session.run(self.zero_ops)
							e_losses.append(epoch_loss)
							run_epochs += 1
							# break

						# Compute accuracy for Training and Test sets on H5. 
						train_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, subset_slides=train_slides, slides=total_slides, patterns=total_patterns, labels=None, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, flat_20x=True)
						test_metrics  = compute_metrics_attention_multimagnifications(model=self, session=session, subset_slides=test_slides,  slides=total_slides, patterns=total_patterns, labels=None, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, flat_20x=True)
						train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set, _, top_w_train = train_metrics
						valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set, _ , _          = train_metrics
						test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set, _, _            = test_metrics
						if len(valid_slides) != 0:
							valid_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, subset_slides=valid_slides, slides=total_slides, patterns=total_patterns, labels=None, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, flat_20x=True)
							valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set, _, _ = valid_metrics
						else:
							valid_metrics = train_metrics
						save_unique_samples(data_out_path=fold_output_path, train_class_set=train_class_set, valid_class_set=valid_class_set, test_class_set=test_class_set, file_name='loss_samples.txt')

						# Save losses, accuracy, recall, and precision.
						loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
						update_csv(model=self, file=csvs[0], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)
						if self.wandb_flag and hdf5_file_path_add is None: 
							wandb.log({'Loss': loss_epoch[0], 'Train Acc': train_accuracy[0], 'Valid Acc': valid_accuracy[0], 'Test Acc': test_accuracy[0], 'Train AUC': train_auc[0], 'Valid AUC': valid_auc[0], 'Test AUC': test_auc[0]})

						# Keep track of top performance epoch.
						top_fold_metrics, improved_flag = self.keep_to_performance(run_metrics=[train_metrics, valid_metrics, test_metrics], top_fold_metrics=top_fold_metrics)
						if improved_flag:
							best_worst_perf = top_w_train

						# Save weight assignations per slide.
						if save_weights_flag and improved_flag:
							save_weights_attention_multimagnifications(model=self, set_type='combined', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides, patterns=total_patterns, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, labels=None, flat_20x=True, train_slides=train_slides, valid_slides=valid_slides)

						train_metrics_add, valid_metrics_add, test_metrics_add =None, None, None
						# Compute accuracy for Training and Validation sets on additional H5. 
						if hdf5_file_path_add is not None:
							train_metrics_add = compute_metrics_attention_multimagnifications(model=self, session=session, subset_slides=train_slides, slides=total_slides, patterns=total_patterns, labels=None, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, flat_20x=True)
							valid_metrics_add = compute_metrics_attention_multimagnifications(model=self, session=session, subset_slides=test_slides,  slides=total_slides, patterns=total_patterns, labels=None, latent_20x=total_latent_20x, latent_10x=total_latent_10x, latent_5x=total_latent_5x, flat_20x=True)
							test_metrics_add  = compute_metrics_attention_multimagnifications(model=self, session=session, slides=total_slides_2, patterns=total_patterns_2, labels=None, latent_20x=total_latent_20x_2, latent_10x=total_latent_10x_2, latent_5x=total_latent_5x_2, flat_20x=True)
							train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set, _, _ = train_metrics_add
							valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set, _, _ = valid_metrics_add
							test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set, _, _  = test_metrics_add
							save_unique_samples(data_out_path=fold_output_path, train_class_set=train_class_set, valid_class_set=valid_class_set, test_class_set=test_class_set, file_name='loss_samples_add.txt')

							# Save losses, accuracy, recall, and precision.
							loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
							update_csv(model=self, file=csvs[1], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)
							if self.wandb_flag: wandb.log({'Loss': loss_epoch[0], 'Train Acc': train_accuracy[0], 'Valid Acc': valid_accuracy[0], 'Test Acc': test_accuracy[0], 'Train AUC': train_auc[0], 'Valid AUC': valid_auc[0], 'Test AUC': test_auc[0]})

							# Keep track of top performance epoch.
							top_fold_metrics_add, improved_flag = self.keep_to_performance(run_metrics=[train_metrics_add, valid_metrics_add, test_metrics_add], top_fold_metrics=top_fold_metrics_add, valid=valid_flag)

							# Save weight assignations per slide.
							if save_weights_flag and improved_flag:
								save_weights_attention_multimagnifications(model=self, set_type='combined_add', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides,   patterns=total_patterns,   latent_20x=total_latent_20x,   latent_10x=total_latent_10x,   latent_5x=total_latent_5x,   labels=None, flat_20x=True, train_slides=train_slides, valid_slides=valid_slides)
								save_weights_attention_multimagnifications(model=self, set_type='test_add',     session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides_2, patterns=total_patterns_2, latent_20x=total_latent_20x_2, latent_10x=total_latent_10x_2, latent_5x=total_latent_5x_2, labels=None, flat_20x=True, train_slides=[], valid_slides=[])

						# Save epoch metrics 
						self.keep_epoch_performance(fold_output_path, epoch, train_metrics, valid_metrics, test_metrics, train_metrics_add, valid_metrics_add, test_metrics_add)

						# Save session.
						saver.save(sess=session, save_path=checkpoints)
						
				top_performance_metrics.append(top_fold_metrics)
				top_performance_metrics_add.append(top_fold_metrics_add)

				if self.wandb_flag:
					wandb.finish()
			save_fold_performance(data_out_path=subtype_data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics, file_name='folds_metrics.csv')
			save_fold_performance(data_out_path=subtype_data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics_add, file_name='folds_metrics_add.csv')



						

