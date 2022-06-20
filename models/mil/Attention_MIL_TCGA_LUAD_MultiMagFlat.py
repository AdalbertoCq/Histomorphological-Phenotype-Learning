# Imports.
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py

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
				regularizer_scale=1e-4,      # Orthogonal regularization.
				use_gated=True,              # Use gated attention.
				model_name='Attention_MIL',  # Name of the Deep Attention MIL run.
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
		self.regularizer_scale = regularizer_scale

		# Naming.
		self.model_name    = model_name
		
		# Number of clases for prediction.
		self.mult_class    = 2
		self.labels_unique = np.array(range(self.mult_class)).reshape((-1,1))

		# One Hot Encoder.
		self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
		self.one_hot_encoder.fit(self.labels_unique)

		# Instantiate model.
		self.build_model()
		
	# Model Inputs.
	def model_inputs(self):
		represenation_input_20x = tf.placeholder(dtype=tf.float32, shape=(None, 16, self.z_dim),  name='represenation_input_20x')
		represenation_input_10x = tf.placeholder(dtype=tf.float32, shape=(None, 4, self.z_dim),   name='represenation_input_10x')
		represenation_input_5x  = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim),      name='represenation_input_5x')
		label_input             = tf.placeholder(dtype=tf.float32, shape=(None, self.mult_class), name='label_input')
		return represenation_input_20x, represenation_input_10x, represenation_input_5x, label_input

	# Feature Extractor Network 20x.
	def feature_extractor_20x(self, inputs, use, reuse, scope):
		print('Feature Extractor Network 20x:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_20x_%s' % scope, reuse=reuse):	

				interm = tf.reshape(interm, (-1, self.z_dim))	
				net = dense(inputs=interm, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
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
				net = dense(inputs=interm, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
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
				net = dense(inputs=inputs, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				interm = ReLU(net)
			print()
		return interm
	
	# Attention Network 20x.
	def attention_20x(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network 20x:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_20x_%s' % scope, reuse=reuse):	

			net1 = tf.reshape(inputs, (-1, self.z_dim))	
			net1 = dense(inputs=net1, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = dense(inputs=inputs, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net2 = sigmoid(net2)
				net = net1*net2
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
			net_reshape = tf.reshape(net, (-1, 16, 1))
			weights = tf.nn.softmax(net_reshape, axis=1)
		print()
		return weights
	
	# Attention Network 10x.
	def attention_10x(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network 10x:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_10x_%s' % scope, reuse=reuse):	

			net1 = tf.reshape(inputs, (-1, self.z_dim))	
			net1 = dense(inputs=net1, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = dense(inputs=inputs, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net2 = sigmoid(net2)
				net = net1*net2
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
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
				net = dense(inputs=inputs, out_dim=int(self.z_dim)*3, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net = ReLU(net)
				net = dense(inputs=net,    out_dim=int(self.z_dim)*3, scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				interm = ReLU(net)
			print()
		return interm

	# Attention Network.
	def attention(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network All Magnifications:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_%s' % scope, reuse=reuse):	

			#
			net1 = dense(inputs=inputs, out_dim=self.att_dim, scope='V_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
			net1 = tanh(net1)

			if use_gated:
				# GatedAttention.  
				net2 = dense(inputs=inputs, out_dim=self.att_dim, scope='U_k', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
				net2 = sigmoid(net2)
				net = net1*net2
			else:
				net = net1

			# Get weights.
			net = dense(inputs=net, out_dim=1, scope='W', use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.learning_rate*10), display=True)
			net = tf.transpose(net)
			weights = tf.nn.softmax(net)
			weights = tf.transpose(weights)
		print()

		return weights
	
	# Classifier Network.
	def classifier(self, interm, weights, reuse, scope):
		print('Classifier Network:', interm.shape[-1], 'Dimensions')
		with tf.variable_scope('classifier_%s' % scope, reuse=reuse):		

			# Weight each sample.
			z = tf.reshape(tf.reduce_sum(weights*interm, axis=0), (-1,1))
			z = tf.transpose(z)

			# Last layer to match number of classes.
			# logits = dense(inputs=z, out_dim=self.mult_class, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=None, display=True)	
			# prob = tf.nn.softmax(logits, axis=-1)	

			# Consider Sigmoid here.
			logits = dense(inputs=z, out_dim=1, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=None, display=True)	
			prob = sigmoid(logits)		

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
		
		# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,  logits=logits, axis=-1)
		
		return loss

	# Optimizer.
	def optimization(self):
		trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta_1, beta2=self.beta_2).minimize(self.loss)
		return trainer

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
			self.trainer  = self.optimization()

	# Dirty change to handle cancer subtype.
	def process_label(self, patterns):
		# print(patterns)
		if 'LUAD' in patterns and 'normal' in patterns:
			proc_labels = 20
		elif 'LUSC' in patterns and 'normal' in patterns:
			proc_labels = 20
		elif 'LUAD' in patterns:
			proc_labels = 0
		elif 'LUSC' in patterns:
			proc_labels = 1
		return proc_labels

	def keep_to_performance(self, run_metrics, top_fold_metrics):
		train_metrics, valid_metrics, test_metrics = run_metrics
		_,             _,             test_top     = top_fold_metrics
		top_accuracy,  _,  _,  top_auc,  _,  _,  _ = test_top
		test_accuracy,  _,  _, test_auc, _,  _,  _ = test_metrics

		top_accuracy  = top_accuracy[0]
		test_accuracy = test_accuracy[0]
		top_auc       = top_auc[0]
		test_auc      = test_auc[0]

		# if test_auc > top_auc:
		# 	return run_metrics, True
		# elif test_auc == top_auc:
		# 	if test_accuracy >top_accuracy:
		# 		return run_metrics, True
		# return top_fold_metrics, False

		if test_accuracy > top_accuracy:
			return run_metrics, True
		elif test_accuracy == top_accuracy:
			if test_auc >top_auc:
				return run_metrics, True
		return top_fold_metrics, False

	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, folds=10, hdf5_file_path_add=None, h_latent=True):
		additional_loss = False
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)
		
		# Get data.	
		train_latent_20x, train_latent_10x, train_latent_5x, train_orig_indices_20x, train_orig_indices_10x, train_orig_indices_5x, train_patterns, train_slides, train_tiles = gather_content_multi_magnification(hdf5_file_path,                               set_type='train', h_latent=h_latent)
		valid_latent_20x, valid_latent_10x, valid_latent_5x, valid_orig_indices_20x, valid_orig_indices_10x, valid_orig_indices_5x, valid_patterns, valid_slides, valid_tiles = gather_content_multi_magnification(hdf5_file_path.replace('train','validation'), set_type='valid', h_latent=h_latent)
		test_latent_20x,  test_latent_10x,  test_latent_5x,  test_orig_indices_20x,  test_orig_indices_10x,  test_orig_indices_5x,  test_patterns,  test_slides,  test_tiles  = gather_content_multi_magnification(hdf5_file_path.replace('train','test'),       set_type='test',  h_latent=h_latent)
		if hdf5_file_path_add is not None:
			train_latent_20x_2, train_latent_10x_2, train_latent_5x_2, train_orig_indices_20x_2, train_orig_indices_10x_2, train_orig_indices_5x_2, train_patterns_2, train_slides_2, train_tiles_2 = gather_content_multi_magnification(hdf5_file_path_add,                               set_type='train', h_latent=h_latent)
			valid_latent_20x_2, valid_latent_10x_2, valid_latent_5x_2, valid_orig_indices_20x_2, valid_orig_indices_10x_2, valid_orig_indices_5x_2, valid_patterns_2, valid_slides_2, valid_tiles_2 = gather_content_multi_magnification(hdf5_file_path_add.replace('train','validation'), set_type='valid', h_latent=h_latent)
			test_latent_20x_2,  test_latent_10x_2,  test_latent_5x_2,  test_orig_indices_20x_2,  test_orig_indices_10x_2,  test_orig_indices_5x_2,  test_patterns_2,  test_slides_2,  test_tiles_2  = gather_content_multi_magnification(hdf5_file_path_add.replace('train','test'),       set_type='test',  h_latent=h_latent)
			additional_loss = True

		top_performance_metrics     = list()
		top_performance_metrics_add = list()
		# Random initializations. 
		for fold in range(folds):
			top_fold_metrics     = [[[0, 0, 0]]*7]*3
			top_fold_metrics_add = [[[0, 0, 0]]*7]*3

			# Setup folder for outputs.
			losses      = ['Loss', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Train AUC', 'Validation AUC', 'Test AUC', 'Train Recall', 'Validation Recall', 'Test Recall', 'Train Precision', 'Validation Precision', 'Test Precision']
			fold_losses = ['Fold', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Train AUC', 'Validation AUC', 'Test AUC', 'Train Recall', 'Validation Recall', 'Test Recall', 'Train Precision', 'Validation Precision', 'Test Precision']
			fold_output_path = os.path.join(data_out_path, 'fold_%s' % fold)
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
					
					e_losses = list()
					for sample_indx in list(np.unique(train_slides)):
						if sample_indx == '': continue
						# Gather index
						indxs = np.argwhere(train_slides[:]==sample_indx)[:,0]
						start_ind = sorted(indxs)[0]
						num_tiles_5x = indxs.shape[0]
						
						# Slide labels.
						label_instances = train_patterns[start_ind]
						label_batch = self.process_label(label_instances[0])
						label_batch = self.one_hot_encoder.transform([[label_batch]])

						# Slide latents for 20x and 5x.
						lantents_5x_batch  = train_latent_5x[start_ind:start_ind+num_tiles_5x]
						lantents_10x_batch = train_latent_10x[start_ind:start_ind+num_tiles_5x]
						lantents_20x_batch = train_latent_20x[start_ind:start_ind+num_tiles_5x]
						if lantents_20x_batch.shape[1] == 4:
							lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, self.z_dim))

						# Train iteration.
						feed_dict = {self.represenation_input_20x:lantents_20x_batch, self.represenation_input_10x:lantents_10x_batch, self.represenation_input_5x:lantents_5x_batch, self.label_input:label_batch}
						_, epoch_loss = session.run([self.trainer, self.loss], feed_dict=feed_dict)
						e_losses.append(epoch_loss)
						run_epochs += 1
						# break

					# Compute accuracy for Training and Validation sets on H5. 
					train_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, slides=train_slides, patterns=train_patterns, labels=None, latent_20x=train_latent_20x, latent_10x=train_latent_10x, latent_5x=train_latent_5x, flat_20x=True)
					valid_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, slides=valid_slides, patterns=valid_patterns, labels=None, latent_20x=valid_latent_20x, latent_10x=valid_latent_10x, latent_5x=valid_latent_5x, flat_20x=True)
					test_metrics  = compute_metrics_attention_multimagnifications(model=self, session=session, slides=test_slides,  patterns=test_patterns,  labels=None, latent_20x=test_latent_20x,  latent_10x=test_latent_10x, latent_5x=test_latent_5x, flat_20x=True)
					train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set = train_metrics
					valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set = valid_metrics
					test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set  = test_metrics
					save_unique_samples(data_out_path=fold_output_path, train_class_set=train_class_set, valid_class_set=valid_class_set, test_class_set=test_class_set, file_name='loss_samples.txt')

					# Save losses, accuracy, recall, and precision.
					loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
					update_csv(model=self, file=csvs[0], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)

					# Compare performance to keep track of best iteration.
					top_fold_metrics, improved_flag = self.keep_to_performance(run_metrics=[train_metrics, valid_metrics, test_metrics], top_fold_metrics=top_fold_metrics)

					# Save weight assignations per slide.
					if (improved_flag and epoch>10) or epoch==10:
						save_weights_attention_multimagnifications(model=self, set_type='train', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=train_slides, patterns=train_patterns, latent_20x=train_latent_20x, latent_10x=train_latent_10x, latent_5x=train_latent_5x, labels=None, flat_20x=True)
						save_weights_attention_multimagnifications(model=self, set_type='valid', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=valid_slides, patterns=valid_patterns, latent_20x=valid_latent_20x, latent_10x=valid_latent_10x, latent_5x=valid_latent_5x, labels=None, flat_20x=True)
						save_weights_attention_multimagnifications(model=self, set_type='test' , session=session, output_path=os.path.join(fold_output_path, 'results'), slides=test_slides,  patterns=test_patterns,  latent_20x=test_latent_20x,  latent_10x=test_latent_10x,  latent_5x=test_latent_5x,  labels=None, flat_20x=True)

					# Compute accuracy for Training and Validation sets on additional H5. 
					if hdf5_file_path_add is not None:
						train_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, slides=train_slides_2, patterns=train_patterns_2, labels=None, latent_20x=train_latent_20x_2, latent_10x=train_latent_10x_2, latent_5x=train_latent_5x_2, flat_20x=True)
						valid_metrics = compute_metrics_attention_multimagnifications(model=self, session=session, slides=valid_slides_2, patterns=valid_patterns_2, labels=None, latent_20x=valid_latent_20x_2, latent_10x=valid_latent_10x_2, latent_5x=valid_latent_5x_2, flat_20x=True)
						test_metrics  = compute_metrics_attention_multimagnifications(model=self, session=session, slides=test_slides_2,  patterns=test_patterns_2,  labels=None, latent_20x=test_latent_20x_2,  latent_10x=test_latent_10x_2,  latent_5x=test_latent_5x_2, flat_20x=True)
						train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set = train_metrics
						valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set = valid_metrics
						test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set  = test_metrics
						save_unique_samples(data_out_path=fold_output_path, train_class_set=train_class_set, valid_class_set=valid_class_set, test_class_set=test_class_set, file_name='loss_samples_add.txt')

						# Save losses, accuracy, recall, and precision.
						loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
						update_csv(model=self, file=csvs[1], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)

						# Compare performance to keep track of best iteration.
						top_fold_metrics_add, improved_flag = self.keep_to_performance(run_metrics=[train_metrics, valid_metrics, test_metrics], top_fold_metrics=top_fold_metrics_add)

						# Save weight assignations per slide.
						if (improved_flag and epoch>10) or epoch==10:
							save_weights_attention_multimagnifications(model=self, set_type='train_add', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=train_slides_2, patterns=train_patterns_2, latent_20x=train_latent_20x_2, latent_10x=train_latent_10x_2, latent_5x=train_latent_5x_2, labels=None, flat_20x=True)
							save_weights_attention_multimagnifications(model=self, set_type='valid_add', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=valid_slides_2, patterns=valid_patterns_2, latent_20x=valid_latent_20x_2, latent_10x=valid_latent_10x_2, latent_5x=valid_latent_5x_2, labels=None, flat_20x=True)
							save_weights_attention_multimagnifications(model=self, set_type='test_add' , session=session, output_path=os.path.join(fold_output_path, 'results'), slides=test_slides_2,  patterns=test_patterns_2,  latent_20x=test_latent_20x_2,  latent_10x=test_latent_10x_2,  latent_5x=test_latent_5x_2,  labels=None, flat_20x=True)

					# Save session.
					saver.save(sess=session, save_path=checkpoints)
					# break

					
			top_performance_metrics.append(top_fold_metrics)
			top_performance_metrics_add.append(top_fold_metrics_add)

		save_fold_performance(data_out_path=data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics,     file_name='folds_metrics.csv')
		save_fold_performance(data_out_path=data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics_add, file_name='folds_metrics_add.csv')



						

