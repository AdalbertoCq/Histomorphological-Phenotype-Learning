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
				z_dim,                             # Latent space dimensionality for projections.
				att_dim,					       # Attention network dimesionality.
				init='xavier',                     # Network initializer.
				bag_size=10000,                    # Maximum number of instances for a bag prediction.
				learning_rate=0.0005, 		       # Learning rate for Deep Attention MIL framework.
				beta_1=0.9,                        # Beta 1 value for Adam optimizer.
				beta_2=0.999,                      # Beta 2 value for Adam optimizer.
				use_gated=True,                    # Use gated attention.
				model_name='Attention_MIL_Histo',  # Name of the Deep Attention MIL run.
				gan_model='SSL_model',      	   # Name of the representation learning model.
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
		represenation_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim),      name='represenation_input')
		label_input         = tf.placeholder(dtype=tf.float32, shape=(None, self.mult_class), name='label_input')
		return represenation_input, label_input

	# Feature Extractor Network.
	def feature_extractor(self, inputs, use, reuse, scope):
		print('Feature Extractor Network:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('feature_extractor_%s' % scope, reuse=reuse):
				net = dense(inputs=inputs, out_dim=int(self.z_dim), scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				# net = ReLU(net)
				# net = dense(inputs=net,    out_dim=int(self.z_dim), scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
			print()

		return interm

	# Attention Network.
	def attention(self, inputs, reuse, scope, use_gated=True):
		print('Attention Network:', inputs.shape[-1], 'Dimensions')
		with tf.variable_scope('attention_%s' % scope, reuse=reuse):

			#
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
			weights = tf.nn.softmax(net, axis=-1)
			weights = tf.transpose(weights)

		return weights

	# Classifier Network.
	def classifier(self, interm, weights, reuse, scope):
		print('Classifier Network:', interm.shape[-1], 'Dimensions')
		with tf.variable_scope('classifier_%s' % scope, reuse=reuse):

			# Weight each sample.
			z = tf.reshape(tf.reduce_sum(weights*interm, axis=0), (-1,1))
			z = tf.transpose(z)

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
		return loss

	# Optimizer.
	def optimization(self):
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
			self.represenation_input, self.label_input = self.model_inputs()
			# Feature Extraction Network.
			self.interm = self.feature_extractor(inputs=self.represenation_input, use=True, reuse=False, scope=1)

		with tf.device('/gpu:0'):
			# Attention Network.
			self.weights = self.attention(inputs=self.interm, use_gated=self.use_gated, reuse=False, scope=1)

		with tf.device('/gpu:0'):
			# Classifier Network.
			self.prob, self.logits, self.z = self.classifier(interm=self.interm, weights=self.weights, reuse=False, scope=1)
			# Loss and Optimizer.
			self.loss = self.loss(label=self.label_input, logits=self.logits, prob=self.prob)
			# self.trainer  = self.optimization()
			self.zero_ops, self.accum_ops, self.train_step = self.optimization()

	# Dirty change to handle cancer subtype.
	def process_label(self, patterns):

		if not isinstance(patterns, str):
			patterns = patterns[0]

		if self.subtype_process in patterns:
			proc_labels = 1
		else:
			proc_labels = 0

		return proc_labels

	def keep_to_performance(self, run_metrics, top_fold_metrics):
		train_metrics, valid_metrics, test_metrics     = run_metrics
		_,             _,             test_top         = top_fold_metrics
		top_accuracy,  _,  _,  top_auc,  _,  _,  _,  _, _ = test_top
		test_accuracy,  _,  _, test_auc, _,  _,  _,  _, _ = test_metrics

		top_accuracy  = top_accuracy[0]
		test_accuracy = test_accuracy[0]
		top_auc       = top_auc[0]
		test_auc      = test_auc[0]

		# if test_accuracy > top_accuracy:
		# 	return run_metrics, True
		# elif test_accuracy == top_accuracy:
		# 	if test_auc >top_auc:
		# 		return run_metrics, True

		if test_auc >top_auc:
			return run_metrics, True
		elif test_auc == top_auc:
			if test_accuracy > top_accuracy:
				return run_metrics, True
		return top_fold_metrics, False

	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, folds=5, hdf5_file_path_add=None, h_latent=True, save_weights_flag=False):
		num_folds       = folds
		additional_loss = False
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)

		if 'LUAD' in self.model_name:
			val_split=True
		else:
			val_split=False

		# Get data.
		total_latent, total_labels, total_patterns, total_slides, total_tiles, total_institutions = gather_content(hdf5_file_path, set_type='combined', h_latent=h_latent)
		if hdf5_file_path_add is not None:
			total_latent_2, total_labels_2, total_patterns_2, total_slides_2, total_tiles_2, _ = gather_content(hdf5_file_path_add, set_type='combined', h_latent=h_latent)
			additional_loss = True

		for subtype in self.labels_prediction:
			self.subtype_process = subtype
			subtype_data_out_path = os.path.join(data_out_path, subtype)

			top_performance_metrics = list()
			top_performance_metrics_add = list()

			# If possible use a static split to compare results.
			if 'Institutions' in self.model_name:
				pickle_file = '%s/utilities/files/folds_%s_Institutions_split.pkl' % ('/'.join(data_out_path.split('/')[:-5]), subtype)
				folds = split_folds_institutions(self, total_slides, total_patterns, total_institutions, val_split=val_split, num_folds=num_folds, random_shuffles=20, file_path=pickle_file)
			elif 'recurrence' in self.model_name:
				pickle_file = '%s/utilities/files/NYU_PFS_folds.pkl' % ('/'.join(data_out_path.split('/')[:-5]))
				folds = split_folds_institutions(self, total_slides, total_patterns, total_institutions, val_split=val_split, num_folds=num_folds, random_shuffles=20, file_path=pickle_file)
			else:
				pickle_file = '%s/utilities/files/folds_%s_split.pkl' % ('/'.join(data_out_path.split('/')[:-5]), subtype)
				folds = split_folds(self, total_slides, total_patterns, num_folds=num_folds, val_split=val_split, file_path=pickle_file)

			# Random initializations.
			for fold, sets in enumerate(folds):
				if self.wandb_flag:
					train_config = save_model_config_att(self)
					run_name = self.model_name + '-' + self.gan_model + '_fold' + str(fold)
					wandb.init(project='Attention MIL Pathology Individual', entity='adalbertocquiros', name=run_name, config=train_config)

				train_slides, valid_slides, test_slides = sets
				random.shuffle(train_slides)

				top_fold_metrics = [[[0, 0, 0]]*9]*3
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
							# Gather slide indices.

							indxs = np.argwhere(total_slides[:]==str(sample_indx))[:,0]
							if indxs.shape[0] < 50:
								print('[INFO] Only train on slides with more than 50 samples:', sample_indx, indxs.shape[0])
								continue

							# Get label str and transform into int.
							label_instances = total_patterns[indxs[0]]
							label_batch = self.process_label(label_instances)
							label_batch = self.one_hot_encoder.transform([[label_batch]])
							# Get latents.
							lantents_batch = total_latent[indxs, :]

							# Train iteration.
							feed_dict = {self.represenation_input:lantents_batch, self.label_input:label_batch}
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
						train_metrics = compute_metrics_attention(model=self, session=session, subset_slides=train_slides, slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent)
						test_metrics  = compute_metrics_attention(model=self, session=session, subset_slides=test_slides,  slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent)
						train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set, _, top_w_train = train_metrics
						valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set, _, _           = train_metrics
						test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set, _, _            = test_metrics
						if len(valid_slides) != 0:
							valid_metrics = compute_metrics_attention(model=self, session=session, subset_slides=valid_slides, slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent)
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

						# Save weight assignations per slide.
						if improved_flag and save_weights_flag:
							save_weights_attention(model=self, set_type='combined', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent, train_slides=train_slides, valid_slides=valid_slides)

						# Compute accuracy for Training and Validation sets on additional H5.
						if hdf5_file_path_add is not None:
							train_metrics = compute_metrics_attention(model=self, session=session, subset_slides=train_slides, slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent)
							valid_metrics = compute_metrics_attention(model=self, session=session, subset_slides=test_slides,  slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent)
							test_metrics  = compute_metrics_attention(model=self, session=session, slides=total_slides_2, patterns=total_patterns_2, labels=total_labels_2, latent=total_latent_2)
							train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set, _, _ = train_metrics
							valid_accuracy, valid_recall, valid_precision, valid_auc, valid_class_set, valid_pred_set, valid_prob_set, _, _ = valid_metrics
							test_accuracy,  test_recall,  test_precision,  test_auc,  test_class_set,  test_pred_set,  test_prob_set, _, _  = test_metrics
							save_unique_samples(data_out_path=fold_output_path, train_class_set=train_class_set, valid_class_set=valid_class_set, test_class_set=test_class_set, file_name='loss_samples_add.txt')

							# Save losses, accuracy, recall, and precision.
							loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
							update_csv(model=self, file=csvs[1], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)
							if self.wandb_flag: wandb.log({'Loss': loss_epoch[0], 'Train Acc': train_accuracy[0], 'Valid Acc': valid_accuracy[0], 'Test Acc': test_accuracy[0], 'Train AUC': train_auc[0], 'Valid AUC': valid_auc[0], 'Test AUC': test_auc[0]})

							# Keep track of top performance epoch.
							top_fold_metrics_add, improved_flag = self.keep_to_performance(run_metrics=[train_metrics, train_metrics, test_metrics], top_fold_metrics=top_fold_metrics_add)

							# Save weight assignations per slide.
							if improved_flag and save_weights_flag:
								save_weights_attention(model=self, set_type='combined_add', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides, patterns=total_patterns, labels=total_labels, latent=total_latent, train_slides=train_slides, valid_slides=valid_slides)
								save_weights_attention(model=self, set_type='valid_add', session=session, output_path=os.path.join(fold_output_path, 'results'), slides=total_slides_2, patterns=total_patterns_2, labels=total_labels_2, latent=total_latent_2, train_slides=[], valid_slides=[])

						# Save session.
						saver.save(sess=session, save_path=checkpoints)
						# break

				top_performance_metrics.append(top_fold_metrics)
				top_performance_metrics_add.append(top_fold_metrics_add)

				if self.wandb_flag:
					wandb.finish()
			save_fold_performance(data_out_path=subtype_data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics, file_name='folds_metrics.csv')
			save_fold_performance(data_out_path=subtype_data_out_path, fold_losses=fold_losses, folds_metrics=top_performance_metrics_add, file_name='folds_metrics_add.csv')
