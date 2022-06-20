# Imports.
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py

# Network related lib.
from models.networks.attention import *
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
		self.regularizer_scale = self.learning_rate*10


		# Naming.
		self.model_name    = model_name

		# DeepHit parameters.
		self.alpha = 1
		self.sigma = 1
		self.constrain_flag = True
		# 60 Month survival threshold.
		self.censored_th = 99

		
		# Number of time steps.
		self.time_steps    = self.censored_th + 1
		self.labels_unique = np.array(range(self.time_steps)).reshape((-1,1))

		# One Hot Encoder.
		self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
		self.one_hot_encoder.fit(self.labels_unique)
		# Batch size.
		self.batch_size = 16
		# Instantiate model.
		self.build_model()

		# Memory cache for patient representations and labels.
		# Rep cache
		self.patient_cache = np.zeros((self.batch_size-1, self.patient_rep.shape[1]))
		# Labels 
		self.patient_one_hot       = np.zeros((self.batch_size-1, self.time_steps))
		self.patient_one_hot_step  = np.zeros((self.batch_size-1, self.time_steps))
		self.patient_censor        = np.zeros((self.batch_size-1, 1))
		self.patient_label         = np.zeros((self.batch_size-1, 1))		

	# Model Inputs.
	def model_inputs(self):
		represenation_input_20x = tf.placeholder(dtype=tf.float32, shape=(None, 16, self.z_dim),               name='represenation_input_20x')
		represenation_input_10x = tf.placeholder(dtype=tf.float32, shape=(None, 4, self.z_dim),                name='represenation_input_10x')
		represenation_input_5x  = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim),                   name='represenation_input_5x')
		
		label_one_hot_ind       = tf.placeholder(dtype=tf.float32, shape=(1, self.time_steps),                 name='label_one_hot_ind')
		label_one_step_ind      = tf.placeholder(dtype=tf.float32, shape=(1, self.time_steps),                 name='label_one_step_ind')
		censored_ind            = tf.placeholder(dtype=tf.float32, shape=(1, 1),                               name='censored_ind')
		label_ind               = tf.placeholder(dtype=tf.float32, shape=(1, 1),                               name='label_ind')

		label_one_hot           = tf.placeholder(dtype=tf.float32, shape=(self.batch_size-1, self.time_steps), name='label_one_hot_batch')
		label_one_step          = tf.placeholder(dtype=tf.float32, shape=(self.batch_size-1, self.time_steps), name='label_one_step_batch')
		censored                = tf.placeholder(dtype=tf.float32, shape=(self.batch_size-1, 1),               name='censored_batch')
		labels                  = tf.placeholder(dtype=tf.float32, shape=(self.batch_size-1, 1),               name='labels_batch')
		patient_reps_batch      = tf.placeholder(dtype=tf.float32, shape=(self.batch_size-1, self.z_dim*3),    name='patient_reps_batch')
		return represenation_input_20x, represenation_input_10x, represenation_input_5x,  \
			   label_one_hot_ind, label_one_step_ind, censored_ind, label_ind, \
			   label_one_hot, label_one_step, censored, labels, patient_reps_batch

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

	# Cause survival network.
	def cause_survival_network(self, inputs, num_time_steps, use, reuse, scope):
		print('Cause Network:', inputs.shape[-1], 'Dimensions')
		interm = inputs
		if use:
			with tf.variable_scope('cause_survival_network_%s' % scope, reuse=reuse):		
				net = dense(inputs=inputs, out_dim=inputs.shape[-1], scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				net = ReLU(net)
				net = dense(inputs=inputs, out_dim=inputs.shape[-1], scope=2, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l2_reg(self.regularizer_scale), display=True)
				interm = ReLU(net)
			print()
		return interm

	# Loss 1 - Log-likelihood
	def loss_log_likelihood(self, event_time_prob, label_mask, label_mask_2, censored):
		def log(x):
			return tf.log(x + 1e-08)

		# Uncesored portion of the loss
		part_1 = tf.reduce_sum(log(event_time_prob*label_mask), axis=-1, keepdims=True)
		part_1 = (1-censored)*part_1

		part_2 = tf.reduce_sum(event_time_prob*label_mask, axis=-1, keepdims=True)
		part_2 = censored*log(part_2)

		loss_1 = -tf.reduce_mean(part_1 + part_2, axis=0)
		return loss_1

	# Loss 2 - Ranking loss.
	def loss_ranking(self, event_time_prob, labels, label_mask, label_mask_2):
		
		one_vector = tf.ones_like(labels, dtype=np.float32) 

		# Accu Risk until time of event.
		R      = tf.matmul(event_time_prob, tf.transpose(label_mask_2))
		diag_R = tf.reshape(tf.diag_part(R), [-1,1]) 
		
		# Relationship between Risks of patients.
		R      = tf.matmul(one_vector, tf.transpose(diag_R)) - R
		R      = tf.transpose(R)

		# Indicator function to find patients with s_i < s_j:
		# 		T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
		I_2 = tf.eye(num_rows=int(labels.shape[0]), num_columns=int(labels.shape[0])) 
		T   = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(labels)) - tf.matmul(labels, tf.transpose(one_vector))))
		T   = tf.matmul(I_2, T)

		interm = self.alpha*T*tf.exp(-R/self.sigma)

		loss_2 = tf.reduce_mean(interm)
		return loss_2

	# Loss function.
	def loss(self, event_time_prob, labels, label_mask, label_mask_2, censored):
		loss_1 = self.loss_log_likelihood(event_time_prob=event_time_prob, label_mask=label_mask, label_mask_2=label_mask_2, censored=censored)
		loss_2 = self.loss_ranking(event_time_prob, labels, label_mask, label_mask_2)
		loss = loss_1 + loss_2
		return loss, loss_1, loss_2

	# Optimizer.
	def optimization(self):

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
			self.represenation_input_20x, self.represenation_input_10x, self.represenation_input_5x, self.label_one_hot_ind, self.label_one_step_ind, self.censored_ind, self.label_ind, \
			self.label_one_hot_batch, self.label_one_step_batch, self.censored_batch, self.labels_batch, self.patient_reps_batch = self.model_inputs()
			
			# Attention Network for Multi-Magnifications.
			self.patient_rep_ind = attention_network(model=self, represenation_input_5x=self.represenation_input_5x, represenation_input_10x=self.represenation_input_10x, \
												 represenation_input_20x=self.represenation_input_20x, regularizer_scale=self.regularizer_scale, reuse=False, name='attention_network')

			# Buid batch from Memory cache.
			self.label_one_hot      = tf.concat([self.label_one_hot_ind,  self.label_one_hot_batch],  axis=0)
			self.label_one_step     = tf.concat([self.label_one_step_ind, self.label_one_step_batch], axis=0)
			self.censored           = tf.concat([self.censored_ind,       self.censored_batch],       axis=0)
			self.labels             = tf.concat([self.label_ind,          self.labels_batch],         axis=0)
			self.patient_rep        = tf.concat([self.patient_rep_ind,    self.patient_reps_batch],   axis=0)
			# From this point on we have a batch of patient representations.

			################### DeepHit portion of the model.
			# Shared Represenation Network.
			# self.shared_rep       = self.shared_network(inputs=self.patient_rep, use=True, reuse=False, scope=1)

			# Cause Specific Network.
			# self.cause_events_out = self.cause_survival_network(inputs=self.patient_rep, num_time_steps=self.z_dim, use=True, reuse=False, scope=1)

			# Fully connected layer and Softmax. 
			net             = dense(inputs=self.patient_rep, out_dim=self.time_steps, scope=1, use_bias=True, spectral=False, init='glorot_uniform', regularizer=l1_reg(self.regularizer_scale), display=True)
			self.event_time_prob = tf.nn.softmax(net, axis=1)
			# The event_time_prob is modeling the log likelihood.

			# Loss and Optimizer.
			self.loss, self.loss_1, self.loss_2 = self.loss(event_time_prob=self.event_time_prob, labels=self.labels, label_mask=self.label_one_hot, label_mask_2=self.label_one_step, censored=self.censored)
			# self.trainer  = self.optimization()
			self.zero_ops, self.accum_ops, self.train_step = self.optimization()


	# Dirty change to handle cancer subtype.
	def process_label(self, labels):
		survival_age = int(labels[0])
		censored     = labels[1]
		if survival_age > self.time_steps-1:
			survival_age = 99
		
		# One hot step.
		one_hot                  = np.zeros((1,self.time_steps))
		one_hot[0, survival_age] = 1

		# One hot step.
		mask_one  = np.ones((1, survival_age+1))
		mask_zero = np.zeros((1,self.time_steps-survival_age-1))
		one_hot_step = np.concatenate([mask_one, mask_zero], axis=1)

		# Survival reformat.
		survival_age = np.array(survival_age).reshape((1,1))
		
		# Censored reformat.
		censored = np.array(censored).reshape((1,1))
		
		return one_hot, one_hot_step, survival_age, censored

	def update_patient_caches(self, patient_rep, one_hot_step, one_hot, censored, label):
		new_patient_cache        = np.zeros(self.patient_cache.shape)
		new_patient_cache[0, :]  = patient_rep
		new_patient_cache[1:, :] = self.patient_cache[:-1, :]
		self.patient_cache       = new_patient_cache

		new_patient_one_hot_step       = np.zeros(self.patient_one_hot_step.shape)
		new_patient_one_hot_step[0,:]  = one_hot_step
		new_patient_one_hot_step[1:,:] = self.patient_one_hot_step[:-1, :]
		self.patient_one_hot_step      = new_patient_one_hot_step

		new_patient_one_hot       = np.zeros(self.patient_one_hot.shape)
		new_patient_one_hot[0,:]  = one_hot
		new_patient_one_hot[1:,:] = self.patient_one_hot[:-1, :]
		self.patient_one_hot      = new_patient_one_hot

		new_patient_censor       = np.zeros(self.patient_censor.shape)
		new_patient_censor[0,:]  = censored[0,:]
		new_patient_censor[1:,:] = self.patient_censor[:-1,:]
		self.patient_censor      = new_patient_censor

		new_patient_labels       = np.zeros(self.patient_label.shape)
		new_patient_labels[0,:]  = label
		new_patient_labels[1:,:] = self.patient_label[:-1,:]
		self.patient_label       = new_patient_labels

	def fill_cache(self, session, slides, patterns, latent_5x, latent_10x, latent_20x, subset_slides):

		# Gather Batch size patient samples and outcomes to use as Memory cache.
		random_slides = list(np.unique(subset_slides))
		random.shuffle(random_slides)
		fill_cache_iter = 0
		for sample_indx in random_slides:

			if fill_cache_iter == self.batch_size-1:
				break

			if sample_indx == '': continue
			# Gather index
			indxs = np.argwhere(slides[:]==sample_indx)[:,0]
			start_ind = sorted(indxs)[0]
			num_tiles_5x = indxs.shape[0]

			# Slide labels.
			labels = patterns[start_ind]
			one_hot, one_hot_step, label, censored = self.process_label(labels)

			if censored==1 and self.no_censored:
				continue

			# Slide latents for 20x and 5x.
			lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
			lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
			lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]
			if lantents_20x_batch.shape[1] == 4:
				lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, self.z_dim))

			feed_dict = {self.represenation_input_20x:lantents_20x_batch, self.represenation_input_10x:lantents_10x_batch, self.represenation_input_5x:lantents_5x_batch}
			patient_rep = session.run([self.patient_rep_ind], feed_dict=feed_dict)[0]

			b = np.sum((self.patient_label>=self.censored_th)*1)
			ratio = b/self.batch_size
			if not self.constrain_flag or ((ratio < 0.3 and label >= self.censored_th) or label<self.censored_th):
				self.update_patient_caches(patient_rep, one_hot_step, one_hot, censored, label)
				fill_cache_iter += 1

	def keep_to_performance(self, run_metrics, top_fold_metrics):
		# if (run_metrics[-1] > top_fold_metrics[-1]) and run_metrics[-2] < 0.4:
		if (run_metrics[-1] > top_fold_metrics[-1]):
			return run_metrics, True
		return top_fold_metrics, False

	def split_folds(self, total_slides, total_patterns, num_folds):
		all_slides      = np.unique(total_slides)

		positive_slides = list()
		negative_slides = list()
		for slide in all_slides:
			# Gather index
			indxs = np.argwhere(total_slides[:]==slide)[:,0]
			start_ind = sorted(indxs)[0]
			# Slide labels.
			labels = total_patterns[start_ind]
			one_hot, one_hot_step, label, censored = self.process_label(labels)

			if censored == 1:
				positive_slides.append(slide)
			else:
				negative_slides.append(slide)

		perct = 1.0/num_folds
		positive_test_size = int(len(positive_slides)*perct)
		negative_test_size = int(len(negative_slides)*perct)
		positive_set = set(positive_slides)
		negative_set = set(negative_slides)

		i = 0
		pos_test_folds = list()
		for i in range(num_folds):
			if i == num_folds-1:
				pos_test_folds.append(positive_slides[i*positive_test_size:])
				break
			pos_test_folds.append(positive_slides[i*positive_test_size:i*positive_test_size+positive_test_size])

		i = 0
		neg_test_folds = list()
		for i in range(num_folds):
			if i == num_folds-1:
				neg_test_folds.append(negative_slides[i*negative_test_size:])
				break
			neg_test_folds.append(negative_slides[i*negative_test_size:i*negative_test_size+negative_test_size])

		folds = list()
		for pos_test_set, neg_test_set in zip(pos_test_folds, neg_test_folds):
			train_set = list()
			test_set  = list()
			pos_train_set = list(positive_set.difference(set(pos_test_set)))
			neg_train_set = list(negative_set.difference(set(neg_test_set)))
			train_set.extend(pos_train_set)
			train_set.extend(neg_train_set)
			test_set.extend(pos_test_set)
			test_set.extend(neg_test_set)
			folds.append((np.array(train_set), np.array(test_set)))

		return folds

	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, folds=5, hdf5_file_path_add=None, h_latent=True):
		additional_loss = False
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)
		
		# Get data.
		total_latent_20x, total_latent_10x, total_latent_5x, _, _, _, total_patterns, total_slides, total_tiles = gather_content_multi_magnification(hdf5_file_path, set_type='combined', h_latent=h_latent)	
		if hdf5_file_path_add is not None:
			total_latent_20x_2, total_latent_10x_2, total_latent_5x_2, _, _, _, total_patterns_2, total_slides_2, total_tiles_2 = gather_content_multi_magnification(hdf5_file_path_add, set_type='combined', h_latent=h_latent)
			additional_loss = True 

		# Get number of folds cross-validation.
		folds = self.split_folds(total_slides, total_patterns, num_folds=folds)

		# Keep track of performance metrics.
		top_performance_metrics   = list()
		top_performance_metrics_2 = list()

		self.no_censored = True

		# Random initializations. 
		for fold, sets in enumerate(folds):
			# if fold != 0: continue
			train_slides, test_slides = sets
			top_fold_metrics   = [0.]*10
			top_fold_metrics_2 = [0.]*10

			# Setup folder for outputs.
			losses      = ['Train Loss Total', 'Train Loss Log-likelihood', 'Train Loss Ranking', 'Train C-th Index', 'Test Loss Total', 'Test Loss Log-likelihood', 'Test Loss Ranking', 'Test C-th Index']
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

				# Initialize to zero accumulated gradients.
				session.run(self.zero_ops)
				self.fill_cache(session, total_slides, total_patterns, total_latent_5x, total_latent_10x, total_latent_20x, subset_slides=train_slides)
				
				iter_slides = list(np.unique(train_slides))
				# random.shuffle(iter_slides)
				for epoch in range(1, epochs+1):
					
					iter_grad = 0
					for sample_indx in iter_slides:
						# Gather index
						indxs = np.argwhere(total_slides[:]==sample_indx)[:,0]
						start_ind = sorted(indxs)[0]
						num_tiles_5x = indxs.shape[0]
						
						# Slide labels.
						labels = total_patterns[start_ind]
						one_hot, one_hot_step, label, censored = self.process_label(labels)

						# Slide latents for 20x and 5x.
						lantents_5x_batch  = total_latent_5x[start_ind:start_ind+num_tiles_5x]
						lantents_10x_batch = total_latent_10x[start_ind:start_ind+num_tiles_5x]
						lantents_20x_batch = total_latent_20x[start_ind:start_ind+num_tiles_5x]
						if lantents_20x_batch.shape[1] == 4:
							lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, self.z_dim))

						# Train iteration.
						feed_dict    = {self.represenation_input_20x:lantents_20x_batch, self.represenation_input_10x:lantents_10x_batch, self.represenation_input_5x:lantents_5x_batch, \
										self.label_one_hot_ind:one_hot, self.label_one_step_ind:one_hot_step, self.censored_ind:censored, self.label_ind:label, \
										self.label_one_hot_batch:self.patient_one_hot, self.label_one_step_batch:self.patient_one_hot_step, self.censored_batch:self.patient_censor, \
										self.labels_batch:self.patient_label, self.patient_reps_batch:self.patient_cache}
						session.run([self.accum_ops], feed_dict=feed_dict)
						iter_grad += 1

						if iter_grad == self.batch_size:
							session.run(self.train_step)
							iter_grad = 0
							session.run(self.zero_ops)

						run_epochs += 1

						self.fill_cache(session, total_slides, total_patterns, total_latent_5x, total_latent_10x, total_latent_20x, subset_slides=train_slides)

						# break

					# Check performance of C-th Index
					metric_performance, data = get_survival_performance(model=self, session=session, slides=total_slides, patterns=total_patterns, latent_5x=total_latent_5x, latent_10x=total_latent_10x, latent_20x=total_latent_20x, train_slides=train_slides, test_slides=test_slides)
					
					# Save losses, accuracy, recall, and precision.
					loss_epoch = metric_performance[0] + metric_performance[1]
					update_csv(model=self, file=csvs[0], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)

					# Allow some burning period before checking
					if epochs > 14:

						# Keep track of top performance.
						top_fold_metrics, improved_flag = self.keep_to_performance(run_metrics=[epoch] + [run_epochs] + loss_epoch, top_fold_metrics=top_fold_metrics)

						if improved_flag:
 							save_survival_weights(model=self, session=session, slides=total_slides, patterns=total_patterns, latent_5x=total_latent_5x, latent_10x=total_latent_10x, latent_20x=total_latent_20x, \
												  output_path=os.path.join(fold_output_path, 'results'), subset_slides=[train_slides, test_slides])
					

					# Additional file performance: to review
					if hdf5_file_path_add is not None:
						# Check performance of C-th Index
						metric_performance_2, _ = get_survival_performance_add(model=self, session=session, slides=total_slides, patterns=total_patterns, latent_5x=total_latent_5x, latent_10x=total_latent_10x, latent_20x=total_latent_20x, train_slides=train_slides, \
									    							  		   slides_2=total_slides_2, patterns_2=total_patterns_2, latent_5x_2=total_latent_5x_2, latent_10x_2=total_latent_10x_2, latent_20x_2=total_latent_20x_2)
						# Save losses, accuracy, recall, and precision.
						loss_epoch = metric_performance_2[0] + metric_performance_2[1]
						update_csv(model=self, file=csvs[1], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)
						if improved_flag: top_fold_metrics_2 = [epoch] + [run_epochs] + loss_epoch

					# Save session.
					saver.save(sess=session, save_path=checkpoints)

			top_performance_metrics.append(top_fold_metrics)
			top_performance_metrics_2.append(top_fold_metrics_2)
		save_fold_performance_survival(data_out_path=data_out_path, fold_losses=losses, folds_metrics=top_performance_metrics,   file_name='folds_metrics.csv')
		save_fold_performance_survival(data_out_path=data_out_path, fold_losses=losses, folds_metrics=top_performance_metrics_2, file_name='folds_metrics_add.csv')
		save_survival_weights(model=self, session=session, slides=total_slides, patterns=total_patterns, latent_5x=total_latent_5x, latent_10x=total_latent_10x, latent_20x=total_latent_20x, output_path=os.path.join(fold_output_path, 'results'), subset_slides=[train_slides, test_slides], file_name='hdf5_attention_weights_last.h5')

