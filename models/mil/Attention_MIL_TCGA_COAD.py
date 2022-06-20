from models.generative.discriminator import *
from models.generative.normalization import *
from models.generative.regularizers import *
from models.generative.activations import *
from models.generative.evaluation import *
from models.generative.optimizer import *
from models.generative.generator import *
from models.evaluation.features import *
from models.generative.encoder import *
from models.generative.utils import *
from data_manipulation.utils import *
from models.generative.tools import *
from models.generative.loss import *
from models.generative.ops import *
import tensorflow.compat.v1 as tf
from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py

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
		self.z_dim         = z_dim
		self.att_dim       = att_dim
		self.bag_size      = bag_size
		self.learning_rate = learning_rate
		self.beta_1        = beta_1
		self.beta_2        = beta_2
		self.model_name    = model_name
		
		self.mult_class    = 2
		self.labels_unique = np.array(range(self.mult_class)).reshape((-1,1))

		self.build_model()
		
	# StyleGAN inputs
	def model_inputs(self):
		represenation_input = tf.placeholder(dtype=tf.float32, shape=(None, self.z_dim), name='represenation_input')
		label_input = tf.placeholder(dtype=tf.float32, shape=(None, self.mult_class), name='label_input')

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
	def attention(self, inputs, reuse, scope):
		print('Attention:', inputs.shape[-1])
		with tf.variable_scope('attention_%s' % scope, reuse=reuse):		
			net1 = dense(inputs=inputs, out_dim=self.att_dim, scope=3, use_bias=True, spectral=False, init='xavier', regularizer=None, display=False)
			net1 = tanh(net1)
			net2 = dense(inputs=inputs, out_dim=self.att_dim, scope=4, use_bias=True, spectral=False, init='xavier', regularizer=None, display=False)
			net2 = sigmoid(net2)
			net = net1*net2
			net = dense(inputs=net, out_dim=1, scope=5, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)
			weigths = tf.nn.softmax(net, axis=0)
		return weigths
	
	# Classifier Network.
	def classifier(self, interm, weights):
		z = interm*weights
		z = tf.reshape(tf.reduce_sum(z, axis=0), (1,-1))
		logits = dense(inputs=z, out_dim=self.mult_class, scope=10, use_bias=True, spectral=False, init='xavier', regularizer=None, display=True)		
		prob = tf.nn.softmax(logits, axis=-1)					
		return prob, logits, z

	# Loss function.
	def loss(self, label, logits, prob):
		# Negative log likelihood
		# prob = tf.clip_by_value(prob, clip_value_min=1e-5, clip_value_max=1.-1e-5)
		# loss = -(label*tf.log(prob) + (1.-label)*tf.log(1.-prob))
		# prob = tf.clip_by_value(prob, clip_value_min=1e-5, clip_value_max=1.-1e-5)
		# loss = -(label*tf.log(prob) + (1.-label)*tf.log(1.-prob))
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,  logits=logits, axis=-1)
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
			self.weights = self.attention(inputs=self.interm, reuse=False, scope=1)

			# Classifier.
			self.prob, self.logits, self.z = self.classifier(interm=self.interm, weights=self.weights)
			self.clas = tf.to_float(tf.math.greater(self.prob, 0.5))
			
			# Losses.
			self.loss = self.loss(label=self.label_input, logits=self.logits, prob=self.prob)

			# Optimizers.
			self.trainer  = self.optimization()

	def compute_metrics(self, session, id_labels, quie_labels, latent, one_hot_encoder, return_weights=False, top_perc=0.01):
		# Variables to return.
		relevant_indeces = list()
		relevant_labels  = list()
		relevant_patches = list()
		relevant_weights = list()

		# Removing 'None'
		unique_ids    = list(np.unique(id_labels))
		unique_ids.remove('None')
		pred_set      = np.ones((len(unique_ids)))*10
		class_set     = np.ones((len(unique_ids)))*20

		# Iterate through slides.
		i = 0
		for pat_id in np.unique(id_labels):
			if pat_id == 'None': continue
			# Gather inde
			indxs = np.argwhere(id_labels[:]==pat_id)[:,0]
			label_batch_int = quie_labels[indxs[0]]
			label_batch = one_hot_encoder.transform([label_batch_int])
			random.shuffle(indxs)
			indxs = sorted(indxs[:10000])
			latents_batch = latent[indxs, :]
			feed_dict = {self.represenation_input:latents_batch}
			if return_weights:
				pred_batch, weights = session.run([self.clas, self.weights], feed_dict=feed_dict)
				ind = np.argsort(weights.reshape((1,-1)))[0,:]
			else:
				pred_batch = session.run([self.clas], feed_dict=feed_dict)[0]

			pred_set[i] = np.argmax(pred_batch)
			class_set[i] = label_batch_int
			
			if return_weights and (pred_set[i]==class_set[i]):
				top_patches = int(len(indxs)*top_perc)
				if top_patches == 0: top_patches += 1
				latents_sample = latents_batch[ind[-top_patches:]]
				indeces_sample = np.array(indxs)[ind[-top_patches:]]
				labels_sample  = np.ones((latents_sample.shape[0],1))*pred_set[i]
				weights_sample = weights[ind[-top_patches:]]

				relevant_patches.append(latents_sample)
				relevant_labels.append(labels_sample.reshape((-1,1)))
				relevant_indeces.append(indeces_sample.reshape((-1,1)))
				relevant_weights.append(weights_sample.reshape((-1,1)))
			i += 1

		accuracy  = np.round(accuracy_score(y_true=class_set,  y_pred=pred_set), 3)
		recall    = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 3)
		precision = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 3)
		auc_                 = np.round(roc_auc_score(class_set, pred_set), 3)

		if not return_weights:
			return accuracy, recall, precision, auc_
		else:
			relevant_patches = np.vstack(relevant_patches)
			relevant_labels = np.vstack(relevant_labels)
			relevant_indeces = np.vstack(relevant_indeces)
			relevant_weights = np.vstack(relevant_weights)
			return [accuracy, recall, precision, auc_], [relevant_patches, relevant_labels, relevant_indeces, relevant_weights]

	def save_relevant(self, relevant, output_path, set_type):
		relevant_patches, relevant_labels, relevant_indeces, relevant_weights = relevant
		dt = h5py.special_dtype(vlen=str)
		hdf5_path = os.path.join(output_path, 'hdf5_relevant_tiles_%s.h5' % set_type)
		with h5py.File(hdf5_path, mode='w') as hdf5_content:   
			latent_storage = hdf5_content.create_dataset(name='latent', shape=relevant_patches.shape,     dtype=np.float32)
			label_storage  = hdf5_content.create_dataset(name='label',  shape=relevant_labels.shape,      dtype=np.float32)
			ind_storage    = hdf5_content.create_dataset(name='indece', shape=relevant_indeces.shape,     dtype=np.float32)
			weight_storage = hdf5_content.create_dataset(name='weight', shape=relevant_weights.shape,     dtype=np.float32)

			for i in range(relevant_patches.shape[0]):
				latent_storage[i, :] = relevant_patches[i, :] 
				label_storage[i] = relevant_labels[i]
				ind_storage[i] = relevant_indeces[i]
				weight_storage[i] = relevant_weights[i]

	def gather_content(self, hdf5_path, set_type):
		# Open file for data manipulation. 
		hdf5_content = h5py.File(hdf5_path, mode='r')
		latent   = hdf5_content['images_h_latent']
		file_name   = hdf5_content['file_name']
		return latent, file_name

	def read_and_cross_check_labels(self, csv_file):
	    assos = dict()
	    i = 0
	    with open(csv_file, 'r') as content:
	        for line in content:
	            i += 1
	            if i == 1:
	            	continue
	            _, id_slide, quie = line.replace('\n','').split(',')
	            assos[id_slide] = float(str(quie))
	    return assos
	    
	def return_quiescence_labels(self, id_quie, file_names):
		quie_labels = np.ones((file_names.shape[0], 1))*3
		id_labels = list()
		no_label = list()
		for i in range(file_names.shape[0]):
			fields = str(file_names[i]).split('-')
			if len(fields) < 3:
				print(i, fields, file_names[i])
				id_labels.append('None')
				continue
			pat_id = fields[2]
			if pat_id not in id_quie:
				id_labels.append('None')
				if pat_id not in no_label:
					no_label.append(pat_id)
					print('ID not found in file:', pat_id)
				continue
			quie_labels[i] = id_quie[pat_id]
			id_labels.append(pat_id)
		return quie_labels, np.array(id_labels)

	# Train function.
	def train(self, epochs, hdf5_file_path, data_out_path, folds=10):
		if os.path.isdir(data_out_path):
			shutil.rmtree(data_out_path)
		
		# Get data.
		train_latent, train_file = self.gather_content(hdf5_file_path,                               set_type='train')
		valid_latent, valid_file = self.gather_content(hdf5_file_path.replace('train','validation'), set_type='valid')
		test_latent,  test_file  = self.gather_content(hdf5_file_path.replace('train','test'),       set_type='test')

		# Load CSV with labels.
		labels  = os.path.join(hdf5_file_path.split('/h224')[0], 'COAD_dream_complex_score.csv')
		id_quie = self.read_and_cross_check_labels(labels)
		quie_labels_train, id_labels_train = self.return_quiescence_labels(id_quie, train_file)
		quie_labels_valid, id_labels_valid = self.return_quiescence_labels(id_quie, valid_file)
		quie_labels_test,  id_labels_test  = self.return_quiescence_labels(id_quie, test_file)

		# 10-Fold.
		for fold in range(folds):

			# Setup folder for outputs.
			losses = ['Loss', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Train AUC', 'Validation AUC', 'Test AUC', 'Train Recall', 'Validation Recall', 'Test Recall', 'Train Precision', 'Validation Precision', \
					  'Test Precision']
			fold_output_path = os.path.join(data_out_path, 'fold_%s' % fold)
			os.makedirs(fold_output_path)
			checkpoints, csvs = setup_output(data_out_path=fold_output_path, model_name=self.model_name, restore=False)
			setup_csvs(csvs=csvs, model=self, losses=losses)
			report_parameters(self, epochs=epochs, restore=False, data_out_path=fold_output_path)

			# One Hot Encoder.
			from sklearn.preprocessing import OneHotEncoder
			one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
			one_hot_encoder.fit(self.labels_unique)

			run_epochs = 0    
			saver = tf.train.Saver()
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			with tf.Session(config=config) as session:
				session.run(tf.global_variables_initializer())
				
				for epoch in range(1, epochs+1):
					
					e_losses = list()
					for pat_id in np.unique(id_labels_train):
						if pat_id == 'None': continue
    					# Gather inde
						indxs = np.argwhere(id_labels_train[:]==pat_id)[:,0]
						label_batch = quie_labels_train[indxs[0]]
						label_batch = one_hot_encoder.transform([label_batch])
						random.shuffle(indxs)
						indxs = sorted(indxs[:10000])
						latents_batch = train_latent[indxs, :]

						# Train iteration.
						feed_dict = {self.represenation_input:latents_batch, self.label_input:label_batch}
						_, epoch_loss = session.run([self.trainer, self.loss], feed_dict=feed_dict)
						e_losses.append(epoch_loss)
						run_epochs += 1
						# break

					# Compute accuracy for Training and Validation sets.
					train_accuracy, train_recall, train_precision, train_auc  = self.compute_metrics(session=session, id_labels=id_labels_train, quie_labels=quie_labels_train, latent=train_latent, one_hot_encoder=one_hot_encoder)
					valid_accuracy, valid_recall, valid_precision, valid_auc  = self.compute_metrics(session=session, id_labels=id_labels_valid, quie_labels=quie_labels_valid, latent=valid_latent, one_hot_encoder=one_hot_encoder)
					test_accuracy,  test_recall,  test_precision , test_auc   = self.compute_metrics(session=session, id_labels=id_labels_test,  quie_labels=quie_labels_test,  latent=test_latent,  one_hot_encoder=one_hot_encoder)

					# Save losses, accuracy, recall, and precision.
					loss_epoch = [np.mean(e_losses), train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
					update_csv(model=self, file=csvs[0], variables=loss_epoch, epoch=epoch, iteration=run_epochs, losses=losses)

					# Save session.
					saver.save(sess=session, save_path=checkpoints)

				# Save relevant tiles for the outcome.
				_, train_relevant = self.compute_metrics(session=session, id_labels=id_labels_train, quie_labels=quie_labels_train, latent=train_latent, one_hot_encoder=one_hot_encoder, return_weights=True, top_perc=0.001)
				_, valid_relevant = self.compute_metrics(session=session, id_labels=id_labels_valid, quie_labels=quie_labels_valid, latent=valid_latent, one_hot_encoder=one_hot_encoder, return_weights=True, top_perc=0.001)
				_, test_relevant  = self.compute_metrics(session=session, id_labels=id_labels_test,  quie_labels=quie_labels_test,  latent=test_latent,  one_hot_encoder=one_hot_encoder, return_weights=True, top_perc=0.001)
				self.save_relevant(relevant=train_relevant, output_path=os.path.join(fold_output_path, 'results'), set_type='train')
				self.save_relevant(relevant=valid_relevant, output_path=os.path.join(fold_output_path, 'results'), set_type='valid')
				self.save_relevant(relevant=test_relevant,  output_path=os.path.join(fold_output_path, 'results'), set_type='test')
				

