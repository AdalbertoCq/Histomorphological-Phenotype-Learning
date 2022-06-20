from data_manipulation.utils import *
from sklearn.metrics import *
import numpy as np
import random
import shutil
import h5py
import os


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def compute_metrics_attention(model, session, slides, patterns, latent, subset_slides=None, labels=None, return_weights=False, top_percent=0.1):
	# Prediction, True labels for metrics.
	prob_set         = list()
	pred_set         = list()
	class_set        = list()
	slide_set        = list()

	iter_slides = subset_slides
	if subset_slides is None:
		iter_slides = np.unique(slides)

	# Iterate through slides.
	i = 0
	for slide in iter_slides:
		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		if indxs.shape[0] < 50:
			print('[INFO] Only train on slides with more than 50 samples:', slide, indxs.shape[0])
			continue

		# Label processing for the tile.
		label_instances = patterns[indxs[0]]
		label_batch_int = model.process_label(label_instances)
		label_batch = model.one_hot_encoder.transform([[label_batch_int]])
		# Latents
		latents_batch = latent[indxs, :]

		# Run the model.
		feed_dict = {model.represenation_input:latents_batch}
		prob_batch = session.run([model.prob], feed_dict=feed_dict)[0]

		# In case we use sigmoid and not multi-class.
		if prob_batch.shape[-1] == 1:
			prob_batch = np.array([1-prob_batch, prob_batch])

		# Keep track of outcomes for slide.
		prob_set.append(prob_batch)
		pred_set.append(np.argmax(prob_batch))
		class_set.append(label_batch_int)
		slide_set.append(slide)

		i += 1

	# Reshape into np.array
	prob_set  = np.vstack(prob_set).reshape((-1,model.mult_class))
	pred_set  = np.vstack(pred_set)
	class_set = np.vstack(class_set)
	slide_set = np.vstack(slide_set)

	# Accuracy and Confusion Matrix.
	cm             = confusion_matrix(y_true=class_set, y_pred=pred_set)
	cm             = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	acc_per_class  = cm.diagonal()
	accuracy_total = balanced_accuracy_score(y_true=class_set,  y_pred=pred_set)
	accuracy       = np.round([accuracy_total] + list(acc_per_class), 2)
	# Recall
	recall         = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 2)
	# Precision
	precision      = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 2)
	# AUC.
	try:
		auc_per_class  = roc_auc_score(y_true=model.one_hot_encoder.transform(class_set.reshape((-1,1))), y_score=prob_set, average=None)
		# Macro, subject to data imbalance.
		auc_total      = np.mean(auc_per_class)
		auc_all        = np.round([auc_total] + list(auc_per_class), 2)
	except:
		auc_all = [None]
		for class_ in range(model.mult_class):
			try:
			    fpr, tpr, thresholds = roc_curve(model.one_hot_encoder.transform(class_set.reshape((-1,1)))[:, class_], prob_set[:, class_])
			    roc_auc = np.round(auc(fpr, tpr), 2)
			except:
				roc_auc = None
			auc_all.append(roc_auc)

	# Get the top worst performers
	diff         = np.abs(class_set-prob_set)
	inds         = np.argsort(diff, axis=0)
	top_nsamples = math.ceil(top_percent*class_set.shape[0])
	top_ind      = inds[-top_nsamples:]
	top_slides   = slide_set[top_ind]

	return accuracy, recall, precision, auc_all, class_set, pred_set, prob_set, slide_set, top_slides


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def compute_metrics_attention_multimagnifications(model, session, slides, patterns, latent_20x, latent_10x, latent_5x, subset_slides=None, labels=None, flat_20x=False, top_percent=0.1):

	# Prediction, True labels for metrics.
	prob_set         = list()
	pred_set         = list()
	class_set        = list()
	slide_set        = list()

	# Unique slides to iterate through.
	if subset_slides is None:
		unique_slides    = list(np.unique(slides))
	# Use specified slides: Histopology subtypes.
	else:
		unique_slides    = subset_slides

	# Iterate through slides.
	i = 0
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		if num_tiles_5x < 50: continue

		# Slide latents for 20x and 5x.
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]
		if flat_20x:
			lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, lantents_5x_batch.shape[-1]))

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])
		if 'Survival' in model.model_name and label_instances[1]==1:
			continue

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch = session.run([model.prob], feed_dict=feed_dict)[0]

		# In case we use sigmoid and not multi-class.
		if prob_batch.shape[-1] == 1:
			prob_batch = np.array([1-prob_batch, prob_batch])

		# Keep track of outcomes for slide.
		prob_set.append(prob_batch)
		pred_set.append(np.argmax(prob_batch))
		class_set.append(label_batch_int)
		slide_set.append(slide)

		i += 1

	# Reshape into np.array
	prob_set  = np.vstack(prob_set).reshape((-1,model.mult_class))
	pred_set  = np.vstack(pred_set)
	class_set = np.vstack(class_set)
	slide_set = np.vstack(slide_set)

	# Accuracy and Confusion Matrix.
	cm             = confusion_matrix(y_true=class_set, y_pred=pred_set)
	cm             = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	acc_per_class  = cm.diagonal()
	accuracy_total = balanced_accuracy_score(y_true=class_set,  y_pred=pred_set)
	accuracy       = np.round([accuracy_total] + list(acc_per_class), 2)
	# Recall
	recall         = np.round(recall_score(y_true=class_set,    y_pred=pred_set, average=None), 2)
	# Precision
	precision      = np.round(precision_score(y_true=class_set, y_pred=pred_set, average=None), 2)
	# AUC.
	try:
		auc_per_class  = roc_auc_score(y_true=model.one_hot_encoder.transform(class_set.reshape((-1,1))), y_score=prob_set, average=None)
		# Macro, subject to data imbalance.
		auc_total      = np.mean(auc_per_class)
		auc_all        = np.round([auc_total] + list(auc_per_class), 2)
	except:
		auc_all = [None]
		for class_ in range(model.mult_class):
			try:
			    fpr, tpr, thresholds = roc_curve(model.one_hot_encoder.transform(class_set.reshape((-1,1)))[:, class_], prob_set[:, class_])
			    roc_auc = np.round(auc(fpr, tpr), 2)
			except:
				roc_auc = None
			auc_all.append(roc_auc)

	# Get the top worst performers
	diff         = np.abs(class_set[:,0]-prob_set[:,1])
	inds         = np.argsort(diff)
	top_nsamples = math.ceil(top_percent*class_set.shape[0])
	top_ind      = inds[-top_nsamples:]

	top_prob     = prob_set[top_ind, 1]
	top_slides   = slide_set[top_ind,0]
	top_class    = class_set[top_ind,0]

	top_sl  = list(zip(top_slides, top_prob, top_class))

	# In case we need the relevant tiles.
	return accuracy, recall, precision, auc_all, class_set, pred_set, prob_set, slide_set, top_sl

# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def save_weights_attention(model, set_type, session, output_path, slides, patterns, latent, train_slides, valid_slides=None, labels=None):

	# Unique slides to iterate through.
	unique_slides    = list(np.unique(slides))

	# Variables to return.
	weights_set   = np.zeros((latent.shape[0], 1))
	probabilities = np.zeros((latent.shape[0], 2))
	labels        = np.zeros((latent.shape[0], 1))
	fold_set      = np.empty((latent.shape[0], 1), dtype=object)
	slides_metric = np.empty((latent.shape[0], 1), dtype=object)

	# Iterate through slides.
	for slide in unique_slides:
		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		if indxs.shape[0] < 50: continue

		random.shuffle(indxs)
		# print('Indxs', indxs.shape)
		indxs = np.array(sorted(indxs[:model.bag_size]))
		latents_batch = latent[indxs, :]

		# Label processing for the tile.
		label_instances = patterns[indxs[0]]
		label_batch_int = model.process_label(label_instances[0])

		if slide in train_slides:
			fold_set_name = 'train'
		elif slide in valid_slides:
			fold_set_name = 'valid'
		else:
			fold_set_name = 'test'

		# Run the model.
		feed_dict = {model.represenation_input:latents_batch}
		prob_batch, weights = session.run([model.prob, model.weights], feed_dict=feed_dict)

		# In case we use sigmoid and not multi-class.
		if prob_batch.shape[-1] == 1:
			prob_batch = np.array([1-prob_batch[:,0], prob_batch[:,0]])[:,0]

		for i, index in enumerate(indxs):
			weights_set[index]   = weights[i,0]
			probabilities[index] = prob_batch
			labels[index]        = label_batch_int
			fold_set[index]      = fold_set_name
			slides_metric[index] = slide

	# Store weights in H5 file.
	hdf5_path = os.path.join(output_path, 'hdf5_attention_weights_%s.h5' % set_type)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:
		hdf5_content.create_dataset('weights',       data=weights_set)
		hdf5_content.create_dataset('probabilities', data=probabilities)
		hdf5_content.create_dataset('labels',        data=labels)
		hdf5_content.create_dataset('fold_set',      data=fold_set.astype('S'))
		hdf5_content.create_dataset('slides_metric', data=slides_metric.astype('S'))


# Compute different metrics for given set: Accuracy, Recall, Precision, and AUC.
def save_weights_attention_multimagnifications(model, set_type, session, output_path, slides, patterns, latent_20x, latent_10x, latent_5x, train_slides, valid_slides=None, labels=None, flat_20x=False):

	# Unique slides to iterate through.
	unique_slides    = list(np.unique(slides))

	# Variables to return.
	weights_5x_set  = np.zeros((latent_5x.shape[0], 1))
	weights_10x_set = np.zeros((latent_5x.shape[0], 4, 1))
	if flat_20x:
		weights_20x_set = np.zeros((latent_5x.shape[0], 16, 1))
	else:
		weights_20x_set = np.zeros((latent_5x.shape[0], 4, 4, 1))
	probabilities   = np.zeros((latent_5x.shape[0], 2))
	labels          = np.zeros((latent_5x.shape[0], 1))
	fold_set        = np.empty((latent_5x.shape[0], 1), dtype=object)
	slides_metric   = np.empty((latent_5x.shape[0], 1), dtype=object)

	total_size = 0
	# Iterate through slides.
	for slide in unique_slides:
		if slide == '': continue

		# Gather tiles for the slide.
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide latents for 20x and 5x.
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]
		if flat_20x:
			lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, lantents_5x_batch.shape[-1]))

		# Label processing for the tile.
		label_instances = patterns[start_ind]
		label_batch_int = model.process_label(label_instances[0])

		if slide in train_slides:
			fold_set_name = 'train'
		elif slide in valid_slides:
			fold_set_name = 'valid'
		else:
			fold_set_name = 'test'

		# Run the model.
		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch}
		prob_batch, weights_5x, weights_10x, weights_20x = session.run([model.prob, model.weights, model.weights_10x, model.weights_20x], feed_dict=feed_dict)

		if prob_batch.shape[-1] == 1:
			prob_batch = np.array([1-prob_batch, prob_batch])
			prob_batch = prob_batch.reshape((1,2))

		for i, index in enumerate(indxs):
			weights_5x_set[index]  = weights_5x[i,0]
			weights_10x_set[index, :, 0] = weights_10x[i,:,0]
			if flat_20x:
				weights_20x_set[index, :, 0] = weights_20x[i,:,0]
			else:
				weights_20x_set[index, :, :, 0] = weights_20x[i,:,:,0]
			probabilities[index] = prob_batch
			labels[index]        = label_batch_int
			fold_set[index]      = fold_set_name
			slides_metric[index] = slide

			total_size += 1

	# Store weights in H5 file.
	dt = h5py.special_dtype(vlen=str)
	hdf5_path = os.path.join(output_path, 'hdf5_attention_weights_%s.h5' % set_type)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:
		hdf5_content.create_dataset('weights_20x',   data=weights_20x_set)
		hdf5_content.create_dataset('weights_10x',   data=weights_10x_set)
		hdf5_content.create_dataset('weights_5x',    data=weights_5x_set)
		hdf5_content.create_dataset('labels',        data=labels)
		hdf5_content.create_dataset('probabilities', data=probabilities)
		hdf5_content.create_dataset('fold_set',      data=fold_set.astype('S'))
		hdf5_content.create_dataset('slides_metric', data=slides_metric.astype('S'))


########################################## SURVIVAL METRICS ##########################################

def get_predictions_set(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, train_slides, subset_slides):
	censoring    = list()
	prediction   = list()
	labels       = list()
	total_losses = list()
	log_losses   = list()
	rank_losses  = list()

	uniq_slides = subset_slides
	for sample_indx in uniq_slides:
		if sample_indx == '': continue
		# Gather index
		indxs = np.argwhere(slides[:]==sample_indx)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide labels.
		labels_pat = patterns[start_ind]
		one_hot, one_hot_step, label, censored = model.process_label(labels_pat)

		if censored == 1 and model.no_censored:
			continue

		# Slide latents for 20x and 5x.
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]
		if lantents_20x_batch.shape[1] == 4:
			lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, model.z_dim))

		model.fill_cache(session, slides, patterns, latent_5x, latent_10x, latent_20x, subset_slides=subset_slides)

		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch, \
					 model.label_one_hot_ind:one_hot, model.label_one_step_ind:one_hot_step, model.censored_ind:censored, model.label_ind:label,
					 model.label_one_hot_batch:model.patient_one_hot, model.label_one_step_batch:model.patient_one_hot_step, model.censored_batch:model.patient_censor, \
					 model.labels_batch:model.patient_label, model.patient_reps_batch:model.patient_cache}
		total_loss, log_loss, rank_loss, event_time_prob = session.run([model.loss, model.loss_1, model.loss_2, model.event_time_prob], feed_dict=feed_dict)

		prediction.append(event_time_prob[0,:])
		censoring.append(censored)
		labels.append(label)
		total_losses.append(total_loss)
		log_losses.append(log_loss)
		rank_losses.append(rank_loss)

	labels       = np.vstack(labels)
	censoring    = np.vstack(censoring)
	prediction   = np.vstack(prediction)
	total_losses = np.vstack(total_losses)
	log_losses   = np.vstack(log_losses)
	rank_losses  = np.vstack(rank_losses)

	return total_losses, log_losses, rank_losses, prediction, censoring, labels, uniq_slides

def c_index(predictions, labels, censoring, t_time):
    risk       = np.sum(predictions[:, :t_time+1], axis=-1)

    N          = predictions.shape[0]
    A          = np.zeros((N, N))
    Q          = np.zeros((N, N))
    N_t        = np.zeros((N, N))
    Num = 0
    Den = 0

    # For each of the patients.
    for i in range(N):
        A[i, np.where(labels[i] < labels)] = 1
        Q[i, np.where(risk[i]   > risk)]   = 1

        if (labels[i]<=t_time and censoring[i]==0):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    return float(Num/Den)

def weighted_c_index(train_labels, train_censoring, test_predictions, test_labels, test_censoring, t_time):
	def CensoringProb(Y, T):
		from lifelines import KaplanMeierFitter

		T = T.reshape([-1]) # (N,) - np array
		Y = Y.reshape([-1]) # (N,) - np array

		kmf = KaplanMeierFitter()
		kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
		G = np.asarray(kmf.survival_function_.reset_index()).transpose()
		G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)

		return G


	G = CensoringProb(Y=train_censoring, T=train_labels)

	risk = np.sum(test_predictions[:, :t_time+1], axis=-1)
	N    = test_predictions.shape[0]
	A    = np.zeros((N, N))
	Q    = np.zeros((N, N))
	N_t  = np.zeros((N, N))
	Num  = 0
	Den  = 0

	# For each of the patients.
	for i in range(N):

		tmp_idx = np.where(G[0,:] >= test_labels[i])[0]

		if len(tmp_idx) == 0:
			weights = (1./G[1,-1])**2
		else:
			weights = (1./G[1,tmp_idx[0]])**2

		A[i, np.where(test_labels[i] < test_labels)] = 1. * weights
		Q[i, np.where(risk[i]   > risk)]   = 1

		if (test_labels[i]<=t_time and test_censoring[i]==0):
			N_t[i,:] = 1

		Num  = np.sum(((A)*N_t)*Q)
		Den  = np.sum((A)*N_t)

	return float(Num/Den)

def get_survival_performance(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, train_slides, test_slides):

	train_total_loss, train_log_loss, train_rank_loss, train_predictions, train_censoring, train_labels, train_slides = get_predictions_set(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, train_slides=train_slides, subset_slides=train_slides)
	test_total_loss,  test_log_loss,  test_rank_loss,  test_predictions,  test_censoring,  test_labels,  test_slides  = get_predictions_set(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, train_slides=train_slides, subset_slides=test_slides)

	train_data = [train_predictions, train_censoring, train_labels, train_slides]
	test_data  = [test_predictions,  test_censoring,  test_labels, test_slides]

	eval_time = [int(np.percentile(train_labels, 25)), int(np.percentile(train_labels, 50)), int(np.percentile(train_labels, 75))]

	train_cth_index_results = np.zeros((1, len(eval_time)))
	test_cth_index_results  = np.zeros((1, len(eval_time)))
	for t, t_time in enumerate(eval_time):

	    train_result = weighted_c_index(train_labels, train_censoring, train_predictions, train_labels, train_censoring, t_time)
	    test_result  = weighted_c_index(train_labels, train_censoring, test_predictions,  test_labels,  test_censoring,  t_time)
	    train_cth_index_results[0, t] = train_result
	    test_cth_index_results[0, t]  = test_result

	train_results = [np.round(np.mean(train_total_loss), 3), np.round(np.mean(train_log_loss), 3), np.round(np.mean(train_rank_loss), 3), np.round(np.mean(train_cth_index_results), 3)]
	test_results  = [np.round(np.mean(test_total_loss), 3),  np.round(np.mean(test_log_loss), 3),  np.round(np.mean(test_rank_loss), 3),  np.round(np.mean(test_cth_index_results), 3)]

	return [train_results, test_results], [train_data, test_data]

def get_survival_performance_add(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, train_slides, slides_2, patterns_2, latent_5x_2, latent_10x_2, latent_20x_2):

	train_total_loss, train_log_loss, train_rank_loss, train_predictions, train_censoring, train_labels, train_slides = get_predictions_set(model, session, slides,   patterns,   latent_5x,   latent_10x,   latent_20x,   train_slides=train_slides, subset_slides=train_slides)
	test_total_loss,  test_log_loss,  test_rank_loss,  test_predictions,  test_censoring,  test_labels,  test_slides  = get_predictions_set(model, session, slides_2, patterns_2, latent_5x_2, latent_10x_2, latent_20x_2, train_slides=train_slides, subset_slides=np.unique(slides_2[:,0]))

	train_data = [train_predictions, train_censoring, train_labels, train_slides]
	test_data  = [test_predictions,  test_censoring,  test_labels, test_slides]

	eval_time = [int(np.percentile(train_labels, 25)), int(np.percentile(train_labels, 50)), int(np.percentile(train_labels, 75))]

	train_cth_index_results = np.zeros((1, len(eval_time)))
	test_cth_index_results  = np.zeros((1, len(eval_time)))
	for t, t_time in enumerate(eval_time):

	    train_result = weighted_c_index(train_labels, train_censoring, train_predictions, train_labels, train_censoring, t_time)
	    test_result  = weighted_c_index(train_labels, train_censoring, test_predictions,  test_labels,  test_censoring,  t_time)
	    train_cth_index_results[0, t] = train_result
	    test_cth_index_results[0, t]  = test_result

	train_results = [np.round(np.mean(train_total_loss), 3), np.round(np.mean(train_log_loss), 3), np.round(np.mean(train_rank_loss), 3), np.round(np.mean(train_cth_index_results), 3)]
	test_results  = [np.round(np.mean(test_total_loss), 3),  np.round(np.mean(test_log_loss), 3),  np.round(np.mean(test_rank_loss), 3),  np.round(np.mean(test_cth_index_results), 3)]

	return [train_results, test_results], [train_data, test_data]

def save_survival_weights(model, session, slides, patterns, latent_5x, latent_10x, latent_20x, output_path, subset_slides, file_name='hdf5_attention_weights.h5'):

	# Variables to return.
	weights_5x_set  = np.zeros((latent_5x.shape[0], 1))
	weights_10x_set = np.zeros((latent_5x.shape[0], 4, 1))
	weights_20x_set = np.zeros((latent_5x.shape[0], 16, 1))
	patient_rep     = np.zeros((latent_5x.shape[0], model.patient_rep_ind.shape[1]))
	predictions     = np.zeros((latent_5x.shape[0], model.event_time_prob.shape[1]))
	censoring       = np.zeros((latent_5x.shape[0], 1))
	labels          = np.zeros((latent_5x.shape[0], 1))
	fold_set        = list()
	slides_h5       = list()

	# Slides sets
	train_slides, test_slides = subset_slides

	# Iterate through slides.
	uniq_slides = list(np.unique(slides))
	for slide in uniq_slides:
		# Gather index
		indxs = np.argwhere(slides[:]==slide)[:,0]
		start_ind = sorted(indxs)[0]
		num_tiles_5x = indxs.shape[0]

		# Slide labels.
		labels_pat = patterns[start_ind]
		one_hot, one_hot_step, label, censored = model.process_label(labels_pat)

		# Slide latents for 20x and 5x.
		lantents_5x_batch  = latent_5x[start_ind:start_ind+num_tiles_5x]
		lantents_10x_batch = latent_10x[start_ind:start_ind+num_tiles_5x]
		lantents_20x_batch = latent_20x[start_ind:start_ind+num_tiles_5x]
		if lantents_20x_batch.shape[1] == 4:
			lantents_20x_batch = np.reshape(lantents_20x_batch, (num_tiles_5x, 16, model.z_dim))

		model.fill_cache(session, slides, patterns, latent_5x, latent_10x, latent_20x, subset_slides=train_slides)

		feed_dict = {model.represenation_input_20x:lantents_20x_batch, model.represenation_input_10x:lantents_10x_batch, model.represenation_input_5x:lantents_5x_batch, model.patient_reps_batch:model.patient_cache}
		patient_rep_ind, prediction_batch, weights_5x_batch, weights_10x_batch, weights_20x_batch = session.run([model.patient_rep_ind, model.event_time_prob, model.weights, model.weights_10x, model.weights_20x], feed_dict=feed_dict)

		if slide in train_slides:
			set_name = 'train'
		else:
			set_name = 'test'

		for i, index in enumerate(indxs):
			# Weights
			weights_5x_set[index]        = weights_5x_batch[i,0]
			weights_10x_set[index, :, 0] = weights_10x_batch[i,:,0]
			weights_20x_set[index, :, 0] = weights_20x_batch[i,:,0]
			patient_rep[index, :]        = patient_rep_ind[0,:]
			# patient_cause[index, :]      = patient_cause_batch[0,:]
			predictions[index, :]        = prediction_batch[0,:]
			censoring[index, 0]          = censored[0,0]
			labels[index, 0]             = label[0,0]
			fold_set.append(set_name)
			slides_h5.append(slide)

	fold_set = np.vstack(fold_set)
	slides_h5 = np.vstack(slides_h5)

	# Store weights in H5 file.
	hdf5_path = os.path.join(output_path, file_name)
	with h5py.File(hdf5_path, mode='w') as hdf5_content:
		hdf5_content.create_dataset('weights_20x', data=weights_20x_set)
		hdf5_content.create_dataset('weights_10x', data=weights_10x_set)
		hdf5_content.create_dataset('weights_5x',  data=weights_5x_set)
		hdf5_content.create_dataset('predictions', data=predictions)
		hdf5_content.create_dataset('censoring',   data=censoring)
		hdf5_content.create_dataset('labels',      data=labels)
		hdf5_content.create_dataset('set',         data=fold_set.astype('S'))
		hdf5_content.create_dataset('slides',      data=slides_h5.astype('S'))
