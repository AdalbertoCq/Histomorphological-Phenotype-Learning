import tensorflow as tf
from sklearn.cluster import KMeans
import numpy as np
import umap
import copy
import h5py
import os
import gc

# Get discriminator projections for real images.
def get_projections_all_dataset_projections(model, data, session, run_options, data_out_path, num_clusters, batch_size=50):
	num_samples = data.training.images.shape[0]
	batches = int(num_samples/batch_size)

	hdf5_features = os.path.join(data_out_path, 'checkpoints/selconditioned_labels.h5')
	if os.path.isfile(hdf5_features):
		os.remove(hdf5_features)

	with h5py.File(hdf5_features, mode='w') as hdf5_features_file:
		features_storage = hdf5_features_file.create_dataset(name='features', shape=(num_samples, model.features_fake.shape[1]), dtype=np.float32)

		print('Projecting images...')
		ind = 0
		for batch_num in range(batches):
			batch_images = data.training.images[batch_num*batch_size:(batch_num+1)*batch_size]
			if np.amax(batch_images) > 1.0: batch_images = batch_images/255.
			feed_dict = {model.real_images:batch_images}
			batch_projections = session.run([model.features_real], feed_dict=feed_dict, options=run_options)[0]
			features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = batch_projections
			ind += batch_size
			if ind%10000==0: print('Processed', ind, 'images')
		print('Processed', ind, 'images')

		print('Running UMAP...')
		umap_reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42, low_memory=True)
		umap_fitted = umap_reducer.fit(features_storage[model.selected_indx, :])
		embedding_umap_clustering = umap_fitted.transform(features_storage)

		# K-Means
		print('Running K_means...')
		kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10).fit(embedding_umap_clustering)
		new_classes = kmeans.predict(umap_reducer.transform(features_storage[model.selected_indx, :]))

		if np.unique(model.feature_labels).shape[0] > 1:
			print('Hungarian matching...')
			match = hungarian_matching(model=model, new_classes=new_classes, current_classes=model.feature_labels, num_clusters=model.k)
			model.mapping_ = [int(j) for i, j in sorted(match)]

		# Set labels.
		print('Mapping...')
		model.feature_labels = np.array([model.mapping_[x] for x in new_classes])

		feature_labels_storage = hdf5_features_file.create_dataset(name='feat_cluster_labels', shape=[num_samples] + [1], dtype=np.float32)
		embedding_storage      = hdf5_features_file.create_dataset(name='embedding',           shape=[num_samples] + [2], dtype=np.float32)

		print('Finding clusters for embeddings...')

		# Save storage for cluster labels.
		for i in range(num_samples):
			i_class = kmeans.predict(embedding_umap_clustering[i,:].reshape((1,-1)))[0]
			feature_labels_storage[i, 0] = model.mapping_[i_class]
			embedding_storage[i, :]      = embedding_umap_clustering[i, :]
			if i%10000==0: print('Processed', i, 'cluster classes')

# Get discriminator projections for real images.
def get_projections(model, data, session, run_options):
	feature_projection = np.zeros((len(model.selected_indx), model.feature_space))
	indx = 0
	while indx < len(model.selected_indx):
		# Real images.
		if (indx + model.batch_size) < len(model.selected_indx):
			current_ind = model.selected_indx[indx:indx+model.batch_size]
		else:
			current_ind = model.selected_indx[indx:]
		batch_images = data.training.images[current_ind, :, :, :]/255.
		feed_dict = {model.real_images:batch_images}
		batch_projections = session.run([model.features_real], feed_dict=feed_dict, options=run_options)[0]
		feature_projection[indx:indx+model.batch_size, :] = batch_projections
		indx += model.batch_size
	return feature_projection

# Get centroids for each cluster.
def get_initialization_centroids(model, embedding):
	means = np.zeros((model.k, embedding.shape[1]))
	for i in range(model.k):
		mask = (model.feature_labels == i)
		i_mean = np.zeros(embedding.shape[1])
		numels = mask.astype(int).sum()
		if numels > 0:
			for index, flag in enumerate(mask):
				if flag: i_mean += embedding[index, :]
			means[i, :] = i_mean/numels
		else:
			means[i, :] = random.randint(0, embedding.shape[1] - 1)
	return means

# Hungarian matching for classes.
def hungarian_matching(model, new_classes, current_classes, num_clusters):
	# from sklearn.utils.linear_assignment_ import linear_assignment
	from scipy.optimize import linear_sum_assignment as linear_assignment
	num_samples = new_classes.shape[0]
	num_correct = np.zeros((num_clusters, num_clusters))

	for i in range(num_clusters):
		for j in range(num_clusters):
			coin = int(((new_classes==i)*(current_classes==j)).sum())
			num_correct[i, j] = coin

	match = linear_assignment(num_samples-num_correct)

	res = list()
	for out_c, gt_c in np.transpose(np.asarray(match)):
		res.append((out_c, gt_c))

	return res

# Recluster based on discriminator projections.
def recluster(model, data, session, run_options):
	# Get centroids.
	print('\tGetting feature projections...')
	feature_projection = get_projections(model=model, data=data, session=session, run_options=run_options)

	# Run K-Means.
	print('\tFitting UMAP...')
	umap_fitted = umap.UMAP(n_components=2, random_state=42, low_memory=True).fit(feature_projection)
	print('\tTransforming UMAP...')
	embedding = umap_fitted.transform(feature_projection)

	# Get projections.
	print('\tGetting previous centroids...')
	if np.unique(model.feature_labels).shape[0] > 1:
		embedding = umap_fitted.transform(feature_projection)
		initialization = get_initialization_centroids(model=model, embedding=embedding)
	else:
		initialization = 'k-means++'

	# Run K-Means.
	print('\tRunning K-Means...')
	kmeans = KMeans(init=initialization, n_clusters=model.k, n_init=10).fit(embedding)
	new_classes = kmeans.predict(embedding)

	# Compute permutation, Hungarian match.
	if np.unique(model.feature_labels).shape[0] > 1:
		print('\tHungarian matching...')
		match = hungarian_matching(model=model, new_classes=new_classes, current_classes=model.feature_labels, num_clusters=model.k)
		model.mapping_ = [int(j) for i, j in sorted(match)]

	# Set labels.
	print('\tMapping...')
	model.feature_labels = np.array([model.mapping_[x] for x in new_classes])

	clust_labels, counts    = np.unique(model.feature_labels, return_counts=True)
	model.categorical       = counts/np.sum(counts)
	model.reclusters_iter  += 1

	return umap_fitted, kmeans


# Get cluster labels for images.
def get_labels_cluster(model, images_batch, session, run_options, umap_fitted, kmeans):
	feed_dict = {model.real_images:images_batch}
	batch_projections = session.run([model.features_real], feed_dict=feed_dict, options=run_options)[0]
	embedding_batch = umap_fitted.transform(batch_projections)
	batch_classes = kmeans.predict(embedding_batch)
	# Takes into account the hungarian matching.
	permuted_prediction = np.array([model.mapping_[x] for x in batch_classes])

	return permuted_prediction

## Self-supervised Clustering SwAV.
#	Sinkhorn Knopp for Cluster Assignment
#   SwAV Paper: https://arxiv.org/abs/2006.09882
#   Q+ = Diag(u)*exp(C.t*Z/eps)*Diag(v)
#
#   solution for max{ QC.tZ.T } + eps H(Q)
#				  Q+ e Q
#
#   u and v are renormalization vector in Re^K and Re^B respectevely.
def sinkhorn(sample_prototype_batch, batch_size, epsilon=0.05, n_iters=3):

	# Clarify this Q
	# sample_prototype_batch (batch_size, prototype_dim)
	Q = tf.transpose(tf.exp(sample_prototype_batch/epsilon))
	# Q (batch_size, prototype_dim)
	n = tf.reduce_sum(Q)
	Q = Q/n
	K,B = Q.shape.as_list()
	B =  batch_size

	u = tf.zeros_like(K, dtype=tf.float32)
	r = tf.ones_like(K, dtype=tf.float32)/float(K)
	c = tf.ones_like(K, dtype=tf.float32)/float(B)

	for _ in range(n_iters):
		u = tf.reduce_sum(Q, axis=1)
		Q *= tf.expand_dims((r/u), axis=1)
		Q *= tf.expand_dims(c/tf.reduce_sum(Q, axis=0), 0)

	final_quantity = Q/tf.reduce_sum(Q, axis=0, keepdims=True)
	final_quantity = tf.transpose(final_quantity)

	return final_quantity

def sinkhorn_np(sample_prototype_batch, epsilon=0.05, n_iters=3):

	# Clarify this Q
	# Q (batch_size, prototype_dim)
	# sample_prototype_batch (batch_size, prototype_dim)
	Q    = np.transpose(np.exp(sample_prototype_batch/epsilon))
	Q   /= np.sum(Q)

	K,B  = Q.shape

	u =  np.zeros(K, dtype=np.float32)
	r =  np.ones(K, dtype=np.float32)/float(K)
	c =  np.ones(B, dtype=np.float32)/float(B)

	for _ in range(n_iters):
		u  = np.sum(Q, axis=1)
		Q  *= np.expand_dims((r/u), axis=1)
		Q  *= np.expand_dims(c/np.sum(Q,axis=0), 0)

	final_quantity = Q/np.sum(Q, axis=0, keepdims=True)
	final_quantity = np.transpose(final_quantity)

	return final_quantity
