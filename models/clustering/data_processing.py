2# Imports
from skbio.stats.composition import clr, ilr, alr, multiplicative_replacement
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import anndata
import copy
import h5py
import copy
import os

# Own libs
from models.evaluation.folds import load_existing_split

'''############### Utilities ###############'''
# Get the reference H5 AnnData file for clustering.
def read_h5ad_reference(h5_complete_path, meta_folder, groupby, fold_number):
	main_cluster_path = os.path.join(h5_complete_path.split('hdf5_')[0], meta_folder)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')
	adata_name        = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold_number)
	h5_path = os.path.join(adatas_path, adata_name) + '.h5ad'
	if os.path.isfile(h5_path):
		adata_train = anndata.read_h5ad(h5_path)
	else:
		h5_path = h5_path.replace('.h5ad', '_subsample.h5ad')
		adata_train = anndata.read_h5ad(h5_path)

	return adata_train, h5_path

# Build a DataFrame from H5 file with representations.
def representations_to_frame(h5_path, meta_field, rep_key='z_latent', check_meta=False):
	if h5_path is not None:
		print('Loading representations:', h5_path)
		with h5py.File(h5_path, 'r') as content:
			keys = [str(key) for key in content.keys()]
			if (str(meta_field) not in keys) and check_meta:
				print('Warning: Meta field not found on H5 File datasets')
				print('Meta field:', meta_field, 'Keys:', keys)
				input("Press Enter to continue...")
			for key in content.keys():
				if rep_key in key:
					representations = content[key][:]
					dim_columns     = list(range(representations.shape[1]))
					frame = pd.DataFrame(representations, columns=dim_columns)
					break

			rest_columns = list()
			for key in content.keys():
				if 'latent' in key:
					continue
				key_ = key.replace('train_', '')
				key_ = key_.replace('valid_', '')
				key_ = key_.replace('test_',  '')
				if key_ not in frame.columns:
					frame[key_] = content[key][:].astype(str)
					rest_columns.append(key_)
	else:
		frame, dim_columns, rest_columns = None, None, None

	return frame, dim_columns, rest_columns

# Save observation dataframe from AnnData.
def adata_to_csv(adata, main_cluster_path, adata_name):
    current_df = pd.DataFrame(adata.obs)
    current_df.to_csv(os.path.join(main_cluster_path, '%s.csv' % adata_name), index=False)
    
# Read CSV files.
def read_csvs(adatas_path, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold=False, force_fold=None):
	# Get sets.
	train_samples, valid_samples, test_samples = fold
	
	# Force to a particular fold if needed.
	if force_fold is not None:
		i = force_fold

	# Case to use the additional H5 file as the 5 fold cross-validation.
	if additional_as_fold:
		if h5_additional_path is None:
			print('Impossible combination: Using additional H5 as fold cross-validation and h5_additional_path is None.')
			print(h5_additional_path)
			exit()

		# Get clusters from the reference clustering train set.
		adata_name      = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
		train_csv       = os.path.join(adatas_path, '%s_train.csv' % adata_name)
		if not os.path.isfile(train_csv):
			train_csv = os.path.join(adatas_path, '%s.csv' % adata_name)
		train_df        = pd.read_csv(train_csv)
		leiden_clusters = np.unique(train_df[groupby].values.astype(int))

		# Train, valid, and test set.
		adata_name    = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
		additional_df = pd.read_csv(os.path.join(adatas_path, '%s.csv' % adata_name))
		train_df      = additional_df[additional_df[matching_field].astype(str).isin(map(str, train_samples))]
		test_df       = additional_df[additional_df[matching_field].astype(str).isin(map(str, test_samples))]

		valid_df    = None
		if len(valid_samples) > 0:
			valid_df    = additional_df[additional_df[matching_field].astype(str).isin(map(str, valid_samples))]

		complete_df   = additional_df.copy(deep=True)
		additional_df = None

	# Regular case: Complete H5 as fold cross-validation, additional H5 for external dataset validation.
	else:
		# Train, valid, and test set.
		adata_name   = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
		train_csv    = os.path.join(adatas_path, '%s_train.csv' % adata_name)
		if not os.path.isfile(train_csv):
			train_csv    = os.path.join(adatas_path, '%s.csv' % adata_name)
		valid_csv    = os.path.join(adatas_path, '%s_valid.csv' % adata_name)
		test_csv     = os.path.join(adatas_path, '%s_test.csv' % adata_name)


		# Gather all sets for clustering. 
		train_df    = pd.read_csv(train_csv)
		test_df     = pd.read_csv(test_csv)
		complete_pd = [train_df, test_df]
		if os.path.isfile(valid_csv):
			valid_df = pd.read_csv(valid_csv)
			complete_pd.append(valid_df)
		complete_df  = pd.concat(complete_pd, ignore_index=True)

		train_df    = complete_df[complete_df[matching_field].astype(str).isin(train_samples)]
		test_df     = complete_df[complete_df[matching_field].astype(str).isin(test_samples)]
		if train_df.shape[0] == 0 or test_df.shape[0] == 0:
			print('Warning:')
			print('\tTrain set DataFrame samples:', train_df.shape[0])
			print('\tTest  set DataFrame samples:', test_df.shape[0])
			print('Example of instances in DataFrame[%s]:' % matching_field, complete_df[matching_field].loc[0])
			print('Example of instances in pickle[%s]:   ' % matching_field, train_samples[:5])
		valid_df    = None
		if len(valid_samples) > 0:
			valid_df = complete_df[complete_df[matching_field].astype(str).isin(valid_samples)]

		# Additional set.
		additional_df = None
		if h5_additional_path is not None:
			adata_name     = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
			additional_csv = os.path.join(adatas_path, '%s.csv' % adata_name)
			additional_df = pd.read_csv(additional_csv)

		# Get clusters from the reference clustering train set.
		leiden_clusters = np.unique(complete_df[groupby].values.astype(int))


	if len(leiden_clusters)-1!=np.max(leiden_clusters):
		print('\t\t[Warning] Resolution %s Fold %s is missing a cluster label' % (groupby, i))
		print('\t\t          Comp len(leiden_clusters) vs max:', len(leiden_clusters), np.max(leiden_clusters))
		print('\t\t          Missing cluster label:', set(range(np.max(leiden_clusters))).difference(set(leiden_clusters)))
		print('\t\t          Bug from ScanPy.')
		leiden_clusters = np.array(range(np.max(leiden_clusters)+1))

	train_df = train_df.dropna(subset=[groupby])
	test_df = test_df.dropna(subset=[groupby])
	complete_df = complete_df.dropna(subset=[groupby])
	if valid_df is not None: valid_df = valid_df.dropna(subset=[groupby])
	if additional_df is not None: additional_df = additional_df.dropna(subset=[groupby])

	return [train_df, valid_df, test_df, additional_df], complete_df, leiden_clusters

# Include additional features into clusters frame.
def include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby):
	frame_clusters = frame_clusters.sort_values(by=groupby)
	for feature in features:
		if feature in leiden_clusters.tolist():
			continue

		if '_' in str(feature):
			cluster_id1, cluster_id2 = str(feature).split('_')
			if cluster_id1==cluster_id2:
				subtype = '%s' % frame_clusters.loc[int(cluster_id1), 'Subtype']
			else:
				subtype = '%s-%s' % (frame_clusters.loc[int(cluster_id1), 'Subtype'], frame_clusters.loc[int(cluster_id2), 'Subtype'])

		row = dict()
		for column in frame_clusters.columns:
			row[column] = -1
		row[groupby]   = feature
		row['Subtype'] = subtype
		frame_clusters = frame_clusters.append(row, ignore_index=True)
	return frame_clusters

'''############### Cluster Characterization ###############'''
# Get distribution for entire population of tiles.
def get_entire_population_dist_df(frame, meta_field, reduction=2):
	# Population label distribution.
	labels, counts = np.unique(frame[meta_field], return_counts=True)
	proportions = counts/np.sum(counts)*100
	population_df = pd.DataFrame(np.concatenate([proportions.reshape(-1,1), counts.reshape(-1,1)], axis=1), columns=['proportions', 'counts'])
	population_df.insert(0, 'proportions_th', population_df['proportions'].apply(lambda x: x + (100-x)/reduction))
	population_df.index = labels

	return population_df

# Check cluster purity.
def cluster_purities(frame, population_df, meta_field, groupby):
	# Get one entry per cluster:
	#   1. Get dominant subtype, decrease in all others of at least 50% of original distribution.
	#   2. Check if this is an statistical relevant enrichment? [Not sure how to do this yet].
	x,y = groupby, meta_field
	frame_clusters = frame.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Subtype Purity(%)').reset_index()
	frame_clusters['Subtype Counts'] = frame.groupby(x)[y].value_counts(normalize=False).rename('Subtype Counts').reset_index()['Subtype Counts'].values

	# Get and set dominant.
	frame_clusters_list = list()
	for cluster in np.unique(frame_clusters[groupby].values):
		frame_cluster   = frame_clusters[frame_clusters[groupby]==cluster]
		dominant_series = frame_cluster.loc[frame_cluster['Subtype Purity(%)'].idxmax()].copy(deep=True)
		type_label  = dominant_series[meta_field]
		prop        = dominant_series['Subtype Purity(%)']
		prop_th     = population_df.loc[type_label,'proportions_th']
		dominant_series['Subtype Counts'] = np.sum(frame_cluster['Subtype Counts'].values)
		cluster_row = dominant_series.values.tolist()
		if prop >= prop_th:
			frame_clusters_list.append(cluster_row+[type_label])
		else:
			frame_clusters_list.append(cluster_row+[-1])
	frame_clusters = pd.DataFrame(np.stack(frame_clusters_list), columns=frame_cluster.columns.tolist()+['Subtype'])
	frame_clusters['Subtype Counts'] = frame_clusters['Subtype Counts'].astype(int)
	frame_clusters[groupby]          = frame_clusters[groupby].astype(int)

	return frame_clusters

# Check cluster diversity of samples, contributors to cluster.
def cluster_diversity(frame, frame_clusters, groupby, diversity_key):
	# Sample diversity per cluster.
	x,y = groupby, diversity_key
	frame_samples                   = frame.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Purity (%)').reset_index()
	frame_samples['Counts']         = frame.groupby(x)[y].value_counts(normalize=False).rename('Counts').reset_index()['Counts'].values
	frame_samples['Subtype']        = [np.NaN]*frame_samples.shape[0]
	frame_samples['Subtype Counts'] = [np.NaN]*frame_samples.shape[0]
	for cluster_id in np.unique(frame_samples[x].values):
		clust_type                      = frame_clusters[frame_clusters[x]==cluster_id]['Subtype'].values[0]
		clust_type_counts               = frame_clusters[frame_clusters[x]==cluster_id]['Subtype Counts'].values[0]
		filter_cluster                  = frame_samples[x]==cluster_id
		frame_samples['Subtype']        = frame_samples['Subtype'].mask(filter_cluster        , clust_type)
		frame_samples['Subtype Counts'] = frame_samples['Subtype Counts'].mask(filter_cluster , clust_type_counts)

	# Add information about patient contributions.
	for cluster_id in np.unique(frame_samples[groupby].values):
		frame_res     = frame_samples[frame_samples[groupby]==cluster_id].copy(deep=True)
		frame_res     = frame_res[frame_res.Counts!=0]
		frame_clusters.loc[frame_clusters[groupby]==cluster_id, 'mean_tile_sample'] = int(np.mean(frame_res.Counts))
		frame_clusters.loc[frame_clusters[groupby]==cluster_id, 'max_tile_sample']  = np.round(int(np.max(frame_res.Counts))/frame_clusters[frame_clusters[x]==cluster_id]['Subtype Counts'].values[0],4)
		frame_clusters.loc[frame_clusters[groupby]==cluster_id, 'percent_sample']   = np.round(len(pd.unique(frame_res[y]))/len(pd.unique(frame_samples[y])), 4)

	return frame_clusters, frame_samples

# Create Cluster and Diversity sample frames.
def create_frames(frame, groupby, meta_field, diversity_key, reduction=2):
	# Fraction reduction on other clusters to consider proportion threshold.
	population_df = get_entire_population_dist_df(frame, meta_field, reduction=reduction)

	# Get one entry per cluster:
	#   1. Get dominant subtype, decrease in all others of at least 50% of original distribution.
	#   2. Check if this is an statistical relevant enrichment? [Not sure how to do this yet].
	frame_clusters = cluster_purities(frame, population_df, meta_field, groupby)

	# Sort by Subtype first and Counts afterwards.
	frame_clusters = (frame_clusters.groupby(['Subtype']).apply(lambda x: x.sort_values(['Subtype Counts'], ascending=True)).reset_index(drop=True))

	# Include diversity information, tile contributions in each cluster.
	frame_clusters, frame_samples = cluster_diversity(frame, frame_clusters, groupby, diversity_key)

	return frame_clusters, frame_samples

'''############### Slide Representations ###############'''
# Ratios between clusters in the slide
def cluster_ratios_slide(frame_classification, matching_field, sample, groupby, leiden_clusters, type_):
	# Space feature vector representation.
	samples_features = [0]*len(leiden_clusters)
	# Get cluster counts for sample.
	clusters_slide, clusters_counts = np.unique(frame_classification[frame_classification[matching_field]==sample][groupby], return_counts=True)
	# Map to vector representation dimensions.
	for clust_id, count in zip(clusters_slide, clusters_counts):
		samples_features[int(clust_id)] = count
	samples_features = np.array(samples_features, dtype=np.float64)

	# Space transformation from compositional data.
	# Compositional data breaks assumptions from common linear models: Logistic Regression but also Cox Proportional Hazard Regression.
	if type_ == 'percent':
		samples_features = np.array(samples_features)/np.sum(samples_features)
	elif type_ == 'alr':
		samples_features = samples_features + 1.
		samples_features = np.log(samples_features[:-1]/samples_features[-1])
	elif type_ == 'clr':
		samples_features = samples_features + 1.
		geo_mean         = np.prod(samples_features)**(1.0/samples_features.shape[0])
		samples_features = np.log(samples_features/geo_mean)
	elif type_ == 'ilr':
		samples_features = samples_features + 1.
		samples_features = ilr(samples_features)
	else:
		print('Not contemplated compositional alternative space.')
		print('Options: Percent, ALR, CLR, and ILR.')
		exit()

	if use_ratios:
		index = 0
		total_features = np.zeros(len(leiden_clusters)+(len(leiden_clusters)*len(leiden_clusters)))
		total_features[:len(leiden_clusters)] = samples_features
		for c_1 in leiden_clusters:
			for c_2 in leiden_clusters:
				if c_1 == c_2: continue
				total_features[index + len(leiden_clusters)] = samples_features[c_1]/samples_features[c_2]
				index += 1
		samples_features = total_features

	return samples_features

# Cluster interactions in the slide.
def cluster_conn_slides(frame, leiden_clusters, groupby, type='slide', own_corr=True, min_tiles=100, type_='percent'):
	if type=='slide':
		return cluster_conn_slides_interactions_slide(frame, leiden_clusters, groupby, own_corr=own_corr, min_tiles=min_tiles, type_=type_)
	elif type=='cluster':
		return cluster_conn_slides_interactions_clusters(frame, leiden_clusters, groupby, own_corr=True, min_tiles=100, type_='percent')
	else:
		print('Cluster interaction normalization not contemplated. Options: \'slide\' or \'cluster\'')
		return

# cluster connectivity for slides, interactions normalized per cluster.
def cluster_conn_slides_interactions_clusters(frame, leiden_clusters, groupby, own_corr=True, min_tiles=100, type_='percent'):
	all_slides = np.unique(frame.slides)
	heatmap    = list()

	# Get connectivity between clusters per slide.
	for slide in all_slides:
		slide_frame = frame[frame.slides==slide]
		if slide_frame.shape[0] < min_tiles:
			continue

		# Cluster proportions.
		slide_cluster_prop = np.zeros(len(leiden_clusters), dtype=np.float64)
		cluster_ids, counts = np.unique(slide_frame[groupby], return_counts=True)
		for cluster_id, count in zip(cluster_ids, counts):
			slide_cluster_prop[cluster_id] = count
		slide_cluster_prop = slide_cluster_prop/np.sum(slide_cluster_prop)

		# Get connectivity for each cluster.
		features = list()
		slide_rep = list()
		for cluster_id in leiden_clusters:
			cluster_conn, counts_conn = np.array([np.NAN]), np.array([1])
			# Connections to other clusters.
			if slide_frame[slide_frame[groupby]==cluster_id].shape[0]!=0:
				# Get connections for those cluster tiles.
				cluster_conn, counts_conn = np.unique(slide_frame[slide_frame[groupby]==cluster_id][['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']], return_counts=True)
			# Remove no connections.
			if 'nan' in cluster_conn.astype(str).tolist():
				idx = np.argwhere(cluster_conn.astype(str)=='nan')[0,0]
				cluster_conn = np.delete(cluster_conn, idx, axis=0)
				counts_conn  = np.delete(counts_conn, idx, axis=0)

			# Connections.
			all_conn = list()
			for cluster_id_conn in leiden_clusters:
				# if '%s_%s' % (cluster_id, cluster_id_conn) in features or '%s_%s' % (cluster_id_conn, cluster_id) in features:
				# 	continue
				if np.argwhere(cluster_conn==cluster_id_conn)[:,0].shape[0] == 0:
					conn = 0
				else:
					idx = np.argwhere(cluster_conn==cluster_id_conn)[0,0]
					conn = counts_conn[idx]

				all_conn.append(conn)
				features.append('%s_%s' % (cluster_id, cluster_id_conn))

			# Normalize for all interactions -- Cluster interaction normalization.
			all_conn = np.array(all_conn, dtype=np.float64)
			if not (all_conn==0).all():
				all_conn = all_conn/np.sum(all_conn)
			all_conn *= slide_cluster_prop[cluster_id]

			# Cluster connections to other clusters.
			slide_rep.extend(all_conn.tolist())

		# Normalize for all interactions -- Slide interaction normalization.
		slide_rep = np.array(slide_rep).astype(np.float64)
		slide_rep = multiplicative_replacement(np.reshape(slide_rep, (1,-1)))
		if type_ == 'clr':
			slide_rep = clr(np.reshape(slide_rep, (1,-1)))

		# Append slide representation.
		slide_rep = [slide] + slide_rep.tolist()
		heatmap.append(slide_rep)

	# Return data and labels.
	fields = ['slides'] + features
	heatmap = pd.DataFrame(heatmap, columns=fields)
	return heatmap, fields

# cluster connectivity for slides, interactions normalized for slide.
def cluster_conn_slides_interactions_slide(frame, leiden_clusters, groupby, own_corr=True, min_tiles=100, type_='percent'):
	all_slides = np.unique(frame.slides)
	heatmap    = list()

	# Get connectivity between clusters per slide.
	for slide in all_slides:
		slide_frame = frame[frame.slides==slide]
		if slide_frame.shape[0] < min_tiles:
			continue

		# Get connectivity for each cluster.
		features = list()
		slide_rep = list()
		for cluster_id in leiden_clusters:
			cluster_conn, counts_conn = np.array([np.NAN]), np.array([1])
			# Connections to other clusters.
			if slide_frame[slide_frame[groupby]==cluster_id].shape[0]!=0:
				# Get connections for those cluster tiles.
				cluster_conn, counts_conn = np.unique(slide_frame[slide_frame[groupby]==cluster_id][['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']], return_counts=True)
			# Remove no connections.
			if 'nan' in cluster_conn.astype(str).tolist():
				idx = np.argwhere(cluster_conn.astype(str)=='nan')[0,0]
				cluster_conn = np.delete(cluster_conn, idx, axis=0)
				counts_conn  = np.delete(counts_conn, idx, axis=0)

			# Connections.
			all_conn = list()
			for cluster_id_conn in leiden_clusters:
				if '%s_%s' % (cluster_id, cluster_id_conn) in features or '%s_%s' % (cluster_id_conn, cluster_id) in features:
					continue
				if np.argwhere(cluster_conn==cluster_id_conn)[:,0].shape[0] == 0:
					conn = 0
				else:
					idx = np.argwhere(cluster_conn==cluster_id_conn)[0,0]
					conn = counts_conn[idx]

				all_conn.append(conn)
				features.append('%s_%s' % (cluster_id, cluster_id_conn))

			# Cluster connections to other clusters.
			slide_rep.extend(all_conn)

		# Normalize for all interactions -- Slide interaction normalization.
		# slide_rep = np.array(slide_rep+1).astype(np.float64)    # Avoid zero for compositional data.
		slide_rep = np.array(slide_rep).astype(np.float64)
		slide_rep = slide_rep/np.sum(slide_rep)
		slide_rep = multiplicative_replacement(np.reshape(slide_rep, (1,-1)))
		if type_ == 'clr':
			slide_rep = clr(np.reshape(slide_rep, (1,-1)))

		# Append slide representation.
		slide_rep = [slide] + slide_rep.tolist()
		heatmap.append(slide_rep)

	# Return data and labels.
	fields = ['slides'] + features
	heatmap = pd.DataFrame(heatmap, columns=fields)
	return heatmap, fields

# Keep top_percent variables variance.
def feature_selector(data, fields, top_percent):
	keep_clusters = fields
	mask = [True]

	var_selector = VarianceThreshold()
	var_selector.fit_transform(data)
	var_threshold = -np.percentile(-var_selector.variances_, q=top_percent)
	mask.extend((var_selector.variances_ > var_threshold).tolist())
	keep_clusters = np.array(keep_clusters)[mask].tolist()

	return mask, keep_clusters

# Method to build sample representation based on clusters.
def sample_representation(frame_classification, matching_field, sample, groupby, leiden_clusters, type_):
	# Space feature vector representation.
	# samples_features = [0]*len(leiden_clusters)
	samples_features = [0]*len(leiden_clusters)
	# Get cluster counts for sample.
	clusters_slide, clusters_counts = np.unique(frame_classification[frame_classification[matching_field]==sample][groupby], return_counts=True)
	# Map to vector representation dimensions.
	for clust_id, count in zip(clusters_slide, clusters_counts):
		samples_features[int(clust_id)] = count
	samples_features = np.array(samples_features, dtype=np.float64)
	samples_features = np.array(samples_features)/np.sum(samples_features)

	# Aitchison Space transformations.
	# Compositional data breaks assumptions from common linear models (Logistic Regression but also Cox Proportional Hazard Regression):
	# One feature is a linear combination of the remaining.
	if type_ == 'alr':
		samples_features = multiplicative_replacement(np.reshape(samples_features, (1,-1)))
		samples_features = alr(np.reshape(samples_features, (1,-1)))
	elif type_ == 'clr':
		samples_features = multiplicative_replacement(np.reshape(samples_features, (1,-1)))
		samples_features = clr(np.reshape(samples_features, (1,-1)))
	elif type_ == 'ilr':
		samples_features = multiplicative_replacement(np.reshape(samples_features, (1,-1)))
		samples_features = ilr(np.reshape(samples_features, (1,-1)))

	return samples_features

''' ############### Set Representations for CSV ############### '''
def build_cohort_representations(meta_folder, meta_field, matching_field, groupby, fold_number, folds_pickle, h5_complete_path, h5_additional_path, type_composition, min_tiles,
								 use_conn=False, use_ratio=False, top_variance_feat=99, reduction=2, return_tiles=False):
	# Build paths references.
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')
	run_path          = os.path.join(main_cluster_path, '%s_fold%s' % (groupby.replace('.','p'), fold_number))
	rep_cluster_path  = os.path.join(run_path, 'representations')
	if not os.path.isdir(rep_cluster_path):
		os.makedirs(rep_cluster_path)

	# Fold
	folds = load_existing_split(folds_pickle)
	fold = folds[fold_number]

	# Read CSV files for train, validation, test, and additional sets.
	dataframes, frame_complete, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, fold_number, fold, h5_complete_path, h5_additional_path)
	train_df, valid_df, test_df, additional_df = dataframes

	# Check clusters and diversity within.
	frame_clusters, frame_samples = create_frames(frame_complete, groupby, meta_field, diversity_key=matching_field, reduction=reduction)

	# Create representations per sample: cluster % of total sample.
	data, data_df, features = prepare_data_classes(dataframes, matching_field, meta_field, groupby, leiden_clusters, type_composition, min_tiles, use_conn=use_conn,
												   use_ratio=use_ratio, top_variance_feat=top_variance_feat, return_tiles=return_tiles)

	# Include features that are not the regular leiden clusters.
	frame_clusters = include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby)

	# Cluster purity 100% LUAD completely - 0% LUSC completely.
	frame_clusters = frame_clusters.sort_values(groupby)
	flags = frame_clusters[meta_field].values.tolist()
	purities = frame_clusters['Subtype Purity(%)'].values.tolist()
	if len(np.unique(flags))>2:
		result_purities = purities
	else:
		result_purities = [purity if flag==1 else 100-purity for flag, purity in zip(flags, purities)]
	result_purities = ['Purity'] + result_purities + [np.NAN]*(len(data_df[0].columns)-len(leiden_clusters)-1)

	# Combine a complete dataframe.
	data_df.insert(0, pd.DataFrame(np.stack(result_purities).reshape((1,-1)), columns=data_df[0].columns.values.tolist()))
	complete_df = pd.concat(data_df, axis=0)

	# Include sample field.
	complete_df.insert(0, 'samples', complete_df['slides'].apply(lambda x: '-'.join(x.split('-')[:3])))

	adata_name    = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s_%s_%s_mintiles_%s.csv' % (groupby.replace('.', 'p'), fold_number, meta_folder, type_composition, min_tiles)
	complete_path = os.path.join(rep_cluster_path, adata_name)
	complete_df.to_csv(complete_path, index=False)

	additional_complete_df = None
	if h5_additional_path is not None:
		adata_name      = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s_%s_%s_mintiles_%s.csv' % (groupby.replace('.', 'p'), fold_number, meta_folder, type_composition, min_tiles)
		additional_path = os.path.join(rep_cluster_path, adata_name)

		additional_complete_df = pd.DataFrame(np.stack(result_purities).reshape((1,-1)), columns=data_df[0].columns.values.tolist())
		additional_complete_df = pd.concat([additional_complete_df, data_df[-1]], axis=0)
		additional_complete_df.to_csv(additional_path, index=False)

	return complete_df, additional_complete_df, frame_clusters, frame_samples, features

'''############### Set Representations for WSI ###############'''
# Prepare set representation with label, slide name, and # tiles.
def prepare_set_representation(frame, matching_field, meta_field, groupby, leiden_clusters, type_, min_tiles):
	lr_data  = list()
	lr_label = list()
	i = 0
	for sample in pd.unique(frame[matching_field]):
		num_tiles = frame[frame[matching_field]==sample].shape[0]
		if num_tiles<min_tiles:
			continue
		# Sample representations
		sample_rep   = sample_representation(frame, matching_field, sample, groupby, leiden_clusters, type_)
		label_sample = frame[frame[matching_field]==sample][meta_field].values[0]
		label_slide  = frame[frame[matching_field]==sample].slides.values[0]
		lr_data.append(sample_rep)
		lr_label.append((label_slide, label_sample, num_tiles))
	lr_data  = np.stack(lr_data)
	lr_label = np.stack(lr_label)

	slide_rep_df = pd.DataFrame(data=lr_data, columns=leiden_clusters)
	slide_rep_df[meta_field]  = lr_label[:,1].astype(float).astype(int)
	slide_rep_df['slides']   = lr_label[:,0].astype(str)
	slide_rep_df['tiles']    = lr_label[:,2].astype(int)
	slide_rep_df = slide_rep_df.set_index('slides')

	return slide_rep_df

'''############### Subtype Classification ###############'''
# Prepare set data for binary classification.
def prepare_set_classes(frame, matching_field, meta_field, groupby, leiden_clusters, type_, min_tiles, use_conn=True, own_corr=True, use_ratio=False, top_variance_feat=100, keep_features=None,
						return_tiles=False):
	# Get slide representations just with % per cluster in slide.
	slide_rep_df = prepare_set_representation(frame, matching_field=matching_field, meta_field=meta_field, groupby=groupby, leiden_clusters=leiden_clusters, type_=type_, min_tiles=min_tiles)
	slide_rep_df = slide_rep_df.reset_index()
	if not return_tiles:
		slide_rep_df = slide_rep_df.drop(columns=['tiles'])

	# Get cluster connectivity within the slides.
	if use_conn:
		slide_con_df, con_fields = cluster_conn_slides(frame, leiden_clusters=leiden_clusters, groupby=groupby, own_corr=own_corr, min_tiles=min_tiles, type_=type_)

	# Only for feature engineering.
	if use_conn or use_ratio:
		# Only on train set check variance of conn feature to check which to drop.
		if keep_features is None:
			keep_features = ['%s_%s' % (cluster_id,cluster_id) for cluster_id in leiden_clusters]
			con_fields    = [field for field in con_fields if field not in keep_features]
			_, keep_features_var = feature_selector(slide_con_df[con_fields[1:]].to_numpy(), fields=con_fields, top_percent=top_variance_feat)
			keep_features.extend(keep_features_var[1:])
			keep_features = ['slides'] + keep_features
			slide_rep            = pd.merge(slide_rep_df, slide_con_df[keep_features], on=matching_field)
		else:
			slide_rep        = pd.merge(slide_rep_df, slide_con_df[keep_features], on=matching_field)
		# features = leiden_clusters.tolist() + keep_features[1:] # Include all cluster %.
		features = keep_features[1:]                            # Only include connectivity features.

	# Just cluster percentage.
	else:
		slide_rep     = slide_rep_df.copy(deep=True)
		features      = copy.deepcopy(leiden_clusters.tolist())
		keep_features = None

	# Binary labels.
	labels_uniq = np.unique(slide_rep[meta_field].values.astype(int).tolist())
	labels = OneHotEncoder().fit_transform(np.array(slide_rep[meta_field].values.astype(int).tolist()).reshape(-1,1)).toarray()

	data = [slide_rep[features].to_numpy(), labels]
	return data, slide_rep, keep_features, features

# Prepare complete fold data for survival: Train/Validation/Test/Additional
def prepare_data_classes(dataframes, matching_field, meta_field, groupby, leiden_clusters, type_composition, min_tiles, use_conn=True, own_corr=True, use_ratio=False, top_variance_feat=100, return_tiles=False):
	train_df, valid_df, test_df, additional_df = dataframes

	# Create representations per sample: cluster % of total sample.
	train, train_slides_df, keep_features, features  = prepare_set_classes(train_df, matching_field, meta_field, groupby, leiden_clusters, type_=type_composition, min_tiles=min_tiles,
																		   use_conn=use_conn, own_corr=own_corr, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=None, return_tiles=return_tiles)
	valid           = None
	valid_slides_df = None
	if valid_df is not None:
		valid, valid_slides_df, _,             _         = prepare_set_classes(valid_df, matching_field, meta_field, groupby, leiden_clusters, type_=type_composition, min_tiles=min_tiles,
																			   use_conn=use_conn, own_corr=own_corr, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features, return_tiles=return_tiles)
	test,  test_slides_df,  _,             _         = prepare_set_classes(test_df,  matching_field, meta_field, groupby, leiden_clusters, type_=type_composition, min_tiles=min_tiles,
																		   use_conn=use_conn, own_corr=own_corr, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features, return_tiles=return_tiles)
	additional           = None
	additional_slides_df = None
	if additional_df is not None:
		additional, additional_slides_df, _, _ = prepare_set_classes(additional_df, matching_field, meta_field, groupby, leiden_clusters, type_=type_composition, min_tiles=min_tiles,
																	 use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features, return_tiles=return_tiles)

	return [train, valid, test, additional], [train_slides_df, valid_slides_df, test_slides_df, additional_slides_df], features

'''############### Survival Regression ###############'''
# Trim maximum time for survival event.
def trim_event_data(frame, event_data_field, max_months=12.0*12.0):
	event_data = np.clip(frame[event_data_field].values.astype(float), a_min=0.0, a_max=max_months).tolist()
	frame =  frame.drop(columns=[event_data_field])
	frame[event_data_field] = event_data
	return frame

# Prepare set data with slide representations.
def prepare_set_representation_survival(frame_classification, matching_field, groupby, leiden_clusters, type_, event_ind_field, event_data_field, min_tiles):
	lr_data  = list()
	lr_samples = list()
	lr_event_data_field = list()
	lr_event_ind_field = list()
	for sample in pd.unique(frame_classification[matching_field]):
		if frame_classification[frame_classification[matching_field]==sample].shape[0]<min_tiles:
			continue
		# Get event time and indicator. 
		sample_event_time = float(frame_classification[frame_classification[matching_field]==sample][event_data_field].values[0])
		sample_event_ind  = int(frame_classification[frame_classification[matching_field]==sample][event_ind_field].values[0])
		sample_rep        = sample_representation(frame_classification, matching_field, sample, groupby, leiden_clusters, type_)
		lr_data.append(sample_rep)
		lr_samples.append(sample)
		lr_event_data_field.append(sample_event_time)
		lr_event_ind_field.append(sample_event_ind)

	lr_data  = np.stack(lr_data)
	slide_rep_df = pd.DataFrame(data=lr_data, columns=leiden_clusters)
	slide_rep_df[matching_field]   = lr_samples
	slide_rep_df[event_data_field] = lr_event_data_field
	slide_rep_df[event_ind_field]  = lr_event_ind_field

	return slide_rep_df

# Prepare set data for survival classification.
def prepare_set_survival(frame, matching_field, groupby, leiden_clusters, type_, event_ind_field, event_data_field, min_tiles,
						 use_conn=True, own_corr=True, use_ratio=False, top_variance_feat=100, keep_features=None):
	# Get slide representations just with % per cluster in slide.
	slide_rep_df = prepare_set_representation_survival(frame, matching_field=matching_field, groupby=groupby, leiden_clusters=leiden_clusters, type_=type_,
													   event_ind_field=event_ind_field, event_data_field=event_data_field, min_tiles=min_tiles)

	# Get cluster connectivity within the slides.
	if use_conn:
		slide_con_df, con_fields = cluster_conn_slides(frame, leiden_clusters=leiden_clusters, groupby=groupby, own_corr=own_corr, min_tiles=min_tiles, type_=type_)

	# Only for feature engineering.
	if use_conn or use_ratio:
		# Only on train set check variance of conn feature to check which to drop.
		if keep_features is None:
			keep_features = ['%s_%s' % (cluster_id,cluster_id) for cluster_id in leiden_clusters]
			con_fields    = [field for field in con_fields if field not in keep_features]
			_, keep_features_var = feature_selector(slide_con_df[con_fields[1:]].to_numpy(), fields=con_fields, top_percent=top_variance_feat)
			keep_features.extend(keep_features_var[1:])
			keep_features = ['slides'] + keep_features
			slide_rep            = pd.merge(slide_rep_df, slide_con_df[keep_features], on=matching_field)
		else:
			slide_rep        = pd.merge(slide_rep_df, slide_con_df[keep_features], on=matching_field)
		features = leiden_clusters.tolist() + keep_features[1:] # Include all cluster %.
		# features = keep_features[1:]                            # Only include connectivity features.

	# Just cluster percentage.
	else:
		slide_rep     = slide_rep_df.copy(deep=True)
		features      = copy.deepcopy(leiden_clusters.tolist())
		keep_features = None

	slides_df = slide_rep[[matching_field, event_data_field, event_ind_field]+features]
	return slides_df, keep_features, features

# Prepare complete fold data for survival: Train/Validation/Test/Additional
def prepare_data_survival(dataframes, groupby, leiden_clusters, type_composition, max_months, matching_field, event_ind_field, event_data_field, min_tiles,
						  use_conn=True, use_ratio=False, top_variance_feat=100, remove_clusters=None):
	# Dataframes.
	train_df, valid_df, test_df, additional_df = dataframes

	# Clip based on maximum # of months.
	train_df = trim_event_data(train_df, event_data_field, max_months=max_months)
	test_df  = trim_event_data(test_df,  event_data_field, max_months=max_months)
	if valid_df is not None:
		valid_df = trim_event_data(valid_df, event_data_field, max_months=max_months)
	if additional_df is not None:
		additional_df = trim_event_data(additional_df, event_data_field, max_months=max_months)

	# Get slide representations and labels
	train_slides_df, keep_features, features = prepare_set_survival(train_df, matching_field, groupby, leiden_clusters, type_composition, event_ind_field, event_data_field, min_tiles=min_tiles,
																	use_conn=use_conn, own_corr=True, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=None)
	test_slides_df,  _,             _        = prepare_set_survival(test_df,  matching_field, groupby, leiden_clusters, type_composition, event_ind_field, event_data_field, min_tiles=min_tiles,
																	use_conn=use_conn, own_corr=True, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features)
	valid_slides_df = None
	if valid_df is not None:
		valid_slides_df, _, _ = prepare_set_survival(valid_df, matching_field, groupby, leiden_clusters, type_composition, event_ind_field, event_data_field, min_tiles=min_tiles,
													 use_conn=use_conn, own_corr=True, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features)
	additional_slides_df = None
	if additional_df is not None:
		additional_slides_df, _, _  = prepare_set_survival(additional_df, matching_field, groupby, leiden_clusters, type_composition, event_ind_field, event_data_field, min_tiles=min_tiles,
														   use_conn=use_conn, own_corr=True, use_ratio=use_ratio, top_variance_feat=top_variance_feat, keep_features=keep_features)

	# TODO - Experimental remove 'background' clusters towards a slide.
	if remove_clusters is not None:
		features = [cluster_id for cluster_id in features if cluster_id not in remove_clusters]

	processed_data = [(train_slides_df, 'train'), (valid_slides_df, 'valid'), (test_slides_df, 'test'), (additional_slides_df, 'additional')]

	# Return set data into lists.
	list_df     = list()
	list_all_df = list()
	for process_df, set_name in processed_data:
		all_      = None
		dataframe = None
		if process_df is not None:
			all_       = process_df[[matching_field, event_data_field, event_ind_field]+features].copy(deep=True)
			dataframe  = process_df[[event_data_field, event_ind_field]+features].copy(deep=True)
		list_df.append((dataframe, set_name))
		list_all_df.append((all_, set_name))

	return list_df, list_all_df, features

