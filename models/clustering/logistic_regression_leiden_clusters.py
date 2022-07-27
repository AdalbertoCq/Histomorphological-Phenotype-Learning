# Imports
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from sklearn.metrics import *
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Own libs.
from models.evaluation.folds import load_existing_split
from models.clustering.data_processing import *
from models.visualization.forest_plots import report_forest_plot_lr
from models.visualization.clusters import cluster_circular, plot_confusion_matrix_lr

''' ############## Combine cluster coefficients ############## '''
# Summarize results for the LR run with same cluster fold.
def summarize_stat_clusters_across_folds(data, meta_field, frame_clusters, alpha, groupby, folds, alpha_path, p_th):
	from scipy.stats import combine_pvalues

	# Get labels.
	train, valid, test, additional = data
	train_data, train_labels = train
	labels = np.unique(np.argmax(train_labels, axis=1)).tolist()
	if len(labels) == 2:
		labels.remove(0)

	for label in labels:
		lr_folds = list()
		counts    = list()
		for i, _ in enumerate(folds):
			lr_run = pd.read_csv(os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i)))
			lr_run['se'] = (lr_run['coef_%s' % label]-lr_run['0.975]_%s' % label])/1.95996
			lr_folds.append(lr_run)
			counts.append(len(folds[i][0]))
		counts    = np.array(counts)
		lr_folds = pd.concat(lr_folds, axis=0)
		lr_coef_pvals = lr_folds[[groupby, 'P>|z|_%s' % label]].copy(deep=True)
		lr_coef       = lr_folds[[groupby, 'coef_%s' % label]].copy(deep=True)
		lr_coef_se    = lr_folds[[groupby, 'se']].copy(deep=True)

		lr_folds = lr_folds.groupby(groupby).mean().reset_index()
		lr_folds = lr_folds.sort_values(by=groupby)

		# Combine coefficients and standard errors for folds.
		avgs = list()
		ses  = list()
		for i in lr_folds[groupby]:
			average = np.average(lr_coef[lr_coef[groupby]==i]['coef_%s' % label].values, weights=counts)
			std = np.sqrt(np.sum((counts - 1) * (lr_coef_se[lr_coef_se[groupby]==i]['se'].values**2))/(np.sum(counts)- counts.shape[0]))
			avgs.append(average)
			ses.append(std)
		lr_folds['coef_%s' % label] = avgs
		lr_folds['se']   = ses
		lr_folds['[0.025_%s' % label] = lr_folds['coef_%s' % label] - (1.95996*lr_folds['se'])
		lr_folds['0.975]_%s' % label] = lr_folds['coef_%s' % label] + (1.95996*lr_folds['se'])

		lr_folds['P>|z|_%s' % label] = lr_coef_pvals.groupby(groupby).apply(lambda x: combine_pvalues(x['P>|z|_%s' % label], method='fisher')[1]).reset_index()[0]
		lr_folds = lr_folds.sort_values(by='coef_%s' % label)
		csv_path  = os.path.join(alpha_path, '%s_stats_all_folds.csv' % (str(groupby).replace('.', 'p')))
		lr_folds.to_csv(csv_path, index=False)
		report_forest_plot_lr(meta_field, lr_folds, directory=alpha_path, file_name='%s_stats_all_folds.csv' % (str(groupby).replace('.', 'p')))

''' ############## Figures ############## '''
# Summarize results for run.
def summarize_run(alphas, resolutions, meta_folder, meta_field, min_tiles, folds_pickle, h5_complete_path, ylim=[0.5,1.01]):

	# Summarize the number of statistically relevant clusters.
	labels = summarize_relevant_clusters(alphas, resolutions, meta_folder, folds_pickle, h5_complete_path, min_tiles)

	# Summarize AUC performance for top 6 penalties.
	alpha_box_plot_auc_results(alphas, meta_folder, meta_field, min_tiles, h5_complete_path, labels, ylim=ylim)

# Summarize the number of statistically relevant clusters.
def summarize_relevant_clusters(alphas, resolutions, meta_folder, folds_pickle, h5_complete_path, min_tiles):
	# Get folds from existing split.
	folds = load_existing_split(folds_pickle)

	# Path for alpha Logistic Regression results.
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)

	# Get number of label runs.
	all_clusters = list()
	for resolution in resolutions:
		groupby = 'leiden_%s' % resolution
		for i, fold in enumerate(folds):
			list_clusters = list()
			for alpha in alphas:
				alpha_path        = os.path.join(main_cluster_path, 'alpha_%s_mintiles_%s' % (str(alpha).replace('.', 'p'), min_tiles))
				csv_path = os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i))
				if os.path.isfile(csv_path):
					frame_clusters    = pd.read_csv(csv_path)
					labels = [column.split('coef_')[1] for column in frame_clusters.columns if 'coef' in column]
					flag_found = True
					break
				if flag_found:
					break
			if flag_found:
				break
		if flag_found:
			break

	# Iterate and pull number of clusters.
	for label in labels:
		all_clusters = list()
		for resolution in resolutions:
			groupby = 'leiden_%s' % resolution
			for i, fold in enumerate(folds):
				list_clusters = list()
				for alpha in alphas:
					alpha_path        = os.path.join(main_cluster_path, 'alpha_%s_mintiles_%s' % (str(alpha).replace('.', 'p'), min_tiles))
					csv_path = os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i))
					if os.path.isfile(csv_path):
						frame_clusters    = pd.read_csv(csv_path)
						relevant_clusters = frame_clusters[frame_clusters['P>|z|_%s'%label]<0.05].shape[0]
					else:
						relevant_clusters = 0
					list_clusters.append(relevant_clusters)
				all_clusters.append([resolution, i] + list_clusters)

		alphas = ['Alpha %s' % alpha for alpha in alphas]
		cluster_stats = pd.DataFrame(all_clusters, columns=['Leiden', 'Fold'] + alphas)
		cluster_stats.to_csv(os.path.join(main_cluster_path, 'clusters_stats_mintiles_%s_label%s.csv' % (min_tiles, label)), index=False)

	return labels

# Box plot results for a given penalty frame.
def box_plot_frame(frame, columns, ax, ylim):
	all_data = list()
	for i in frame.index.tolist():
		values = frame.loc[i].values.tolist()
		for i, column in enumerate(columns[2:]):
			entry = [values[0].replace('leiden_',''), values[1]] + [values[i+2], column]
			all_data.append(entry)
	columns=['Leiden','Fold','AUC','Set']
	all_data = pd.DataFrame(all_data, columns=columns)

	meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}
	sns.pointplot(x='Leiden', hue='Set', y='AUC', data=all_data, ax=ax, linewidth=0.25, dodge=.4, join=False, capsize=.04, markers='s')
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_title('Leiden + Logistic Regression', fontweight='bold', fontsize=18)
	ax.legend(loc='upper left')

# Point plots for logistic regression performance per Leiden clustering.
def box_plot_auc_results(frame, columns, path_file, ylim=None):
	sns.set_theme(style='darkgrid')
	mosaic = '''A'''
	fig = plt.figure(figsize=(20,7), constrained_layout=True)
	ax_dict = fig.subplot_mosaic(mosaic, sharex=True, sharey=True)
	box_plot_frame(frame, columns, ax=ax_dict['A'], ylim=ylim)
	plt.savefig(path_file.replace('.csv', '.jpg'))
	plt.close(fig)

# Results summary figure for all penalties and resolutions, retrieves from csv.
def alpha_box_plot_auc_results(alphas, meta_folder, meta_field, min_tiles, h5_complete_path, labels, fontsize=18, ylim=[0.5,1.01]):

	sns.set_theme(style='darkgrid')
	mosaic = '''01A
				23B
				45C'''
	fig = plt.figure(figsize=(42,24), constrained_layout=True)
	ax_dict = fig.subplot_mosaic(mosaic, sharex=False, sharey=False)

	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)

	for label in labels:
		cluster_stats     = pd.read_csv(os.path.join(main_cluster_path, 'clusters_stats_mintiles_%s_label%s.csv' % (min_tiles, label)))
		for i, alpha in enumerate(alphas):
			alpha_path  = os.path.join(main_cluster_path, 'alpha_%s_mintiles_%s' % (str(alpha).replace('.', 'p'), min_tiles))
			results_csv = os.path.join(alpha_path, '%s_auc_results_mintiles_%s.csv' % (meta_field, min_tiles))
			if not os.path.isfile(results_csv):
				print('Not found:', results_csv)
				continue
			results_df = pd.read_csv(results_csv)
			box_plot_frame(results_df, results_df.columns, ax=ax_dict[str(i)], ylim=ylim)
			ax_dict[str(i)].set_title('Alpha %s' % alpha, fontweight='bold', fontsize=18)

		# Snap together the 3 axis for the table.
		gs = ax_dict['A'].get_gridspec()
		ax_dict['A'].remove()
		ax_dict['B'].remove()
		ax_dict['C'].remove()
		axbig = fig.add_subplot(gs[0:, -1])

		# Table.
		tb = axbig.table(cellText=cluster_stats.values, colLabels=cluster_stats.columns, loc='center', cellLoc='center', bbox=[0, 0, 1.1, 1])
		axbig.axis('off')
		tb.auto_set_font_size(False)
		tb.set_fontsize(fontsize)
		flag = True
		for (row, col), cell in tb.get_celld().items():
			current_res = cluster_stats['Leiden'].values[row-1]
			prev_res    = cluster_stats['Leiden'].values[row-2]
			if row == 1:
				prev_res = None
			if current_res != prev_res and col==0:
				flag = not(flag)
			if flag:
				cell.set_facecolor('silver')
			if (row == 0):
				cell.set_text_props(fontproperties=FontProperties(weight='bold', size=fontsize))

		plt.savefig(os.path.join(main_cluster_path, 'alphas_summary_auc_mintiles_%s_label%s.jpg' % (min_tiles, label)))
		plt.close(fig)

# Circular plots for clusters.
def run_circular_plots(resolutions, folder_meta_field, meta_field, matching_field, folds_pickle, h5_complete_path, h5_additional_path, diversity_key=None):
	# Get folds from existing split.
	folds = load_existing_split(folds_pickle)

	# If diversity key is not specified, use the key that represents samples.
	if diversity_key is None:
		diversity_key = matching_field

	# Main path
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, folder_meta_field)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')
	clus_img_path     = os.path.join(main_cluster_path, 'cluster_circularplots')
	if not os.path.isdir(clus_img_path):
		os.makedirs(clus_img_path)

	for resolution in resolutions:
		groupby     = 'leiden_%s' % resolution
		for i, fold in enumerate(folds):
			# Get sets.
			train_samples, valid_samples, test_samples = fold

			# Read CSV files for train, validation, test, and additional sets.
			adata_name  = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)

			try:
				# Cluster data visualizations for the train subsample set.
				csv_path = os.path.join(adatas_path,   '%s_subsample.csv' % adata_name)
				jpg_path = os.path.join(clus_img_path, '%s_subsample.csv' % adata_name)
				if os.path.isfile(csv_path):
					train_df = pd.read_csv(csv_path)
					frame_clusters, frame_samples = create_frames(train_df, groupby, meta_field, diversity_key=matching_field, reduction=2)
					cluster_circular(frame_clusters, groupby, meta_field, jpg_path)
			except:
				None

			try:
				# Cluster data visualizations for the whole train set.
				csv_path = os.path.join(adatas_path,   '%s.csv' % adata_name)
				jpg_path = os.path.join(clus_img_path, '%s.csv' % adata_name)
				if os.path.isfile(csv_path):
					train_df = pd.read_csv(csv_path)
					frame_clusters, frame_samples = create_frames(train_df, groupby, meta_field, diversity_key=matching_field, reduction=2)
					cluster_circular(frame_clusters, groupby, meta_field, jpg_path)
			except:
				None

			# Read CSV files for train, validation, test, and additional sets.
			if h5_additional_path is not None:
				adata_name  = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)

				csv_path = os.path.join(adatas_path,   '%s.csv' % adata_name)
				jpg_path = os.path.join(clus_img_path, '%s.csv' % adata_name)
				if os.path.isfile(csv_path):
					additional_df = pd.read_csv(csv_path)
					frame_clusters, frame_samples = create_frames(additional_df, groupby, meta_field, diversity_key=matching_field, reduction=2)
					cluster_circular(frame_clusters, groupby, meta_field, jpg_path)

''' ############## Logistic Regression ############## '''
# Get AUC performance.
def get_aucs(model, data, label):
	train, valid, test, additional = data
	train_data, train_labels = train
	if valid is not None:
		valid_data, valid_labels = valid
	test_data,  test_labels  = test
	if additional is not None:
		additional_data, additional_labels = additional

	# Predictions.
	train_pred = model.predict(exog=train_data)
	if valid is not None:
		valid_pred = model.predict(exog=valid_data)
	test_pred  = model.predict(exog=test_data)
	if additional is not None:
		additional_pred = model.predict(exog=additional_data)

	train_auc = roc_auc_score(y_true=list(train_labels[:,label]), y_score=list(train_pred))
	aucs = [train_auc]
	valid_auc = None
	if valid is not None:
		valid_auc = roc_auc_score(y_true=list(valid_labels[:,label]), y_score=list(valid_pred))
		aucs.append(valid_auc)
	test_auc  = roc_auc_score(y_true=list(test_labels[:,label]),  y_score=list(test_pred))
	aucs.append(test_auc)
	additional_auc = None
	if additional is not None:
		additional_pred = model.predict(exog=additional_data)
		additional_auc  = roc_auc_score(y_true=list(additional_labels[:,label]),   y_score=list(additional_pred))
		aucs.append(additional_auc)
	return aucs

# Include coefficients into clusters dataframe.
def include_coefficients(model, frame_clusters_orig, features, label, groupby):
	frame_clusters = frame_clusters_orig.copy(True)

	# Include model coef.
	results_summary = model.summary()
	results_as_html = results_summary.tables[1].as_html()
	results_df      = pd.read_html(results_as_html, header=0, index_col=0)[0]

	for column in ['coef', 'P>|z|','[0.025','0.975]']:
		frame_clusters['%s_%s' % (column, label)] = [np.inf]*frame_clusters.shape[0]
		for i, cluster_id in enumerate(features):
			frame_clusters.loc[frame_clusters[groupby]==cluster_id, '%s_%s' % (column, label)] = float(results_df.loc['x%s' % str(int(i)+1), column])

	return frame_clusters

# Build confusion matrices.
def get_confusion_matrix(model, data, label):
	train, valid, test, additional = data
	test_data,  test_labels  = test
	if additional is not None:
		additional_data, additional_labels = additional

	test_pred  = model.predict(exog=test_data)
	if additional is not None:
		additional_pred = model.predict(exog=additional_data)

	test_labels = test_labels[:,label]
	test_pred   = (test_pred > 0.5)*1.0

	cm_test = confusion_matrix(test_labels, test_pred)
	cm_additional = None
	if additional is not None:
		additional_labels = additional_labels[:,label]
		additional_pred   = (additional_pred > 0.5)*1.0
		cm_additional = confusion_matrix(additional_labels, additional_pred)
	return [cm_test, cm_additional]

# Fit logistic regression and check performance.
def classification_performance_stats(data, leiden_clusters, frame_clusters, features, groupby, alpha):
	train, valid, test, additional = data
	train_data, train_labels = train

	# Train classifier
	labels = np.unique(np.argmax(train_labels, axis=1)).tolist()
	if len(labels) == 2:
		labels.remove(0)

	# One-vs-rest for Logistic Regression.
	num_sets = len([1 for set in data if set is not None])
	shape_aucs = (len(labels), num_sets)
	total_aucs = np.zeros(shape_aucs)
	cms        = dict()
	for label in labels:
		model                  = sm.Logit(endog=train_labels[:,label], exog=train_data).fit_regularized(method='l1', alpha=alpha, disp=0)
		total_aucs[label-1,:]  = get_aucs(model, data, label)

		# Include information in Clusters DataFrame.
		frame_clusters = include_coefficients(model, frame_clusters, features, label, groupby)

		# Confusion matrices.
		label_cms = get_confusion_matrix(model, data, label)
		cms[label] = label_cms

	aucs = total_aucs.mean(axis=0).tolist()
	return frame_clusters, aucs, cms

# Run logistic regression based on L1 penalty.
def run_logistic_regression(alphas, resolutions, meta_folder, meta_field, matching_field, folds_pickle, h5_complete_path, h5_additional_path, force_fold, additional_as_fold, diversity_key=None,
							use_conn=True, use_ratio=False, top_variance_feat=10, type_composition='clr', min_tiles=50, p_th=0.05):

	# If diversity key is not specified, use the key that represents samples.
	if diversity_key is None:
		diversity_key = matching_field

	# Get folds from existing split.
	folds = load_existing_split(folds_pickle)

	# Path for alpha Logistic Regression results.
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')

	# Loading data first.
	print('Loading data:')
	data_res_folds = dict()
	for resolution in resolutions:
		groupby = 'leiden_%s' % resolution
		print('\tResolution', groupby)
		data_res_folds[resolution] = dict()
		for i, fold in enumerate(folds):
			# Read CSV files for train, validation, test, and additional sets.
			dataframes, complete_df, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold=additional_as_fold, force_fold=force_fold)
			train_df, valid_df, test_df, additional_df = dataframes

			# Check clusters and diversity within.
			frame_clusters, frame_samples = create_frames(complete_df, groupby, meta_field, diversity_key=matching_field, reduction=2)

			# Create representations per sample: cluster % of total sample.
			data, data_df, features = prepare_data_classes(dataframes, matching_field, meta_field, groupby, leiden_clusters, type_composition, min_tiles,
														   use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat)

			# Include features that are not the regular leiden clusters.
			frame_clusters = include_features_frame_clusters(frame_clusters, leiden_clusters, features, groupby)

			# Store representations.
			data_res_folds[resolution][i] = {'data':data, 'features':features, 'frame_clusters':frame_clusters, 'leiden_clusters':leiden_clusters}

			# Information.
			print('\t\tFold', i, 'Features:', len(features), 'Clusters:', len(leiden_clusters))

	print()

	# For alpha value
	print('Running logistic regression:')
	for alpha in alphas:
		# Output path for alpha penalty.
		alpha_path = os.path.join(main_cluster_path, 'alpha_%s_mintiles_%s' % (str(alpha).replace('.', 'p'), min_tiles))
		if not os.path.isdir(alpha_path):
			os.makedirs(alpha_path)

		print('\tAlpha', alpha)
		all_data = list()
		for resolution in resolutions:
			groupby = 'leiden_%s' % resolution
			print('\t\tResolution', groupby)
			for i, fold in enumerate(folds):
				# Load data for classification.
				data            = data_res_folds[resolution][i]['data']
				features        = data_res_folds[resolution][i]['features']
				frame_clusters  = data_res_folds[resolution][i]['frame_clusters']
				leiden_clusters = data_res_folds[resolution][i]['leiden_clusters']

				# Logistic regression
				try:
					frame_clusters, aucs, cms = classification_performance_stats(data, leiden_clusters, frame_clusters, features, groupby, alpha=alpha)
					print('\t\t\tFold %s %-3s features Train/Validation/Test/Additional AUCs:' % (i, len(features)), np.round(aucs,2))
				except Exception as ex:
					print('\t\tIssue running logistic regression for resolution %s, alpha %s, and fold %s.' % (resolution, alpha, i))
					if hasattr(ex, 'message'):
						print('\t\tException:', ex.message)
					else:
						print('\t\tException:', ex)
					continue

				# Keep track of data.
				all_data.append([groupby, i] + aucs)
				frame_clusters.to_csv(os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i)), index=False)

				# Report forest plots and confusion matrices for logistic regression.
				report_forest_plot_lr(meta_field, frame_clusters, directory=alpha_path, file_name='%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i))
			# plot_confusion_matrix_lr(cms, directory=alpha_path, file_name='%s_fold%s_cm.jpg' % (str(groupby).replace('.', 'p'), i))

			if force_fold is not None:
				summarize_stat_clusters_across_folds(data, meta_field, frame_clusters, alpha, groupby, folds, alpha_path, p_th)

		# Save to CSV.
		columns = ['Leiden Resolution', 'Fold', 'Train AUC']
		if valid_df is not None: columns.append('Valid AUC')
		columns.append('Test AUC')
		if additional_df is not None: columns.append('Additional AUC')
		results_df = pd.DataFrame(all_data, columns=columns)
		results_df.to_csv(os.path.join(alpha_path, '%s_auc_results_mintiles_%s.csv' % (meta_field, min_tiles)), index=False)

		# Performance figures.
		box_plot_auc_results(frame=results_df, columns=columns, path_file=os.path.join(alpha_path, '%s_auc_results_mintiles_%s.csv' % (meta_field, min_tiles)))
