# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
import numpy as np
import os

# Survival Libraries
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored

# Own libraries.
from models.clustering.data_processing import *
from models.evaluation.folds import load_existing_split
from models.visualization.survival import save_fold_KMs
from models.visualization.forest_plots import report_forest_plot_cph, summary_cox_forest_plots

# Summarize results for the Cox Individual run.
def summarize_stat_clusters_across_folds(event_ind_field, test_ci, additional_ci, test_pval, additional_pval, alpha, groupby, folds, meta_folder, alpha_path, p_th):
	from scipy.stats import combine_pvalues

	cox_folds = list()
	counts    = list()
	for i, _ in enumerate(folds):
		cox_run = pd.read_csv(os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i)))
		cox_run['se'] = (cox_run['coef']-cox_run['coef lower 95%'])/1.95996
		cox_folds.append(cox_run)
		counts.append(len(folds[i][0]))
	counts    = np.array(counts)
	cox_folds = pd.concat(cox_folds, axis=0)
	cox_coef_pvals = cox_folds[[groupby, 'p']].copy(deep=True)
	cox_coef       = cox_folds[[groupby, 'coef']].copy(deep=True)
	cox_coef_se    = cox_folds[[groupby, 'se']].copy(deep=True)

	cox_folds = cox_folds.groupby(groupby).mean().reset_index()
	cox_folds = cox_folds.sort_values(by=groupby)

	# Combine coefficients and standard errors for folds.
	avgs = list()
	ses  = list()
	for i in cox_folds[groupby]:
		average = np.average(cox_coef[cox_coef[groupby]==i]['coef'].values, weights=counts)
		std = np.sqrt(np.sum((counts - 1) * (cox_coef_se[cox_coef_se[groupby]==i]['se'].values**2))/(np.sum(counts)-counts.shape[0]))
		avgs.append(average)
		ses.append(std)
	cox_folds['coef'] = avgs
	cox_folds['se']   = ses
	cox_folds['coef lower 95%'] = cox_folds['coef'] - (1.95996*cox_folds['se'])
	cox_folds['coef upper 95%'] = cox_folds['coef'] + (1.95996*cox_folds['se'])

	cox_folds['p'] = cox_coef_pvals.groupby(groupby).apply(lambda x: combine_pvalues(x['p'], method='fisher')[1]).reset_index()[0]
	cox_folds = cox_folds.sort_values(by='coef')
	csv_path  = os.path.join(alpha_path, '%s_stats_all_folds.csv' % (str(groupby).replace('.', 'p')))
	cox_folds.to_csv(csv_path, index=False)
	report_forest_plot_cph('os_event_ind', cox_folds, csv_path, p_th=p_th)

	# Save performance and p-value for test and additional set.
	columns = cox_folds.columns
	complete_df = list()
	complete_df.append(['Test']+['']*(len(columns)-1))
	complete_df.append(['', 'C-Index CI', test_ci[0], test_ci[1], test_ci[2]]+['']*(len(columns)-5))
	complete_df.append(['', 'P-Value',  test_pval]+['']*(len(columns)-3))
	complete_df.append(['', 'Alpha L2', alpha]+['']*(len(columns)-3))
	if additional_ci is not None:
		complete_df.append(['Additional']+['']*(len(columns)-1))
		complete_df.append(['', 'C-Index CI', additional_ci[0], additional_ci[1], additional_ci[2]]+['']*(len(columns)-5))
		complete_df.append(['', 'P-Value', additional_pval]+['']*(len(columns)-3))
	complete_df = pd.DataFrame(complete_df, columns=columns)
	complete_df = pd.concat([complete_df, cox_folds[cox_folds['p']<p_th]])
	complete_df.to_csv(os.path.join(alpha_path, '%s_stats_performance.csv' % (str(groupby).replace('.', 'p'))), index=False)

# Calculate mean and 95% confidence intervals.
# Using bootstrap to calculate CI.
def mean_confidence_interval(data, confidence=0.95):
	boots = sns.algorithms.bootstrap(data, func=np.mean, n_boot=1000, units=None, seed=None)
	minus, plus = sns.utils.ci(boots, which=confidence*100)
	mean = np.mean(data)
	return np.array([mean, minus, plus])

# Train lifelines CPH model.
def train_cox(datas, penalizer, l1_ratio, event_ind_field='event_ind', event_data_field='event_data', robust=True, frame_clusters=None, groupby=None):
	# Train Cox Proportional Hazard.
	train, set_name = datas[0]
	cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
	cph.fit(train, duration_col=event_data_field, event_col=event_ind_field, show_progress=False, robust=robust)

	# Partial hazard prediction for each list.
	predictions = list()
	for data, set_name in datas:
		if data is not None:
			pred = cph.predict_partial_hazard(data)
		else:
			pred = None
		predictions.append((pred, set_name))

	summary_table = cph.summary
	if frame_clusters is not None:
		frame_clusters = frame_clusters.sort_values(by=groupby)
		for column in ['coef', 'coef lower 95%', 'coef upper 95%', 'p']:
			for cluster_id in [col for col in train if col not in [event_ind_field, event_data_field]]:
				frame_clusters.loc[frame_clusters[groupby]==cluster_id, column] = summary_table.loc[cluster_id, column].astype(np.float32)
		frame_clusters = frame_clusters.sort_values(by='coef')

	return cph, predictions, frame_clusters

# Evaluate survival model: C-Index.
def evalutaion_survival(datas, predictions, event_ind_field='event_ind', event_data_field='event_data', c_index_type='Harrels'):
	cis = list()
	for i, data_i in enumerate(datas):
		data, set_named = data_i
		if data is not None:
			prediction, set_namep = predictions[i]
			# Concordance index for right-censored data.
			if c_index_type=='Harrels':
				c_index = np.round(concordance_index_censored(data[event_ind_field]==1.0, data[event_data_field], prediction)[0], 2)
			# Concordance index for right-censored data based on inverse probability of censoring weights
			elif c_index_type=='ipcw':
				train_data = datas[0][0].copy(deep=True)
				train_data[event_ind_field]  = train_data[event_ind_field].astype(bool)
				train_data[event_data_field] = train_data[event_data_field].astype(float)
				data[event_ind_field]       = data[event_ind_field].astype(bool)
				data[event_data_field]      = data[event_data_field].astype(float)
				c_index = np.round(concordance_index_ipcw(survival_train=train_data[[event_ind_field,event_data_field]].to_records(index=False), survival_test=data[[event_ind_field,event_data_field]].to_records(index=False), estimate=prediction)[0], 2)
			if set_namep != set_named:
				print('Mismatch between adata and predictions')
				print('Data set:', set_named, 'Prediction set:', set_namep)
				exit()
		else:
			c_index   = None
			set_namep = data_i[1]
		cis.append((c_index, set_namep))
	return cis

# Divide group into X buckets.
def get_high_low_risks(predictions, datas, fold, matching_field, q_buckets=2):
	labels_buckets = list(range(q_buckets))
	high_lows = list()
	for index, set_data in enumerate(zip(datas, predictions)):
		data, prediction = set_data
		if data[0] is None:
			high_lows.append((None, None, None))
			continue
		current_hazard                     = data[0].copy(deep=True)
		current_hazard['hazard']           = prediction[0]
		if index == 0:
			median_cutoff = current_hazard['hazard'].median()
		current_hazard['h_bin']            = (current_hazard['hazard']>median_cutoff)*1
		current_hazard['h_bin_%s' % fold]  = (current_hazard['hazard']>median_cutoff)*1
		current_low_risk_slides            = current_hazard[current_hazard['h_bin']==labels_buckets[0]][matching_field].values
		current_high_risk_slides           = current_hazard[current_hazard['h_bin']==labels_buckets[1]][matching_field].values

		high_risk_df = current_hazard[current_hazard[matching_field].isin(current_high_risk_slides)].copy(deep=True)
		low_risk_df  = current_hazard[current_hazard[matching_field].isin(current_low_risk_slides)].copy(deep=True)
		high_lows.append((low_risk_df, high_risk_df, current_hazard))
	return high_lows

# Combine Risk Groups over the folds. In the case of additional dataset, mayority vote over folds.
def combine_risk_groups(risk_groups, additional_risk, high_lows, fold, num_folds, matching_field, event_ind_field='event_ind', event_data_field='event_data'):
	risk_groups[1] = risk_groups[1].append(high_lows[2][1], ignore_index=True)
	risk_groups[0] = risk_groups[0].append(high_lows[2][0], ignore_index=True)

	if high_lows[3][2] is not None:
		if fold == 0:
			additional_risk = additional_risk.append(high_lows[3][2][[matching_field, event_data_field, event_ind_field]], ignore_index=True)
		additional_risk['h_bin_%s'%fold] = high_lows[3][2]['h_bin_%s'%fold].values

		if fold == num_folds-1:
			additional_risk['risk'] = (np.mean(additional_risk[['h_bin_%s'%fold for fold in range(num_folds)]], axis=1)>0.5)*1
			high_risk_df = additional_risk[additional_risk[matching_field].isin(additional_risk[additional_risk['risk']==1][matching_field].values)].copy(deep=True)
			low_risk_df  = additional_risk[additional_risk[matching_field].isin(additional_risk[additional_risk['risk']==0][matching_field].values)].copy(deep=True)
			additional_risk     = [low_risk_df, high_risk_df]
	else:
		additional_risk = None
	return risk_groups, additional_risk

# Save data into CSV.
def  keep_track_data(resolution, alpha, fold, cis, cox_data, l1_ratio, min_tiles, meta_folder, results_path):
	row = [resolution, alpha, fold]
	col = ['resolution', 'alpha', 'fold']
	for c_index, set_name in cis:
		if c_index is not None:
			row.append(c_index)
			col.append('c_index_%s' % set_name)
	cox_data.append(row)

	cox_ridge_data_df = pd.DataFrame(cox_data, columns=col)
	cox_ridge_data_df.to_csv(os.path.join(results_path, 'c_index_%s_l1_ratio_%s_mintiles_%s.csv' % (meta_folder, l1_ratio, min_tiles)), index=False)

	return cox_data

# C-Index mean and confidence interval figure.
def mean_ci_cox(all_data, results_path_csv, ylim=[0.4, 1.0]):
	sns.set_theme(style='darkgrid')
	fig, ax = plt.subplots(figsize=(20, 7), nrows=1, ncols=1)
	meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"}
	sns.pointplot(x='Resolution', hue='Set', y='C-Index', data=all_data, ax=ax, linewidth=0.01, dodge=.3, join=False, capsize=.04, markers='s', ci=95)
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_title('Leiden + Cox Regression', fontweight='bold', fontsize=18)
	ax.legend(loc='upper left')
	start, end = ax.get_ylim()
	ax.yaxis.set_ticks(np.arange(start, end, 0.05))
	plt.savefig(results_path_csv.replace('.csv', '.jpg'))

# Summarize c-index across penalties, panel of figures.
def summarize_cindex_diff_clusters(cox_ridge_data_df, resolutions, results_path_csv):
	sets_data = [column for column in cox_ridge_data_df.columns if 'c_index' in column]

	all_data = list()
	for resolution in resolutions:
		frame_resolution = cox_ridge_data_df[cox_ridge_data_df.resolution==resolution]
		for fold_n in np.unique(cox_ridge_data_df.fold):
			frame_fold = frame_resolution[frame_resolution.fold==fold_n]
			index_value = frame_fold.index[frame_fold['c_index_test'].max()==frame_fold['c_index_test']].tolist()[0]
			fold_row = [resolution, fold_n]
			for c_index_field in sets_data:
				figure_set_name = c_index_field.replace('c_index', 'C-Index')
				figure_set_name = figure_set_name.replace('_', ' ')
				all_data.append(fold_row + [figure_set_name, frame_fold.loc[index_value, c_index_field], frame_fold.loc[index_value, 'alpha']])
	all_data = pd.DataFrame(all_data, columns=['Resolution', 'Fold', 'Set', 'C-Index', 'Alpha'])
	all_data.to_csv(results_path_csv.replace('.csv', '_summary.csv'), index=False)

	mean_ci_cox(all_data, results_path_csv, ylim=[0.4, 1.0])

# Summarize c-index across penalties, panel of figures.
def summarize_cindex_same_clusters(cox_ridge_data_df, resolutions, results_path_csv):
	# Get set names with results.
	set_names = list()
	for column in cox_ridge_data_df.columns:
		if 'c_index' in column:
			set_names.append(column)

	# Figure
	sns.set_theme(style='darkgrid')
	mosaic = '''123
				456
				789'''
	fig     = plt.figure(figsize=(20,20), constrained_layout=True)
	ax_dict = fig.subplot_mosaic(mosaic, sharex=False, sharey=False)
	for j, resolution in enumerate(resolutions):
		groupby     = 'leiden_%s' % resolution
		alphas_df   = cox_ridge_data_df[cox_ridge_data_df.resolution==resolution]
		alphas      = np.unique(alphas_df.alpha.values)
		for c_index_set_name in set_names:
			set_name = c_index_set_name.split('c_index_')[1]
			confidence_interval  = np.array([mean_confidence_interval(alphas_df[alphas_df.alpha==alpha][c_index_set_name].values) for alpha in alphas])
			ax_dict[str(j+1)].plot(alphas, confidence_interval[:,0], label=set_name)
			ax_dict[str(j+1)].fill_between(alphas, confidence_interval[:,1], confidence_interval[:,2], alpha=.15)
			if 'test' in set_name:
				ax_dict[str(j+1)].axvline(alphas[np.argwhere(confidence_interval[:,0]==max(confidence_interval[:,0]))[0,0]], c="C1")
		ax_dict[str(j+1)].set_title(groupby, fontweight='bold')
		ax_dict[str(j+1)].set_xscale("log")
		ax_dict[str(j+1)].set_ylabel("concordance index")
		ax_dict[str(j+1)].set_xlabel("alpha")
		ax_dict[str(j+1)].axhline(0.5, color="grey", linestyle="--")
		ax_dict[str(j+1)].grid(True)
		ax_dict[str(j+1)].legend(loc='upper right')
		ax_dict[str(j+1)].set_ylim([0.25, 1.0])
	plt.savefig(results_path_csv.replace('.csv', '.jpg'))

# Create summary image for C-Index results.
def summary_resolution_cindex(resolutions, h5_complete_path, meta_folder, l1_ratios, min_tiles, force_fold):
	for l1_ratio in l1_ratios:
		# Paths and result file.
		main_cluster_path = h5_complete_path.split('hdf5_')[0]
		main_cluster_path = os.path.join(main_cluster_path, meta_folder)
		results_path_csv  = os.path.join(main_cluster_path, 'c_index_%s_l1_ratio_%s_mintiles_%s.csv' % (meta_folder, l1_ratio, min_tiles))
		if not os.path.isfile(results_path_csv):
			print('File not found:', results_path_csv)
			continue
		cox_ridge_data_df = pd.read_csv(results_path_csv)

		if force_fold is not None:
			summarize_cindex_same_clusters(cox_ridge_data_df, resolutions, results_path_csv)
		else:
			summarize_cindex_diff_clusters(cox_ridge_data_df, resolutions, results_path_csv)

# Retrieve best alpha for a given resolution
def get_best_alpha(main_cluster_path, meta_folder, l1_ratio, min_tiles, resolution, force_fold, additional=False):
	results_path_csv  = os.path.join(main_cluster_path, 'c_index_%s_l1_ratio_%s_mintiles_%s.csv' % (meta_folder, l1_ratio, min_tiles))
	results           = pd.read_csv(results_path_csv)

	test_field = None
	for column in results.columns:
		if 'test' in column and not additional:
			test_field = column
			break
		if 'additional' in column and additional:
			test_field = column
			break

	results_resolution  = results[results.resolution==resolution]
	alphas              = np.unique(results_resolution.alpha.values)
	confidence_interval = np.array([mean_confidence_interval(results_resolution[results_resolution.alpha==alpha][test_field].values) for alpha in alphas])

	folds = np.unique(results_resolution.fold)
	if force_fold is not None:
		alpha = alphas[np.argwhere(confidence_interval[:,0]==max(confidence_interval[:,0]))[0,0]]
		alpha = dict(zip(list(range(len(folds))),[alpha]*len(folds)))
	else:
		alpha = dict()
		for fold in np.unique(folds):
			results_fold = results_resolution[results_resolution.fold==fold]
			alpha[fold]  = results_fold[results_fold[test_field]==max(results_fold[test_field])]['alpha'].values[0]

	return alpha, alphas, confidence_interval

# Run Cox Proportional Hazard regression for a penalty and resolution.
def run_cph_regression_individual(orig_alpha, resolution, meta_folder, matching_field, folds_pickle, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key,
								  type_composition, min_tiles, max_months, additional_as_fold, force_fold, l1_ratio=0.0, q_buckets=2, use_conn=False, use_ratio=False, top_variance_feat=10,
								  remove_clusters=None, p_th=0.05):
	groupby     = 'leiden_%s' % resolution

	# Get folds from existing split.
	folds     = load_existing_split(folds_pickle)
	num_folds = len(folds)

	# If diversity key is not specified, use the key that represents samples.
	if diversity_key is None:
		diversity_key = matching_field

	# Paths.
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')

	# Retrieve the best alpha performance.
	alpha, alphas, _ = get_best_alpha(main_cluster_path, meta_folder, l1_ratio, min_tiles, resolution, force_fold)

	# Particular run path
	alpha_path = os.path.join(main_cluster_path, '%s_%s_alpha_%s_l1ratio_%s_mintiles_%s' % (meta_folder, groupby, str(orig_alpha).replace('.','p'), str(l1_ratio).replace('.','p'), min_tiles))
	if not os.path.isdir(alpha_path): os.makedirs(alpha_path)

	# First run the Cox regression for the selected resolution
	run_cph_regression(alphas, [resolution], meta_folder, matching_field, folds, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key, type_composition,
					   min_tiles, max_months, additional_as_fold, force_fold, [l1_ratio], adatas_path, alpha_path, use_conn, use_ratio, top_variance_feat, remove_clusters)

	# Retrieve best alpha performance for previous run.
	# This allows from flexibility to check the impact of removing background/artifact clusters.
	alpha, alphas, confidence_interval = get_best_alpha(alpha_path, meta_folder, l1_ratio, min_tiles, resolution, force_fold)
	additional_confidence_interval     = None
	if h5_additional_path is not None and not additional_as_fold:
		_, _, additional_confidence_interval = get_best_alpha(alpha_path, meta_folder, l1_ratio, min_tiles, resolution, force_fold, additional=True)
	if orig_alpha is not None:
		alpha = dict(zip(list(range(len(folds))),[orig_alpha]*len(folds)))

	# Fold cross-validation performance.
	print('')
	print('\tResolution', resolution)
	risk_groups     = [pd.DataFrame(), pd.DataFrame()]
	additional_risk = pd.DataFrame()
	cis_folds       = list()
	estimators      = list()
	for i, fold in enumerate(folds):
		# Read CSV files for train, validation, test, and additional sets.
		dataframes, complete_df, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold, force_fold)

		# Check clusters and diversity within.
		frame_clusters, frame_samples = create_frames(complete_df, groupby, event_ind_field, diversity_key=matching_field, reduction=2)

		# Prepare data for COX.
		data, datas_all, features = prepare_data_survival(dataframes, groupby, leiden_clusters, type_composition, max_months, matching_field, event_ind_field, event_data_field, min_tiles,
														  use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat, remove_clusters=remove_clusters)

		# COX Regression
		estimator, predictions, frame_clusters = train_cox(data, penalizer=alpha[i], l1_ratio=l1_ratio, robust=True, event_ind_field=event_ind_field, event_data_field=event_data_field,
														   frame_clusters=frame_clusters, groupby=groupby)
		estimators.append(estimator)

		# Evaluation metrics.
		cis = evalutaion_survival(data, predictions, event_ind_field=event_ind_field, event_data_field=event_data_field)
		cis_folds.append([ci[0] for ci in cis])

		# Report forest plot for cox regression.
		frame_clusters.to_csv(os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i)), index=False)
		report_forest_plot_cph(event_ind_field, frame_clusters, os.path.join(alpha_path, '%s_fold%s_clusters.csv' % (str(groupby).replace('.', 'p'), i)), p_th=p_th)

		# High, low risk groups.
		high_lows = get_high_low_risks(predictions, datas_all, i, matching_field, q_buckets=q_buckets)
		risk_groups, additional_risk = combine_risk_groups(risk_groups, additional_risk, high_lows, i, num_folds, matching_field, event_ind_field, event_data_field)

		print('\t\tFold', i, 'Alpha', np.round(alpha[i],2), 'Train/Valid/Test/Additional C-Index:', '/'.join([str(i) for i in cis_folds[i]]))

	print()
	test_ci       = mean_confidence_interval([a[2] for a in cis_folds])
	additional_ci = None
	print('\tTest       Mean/Mean-2*Std/Mean+2*Std: %s' % np.round(test_ci,2))
	if dataframes[-1] is not None:
		additional_ci = mean_confidence_interval([a[3] for a in cis_folds])
		print('\tAdditional Mean/Mean-2*Std/Mean+2*Std: %s' % np.round(additional_ci,2))

	# Kaplan-Meier plots for train, valid, test, and additional sets.
	test_pval, additional_pval = save_fold_KMs(risk_groups, additional_risk, resolution, groupby, cis_folds, event_ind_field, event_data_field, max_months, alpha_path)

	# Save summary of the Cox Hazard Ratios.
	summary_cox_forest_plots(estimators, cis_folds, alpha, alphas, confidence_interval, additional_confidence_interval, groupby, alpha_path, force_fold)

	# Summarize Statistically relevant clusters across folds
	# This makes sense when clustering is done and maintained across folds.
	if force_fold is not None:
		summarize_stat_clusters_across_folds(event_ind_field, test_ci, additional_ci, test_pval, additional_pval, alpha, groupby, folds, meta_folder, alpha_path, p_th)

# Wrapper for a exhaustive Cox Proportional Hazard regression for a given group of penalties and resolutions.
def run_cph_regression_exhaustive(alphas, resolutions, meta_folder, matching_field, folds_pickle, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key, type_composition,
					   min_tiles, max_months, additional_as_fold, force_fold, l1_ratios, use_conn=True, use_ratio=False, top_variance_feat=10):
	# Get folds from existing split.
	folds = load_existing_split(folds_pickle)

	# If diversity key is not specified, use the key that represents samples.
	if diversity_key is None:
		diversity_key = matching_field

	# Paths.
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_folder)
	adatas_path       = os.path.join(main_cluster_path, 'adatas')

	run_cph_regression(alphas, resolutions, meta_folder, matching_field, folds, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key, type_composition,
					   min_tiles, max_months, additional_as_fold, force_fold, l1_ratios, adatas_path, main_cluster_path, use_conn, use_ratio, top_variance_feat)

# Runs Cox Proportional Hazard regression for a given group of penalties and resolutions.
def run_cph_regression(alphas, resolutions, meta_folder, matching_field, folds, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key, type_composition,
					   min_tiles, max_months, additional_as_fold, force_fold, l1_ratios, adatas_path, main_cluster_path, use_conn=True, use_ratio=False, top_variance_feat=10, remove_clusters=None):

	# Loading data first.
	print('Loading data:')
	data_res_folds = dict()
	for resolution in resolutions:
		groupby = 'leiden_%s' % resolution
		print('\tResolution', groupby)
		data_res_folds[resolution] = dict()
		for i, fold in enumerate(folds):
			# Read CSV files for train, validation, test, and additional sets.
			dataframes, _, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, i, fold, h5_complete_path, h5_additional_path, additional_as_fold, force_fold)
			# Prepare data for COX.
			data, datas_all, features = prepare_data_survival(dataframes, groupby, leiden_clusters, type_composition, max_months, matching_field, event_ind_field, event_data_field, min_tiles,
															  use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat, remove_clusters=remove_clusters)

			# Store representations.
			data_res_folds[resolution][i] = {'data':data, 'features':features}

			# Information
			print('\t\tFold', i, 'Features:', len(features), 'Clusters:', len(leiden_clusters))

	# Run Cox Proportional Hazard regression.
	for l1_ratio in l1_ratios:
		print('L1 Penalty', l1_ratio)
		cox_data = list()
		for resolution in resolutions:
			groupby     = 'leiden_%s' % resolution
			alphas_df   = list()
			print('\tResolution', resolution)
			for alpha in alphas:
				print('\t\tResolution', resolution, 'Alpha', alpha)
				for i, fold in enumerate(folds):

					# Load data.
					data     = data_res_folds[resolution][i]['data']
					features = data_res_folds[resolution][i]['features']

					# COX Regression
					estimator, predictions, _ = train_cox(data, penalizer=alpha, l1_ratio=l1_ratio, robust=True, event_ind_field=event_ind_field, event_data_field=event_data_field)

					# Evaluation metrics.
					cis = evalutaion_survival(data, predictions, event_ind_field=event_ind_field, event_data_field=event_data_field)
					print('\t\t\tFold %s %-3s features C-Index:' % (i, len(features)), cis)

					# Keep track of performance.
					cox_data = keep_track_data(resolution, alpha, i, cis, cox_data, l1_ratio, min_tiles, meta_folder, main_cluster_path)