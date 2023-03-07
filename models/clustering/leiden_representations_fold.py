# Imports.
import scanpy as sc
import anndata
import argparse
import copy
import csv

# Add project path
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

# Data science packages.
import seaborn as sns
import pandas as pd
import numpy as np

# Other libraries.
import random
import h5py
import os
import gc 

# Own libs.
from models.evaluation.folds import load_existing_split


def adata_to_csv(adata, main_cluster_path, adata_name):
    current_df = pd.DataFrame()
    for column in adata.obs.columns:
        current_df[column] = adata.obs[column].values
    current_df.to_csv(os.path.join(main_cluster_path, '%s.csv' % adata_name), index=False)
    

def representations_to_frame(h5_path, rep_key='z_latent'):
	if h5_path is not None:
		with h5py.File(h5_path, 'r') as content:
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
				frame[key] = content[key][:].astype(str)
				rest_columns.append(key)
	else:
		frame, dim_columns, rest_columns = None, None, None

	return frame, dim_columns, rest_columns


def run_clustering(frame, dim_columns, rest_columns, resolution, groupby, n_neighbors, main_cluster_path, adata_name, subsample=None, save_adata=False, tabs='\t\t'):

	if subsample is not None: 
		print('%sSubsampling DataFrame to %s samples' % (tabs, subsample))
		frame_sub = frame.sample(n=subsample, random_state=1)
	else:
		frame_sub = frame.copy(deep=True)

	print('%s%s File' % (tabs, adata_name))
	adata = anndata.AnnData(X=frame_sub[dim_columns].to_numpy(), obs=frame_sub[rest_columns])
	# Nearest Neighbors
	print('%sPCA' % tabs)
	sc.tl.pca(adata, svd_solver='arpack', n_comps=adata.X.shape[1]-1)
	print('%sNearest Neighbors' % tabs)
	sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.X.shape[1]-1, method='umap', metric='euclidean', key_added='nn_leiden')
	print('%sLeiden' % tabs, resolution)
	sc.tl.leiden(adata, resolution=resolution, key_added=groupby, neighbors_key='nn_leiden')
	adata_to_csv(adata, main_cluster_path, adata_name)
	if save_adata:
		adata.write(os.path.join(main_cluster_path, adata_name) + '.h5ad', compression='gzip')
	print()
	return adata

def assign_clusters(frame, dim_columns, rest_columns, groupby, adata, main_cluster_path, adata_name, save_adata=False, tabs='\t\t'):
	print('%s%s File' % (tabs, adata_name))
	adata_test  = anndata.AnnData(X=frame[dim_columns].to_numpy(), obs=frame[rest_columns])
	print('%sNearest Neighbors on data' % tabs)
	sc.tl.ingest(adata_test, adata, obs=groupby, embedding_method='pca', neighbors_key='nn_leiden')
	adata_to_csv(adata_test, main_cluster_path, adata_name)
	if save_adata:
		adata_test.write(os.path.join(main_cluster_path, adata_name + '.h5ad'), compression='gzip')
	print()
	del adata_test

########### Code to introduce into run_representationsleiden.py if needed ###########
# After print('\tFold', i) if needed.
# This needs a fix here, AnnData has a memory leak issue: https://github.com/theislab/anndata/issues/360
# script_path = os.path.dirname(os.path.realpath(__file__))
# command = 'python3 %s/leiden_representations_fold.py --meta_name %s ---matching_field %s --fold %s --resolution %s --n_neighbors %s \
# 		   --subsample %s --rep_key %s --folds_pickle %s --h5_complete_path %s --h5_additional_path %s ' % (script_path, meta_field, matching_field, i, \
# 		   resolution, n_neighbors, subsample, rep_key, folds_pickle, h5_complete_path, h5_additional_path)
# if save_adata:
# 	command += '--save_adata'
# os.system(command)

##### Main #######
parser = argparse.ArgumentParser(description='Script to combine all H5 representation file into a \'complete\' one.')
parser.add_argument('--meta_name',           dest='meta_name',           type=str,             required=True,  help='Purpose of the clustering, name of folder.')
parser.add_argument('--matching_field',      dest='matching_field',      type=str,             required=True,  help='Key used to match folds split and H5 representation file.')
parser.add_argument('--fold',                dest='fold',                type=int,             required=True,  help='Fold number to run flow on.')
parser.add_argument('--resolution',          dest='resolution',          type=int,             required=True,  help='Resolution for Leiden algorithm.')
parser.add_argument('--n_neighbors',         dest='n_neighbors',         type=int,             required=True,  help='Number of neighbors to use for Leiden.')
parser.add_argument('--subsample',           dest='subsample',           type=int,             required=True,  help='Number of samples to use on Leiden (given memory constrains).')
parser.add_argument('--rep_key',             dest='rep_key',             type=str,             required=True,  help='Key pattern for representations to grab: z_latent, h_latent.')
parser.add_argument('--folds_pickle',        dest='folds_pickle',        type=str,             required=True,  help='Pickle file with folds information.')
parser.add_argument('--h5_complete_path',    dest='h5_complete_path',    type=str,             required=True,  help='H5 file path to run the leiden clustering folds.')
parser.add_argument('--h5_additional_path',  dest='h5_additional_path',  type=str,             default=None,   help='Additional H5 representation to assign leiden clusters.')
parser.add_argument('--save_adata',          dest='save_adata',          action='store_true',  default=False,  help='Save AnnData file for each fold.')
args               = parser.parse_args()
meta_name          = args.meta_name
matching_field     = args.matching_field
fold               = args.fold
resolution         = args.resolution
n_neighbors        = args.n_neighbors
subsample          = args.subsample
rep_key            = args.rep_key
folds_pickle       = args.folds_pickle
main_path          = args.main_path
h5_complete_path   = args.h5_complete_path
h5_additional_path = args.h5_additional_path
save_adata         = args.save_adata

# Get folds from existing split.
folds = load_existing_split(folds_pickle)

complete_frame,   complete_dims,   complete_rest   = representations_to_frame(h5_complete_path,   rep_key=rep_key)
additional_frame, additional_dims, additional_rest = representations_to_frame(h5_additional_path, rep_key=rep_key)

# Setup folder esqueme
main_cluster_path = h5_complete_path.split('hdf5_')[0]
main_cluster_path = os.path.join(main_cluster_path, meta_name)
main_cluster_path = os.path.join(main_cluster_path, 'adatas')

print('Leiden %s' % resolution)
groupby = 'leiden_%s' % resolution
print('\tFold', fold)

train_samples, valid_samples, test_samples = folds[fold]

# Train set.
train_frame = complete_frame[complete_frame[matching_field].isin(train_samples)]
adata_name  = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold)
adata_train = run_clustering(train_frame, complete_dims, complete_rest, resolution, groupby, n_neighbors, main_cluster_path, '%s_train_subsample' % adata_name, subsample=subsample, save_adata=True)
if subsample is not None: 
	assign_clusters(train_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_train' % adata_name, save_adata=save_adata)

# Validation set.
if len(valid_samples) > 0:
	valid_frame = complete_frame[complete_frame[matching_field].isin(valid_samples)]
	assign_clusters(valid_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_valid' % adata_name, save_adata=save_adata)

# Test set.
test_frame = complete_frame[complete_frame[matching_field].isin(test_samples)]
assign_clusters(test_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_test' % adata_name, save_adata=save_adata)

if additional_frame is not None:
	adata_name  = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold)
	assign_clusters(additional_frame, additional_dims, additional_rest, groupby, adata_train, main_cluster_path, adata_name, save_adata=save_adata)

del adata_train
