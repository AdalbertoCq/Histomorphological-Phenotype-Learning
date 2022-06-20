# Imports.
import scanpy as sc
import anndata
import copy
import csv

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
from models.clustering.data_processing import *
from models.visualization.attention_maps import get_x_y


# For each slide tile, it finds and includes cluster assignations for surrounding tiles.
def mapping_ABC(frame, groupby, index_string, x, y, tiles_key):
    '''
    A B C
    D X E
    F G H
    '''
    if 'A' == index_string:
        x -= 1
        y += 1
    elif 'B' == index_string:
        y += 1
    elif 'C' == index_string:
        x += 1
        y += 1
    elif 'D' == index_string:
        x -= 1
    elif 'E' == index_string:
        x += 1
    elif 'F' == index_string:
        x -= 1
        y -= 1
    elif 'G' == index_string:
        y -= 1
    elif 'H' == index_string:
        x += 1
        y -= 1

    if '.' in frame[tiles_key].values[0]:
        g = frame[(frame[tiles_key].values.astype(str) == '%s_%s.jpeg' % (x, y))][groupby]
    else:
        g = frame[(frame[tiles_key].values.astype(str) == '%s_%s' % (x, y))][groupby]
    if g.shape[0] == 0:
        return None
    return g.values[0]


# Takes a frame and includes surrounding tiles connections.
def include_tile_connections(groupby, main_cluster_path, adata_name):
    frame = pd.read_csv(os.path.join(main_cluster_path, '%s.csv' % adata_name))
    frame = frame.sort_values(by='slides')

    tiles_key = None
    for key in frame.columns:
        if 'tiles' in key:
            tiles_key = key
            break

    mappings = dict()
    mappings['x'] = list()
    mappings['y'] = list()
    for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        mappings[s] = list()

    all_tiles = frame[tiles_key].values.tolist()
    all_slides = frame['slides'].values.tolist()

    slide = None
    for i, fields in enumerate(zip(all_tiles, all_slides)):
        x, y = get_x_y(fields[0])
        mappings['x'].append(x)
        mappings['y'].append(y)

        if slide != fields[1]:
            slide = fields[1]
            slide_frame = frame[frame['slides'] == fields[1]]

        for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            next_cluster = mapping_ABC(frame=slide_frame, groupby=groupby, index_string=s, x=x, y=y, tiles_key=tiles_key)
            mappings[s].append(next_cluster)

    frame['x'] = np.array(mappings['x'], dtype=int)
    frame['y'] = np.array(mappings['y'], dtype=int)
    for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        frame[s] = mappings[s]

    frame.to_csv(os.path.join(main_cluster_path, '%s.csv' % adata_name), index=False)


# Takes a frame and includes surrounding tiles connections.
def include_tile_connections_frame(frame, groupby):
    frame_all = frame.copy(deep=True)
    frame_all = frame_all.sort_values(by='slides')

    tiles_key = None
    for key in frame_all.columns:
        if 'tiles' in key:
            tiles_key = key
            break

    mappings = dict()
    mappings['x'] = list()
    mappings['y'] = list()
    for s in ['A','B','C','D','E','F','G','H']:
        mappings[s] = list()

    all_tiles  = frame_all[tiles_key].values.tolist()
    all_slides = frame_all['slides'].values.tolist()

    slide = None
    for i, fields in enumerate(zip(all_tiles, all_slides)):
        x, y = get_x_y(fields[0])
        mappings['x'].append(x)
        mappings['y'].append(y)

        if slide != fields[1]:
            slide = fields[1]
            slide_frame = frame_all[frame_all['slides']==fields[1]]

        for s in ['A','B','C','D','E','F','G','H']:
            next_cluster = mapping_ABC(frame=slide_frame, groupby=groupby, index_string=s, x=x, y=y, tiles_key=tiles_key)
            mappings[s].append(next_cluster)

    frame_all['x'] = np.array(mappings['x'], dtype=int)
    frame_all['y'] = np.array(mappings['y'], dtype=int)
    for s in ['A','B','C','D','E','F','G','H']:
        frame_all[s] = mappings[s]

    return frame_all


# Sanity check, some representation instances introduce a bug where the # NN is smaller than specified and breaks scanpy
# line 286, in _init_pynndescent: init_indices = np.hstack((first_col, np.stack(distances.tolil().rows))) --> These rows have different lengths.
# ValueError: all input arrays must have the same shape
def sanity_check_neighbors(adata, dim_columns, neighbors_key='nn_leiden', tabs='\t\t'):
    distances = adata.obsp['%s_distances' % neighbors_key]
    n_neighbors = adata.uns['nn_leiden']['params']['n_neighbors']

    # Get locations with issues.
    original_frame_locs = list()
    i = 0
    for row in distances.tolil().rows:
        if len(row) != n_neighbors - 1:
            original_frame_locs.append(i)
        i += 1

    # If there's no problematic instances, continue
    if len(original_frame_locs) == 0:
        return False, None

    print('%sFound %s problematic instances' % (tabs, len(original_frame_locs)))
    print('%sRe-running clustering.' % tabs)
    # Recover from adata
    frame_sub = pd.DataFrame(adata.X, columns=dim_columns)
    for column in adata.obs.columns:
        frame_sub[column] = adata.obs[column].values

    # Drop problematic instances
    frame_sub = frame_sub.drop(original_frame_locs)
    return True, frame_sub


# Runs clustering flow on given frame.
def run_clustering(frame, dim_columns, rest_columns, resolution, groupby, n_neighbors, main_cluster_path, adata_name, subsample=None, include_connections=False, save_adata=False, tabs='\t\t'):
    # Handling subsampling.
    subsample_orig = subsample
    if subsample is None or subsample > frame.shape[0]:
        subsample      = int(frame.shape[0])
        adata_name     = adata_name.replace('_subsample', '')

    print('%sSubsampling DataFrame to %s samples' % (tabs, subsample))
    frame_sub = frame.sample(n=subsample, random_state=1)

    problematic_flag = True
    while problematic_flag:
        problematic_flag = False
        print('%s%s File' % (tabs, adata_name))
        adata = anndata.AnnData(X=frame_sub[dim_columns].to_numpy(), obs=frame_sub[rest_columns].astype('category'))
        # Nearest Neighbors
        print('%sPCA' % tabs)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=adata.X.shape[1] - 1)
        print('%sNearest Neighbors' % tabs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.X.shape[1] - 1, method='umap', metric='euclidean', key_added='nn_leiden')

        # Sanity check, some representation instances introduce a bug where the # NN is smaller than specified and breaks scanpy
        problematic_flag, frame_sub = sanity_check_neighbors(adata, dim_columns, neighbors_key='nn_leiden', tabs=tabs)

    # Leiden clustering.
    print('%sLeiden' % tabs, resolution)
    sc.tl.leiden(adata, resolution=resolution, key_added=groupby, neighbors_key='nn_leiden')

    # Save to csv.
    adata_to_csv(adata, main_cluster_path, adata_name)

    # Looks and dumps surrounding tile leiden connections per tile.
    if include_connections: include_tile_connections(groupby, main_cluster_path, adata_name)

    # Keep H5ad file.
    if save_adata: adata.write(os.path.join(main_cluster_path, adata_name) + '.h5ad', compression='gzip')
    print()

    # adata = anndata.read_h5ad(os.path.join(main_cluster_path, adata_name) + '.h5ad')

    if subsample_orig is None or subsample_orig > frame.shape[0]:
        subsample = None

    return adata, subsample


# Assign cluster to representation give a frame reference.
def assign_clusters(frame, dim_columns, rest_columns, groupby, adata, main_cluster_path, adata_name, include_connections=False, save_adata=False, tabs='\t\t'):
    # Crete AnnData based on frame.
    print('%s%s File' % (tabs, adata_name))
    adata_test = anndata.AnnData(X=frame[dim_columns].to_numpy(), obs=frame[rest_columns].astype('category'))

    # Assign cluster based on nearest neighbors from reference frame assignations.
    print('%sNearest Neighbors on data' % tabs)
    sc.tl.ingest(adata_test, adata, obs=groupby, embedding_method='pca', neighbors_key='nn_leiden')

    # Save to csv.
    adata_to_csv(adata_test, main_cluster_path, adata_name)

    # Looks and dumps surrounding tile leiden connections per tile.
    if include_connections:
        include_tile_connections(groupby, main_cluster_path, adata_name)

    # Keep H5ad file.
    if save_adata:
        adata_test.write(os.path.join(main_cluster_path, adata_name + '.h5ad'), compression='gzip')
    print()
    del adata_test


# Flow to assign clusters to an additional h5 file.
def assign_additional_only(meta_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions, include_connections=False, save_adata=False):
    # Get folds from existing split.
    folds = load_existing_split(folds_pickle)

    additional_frame, additional_dims, additional_rest = representations_to_frame(h5_additional_path, meta_field=meta_field, rep_key=rep_key)

    # Setup folder esqueme
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_field)
    main_cluster_path = os.path.join(main_cluster_path, 'adatas')

    if not os.path.isdir(main_cluster_path):
        print('Clustering run not found:')
        print('\t%s' % main_cluster_path)

    print()
    for resolution in resolutions:
        print('Leiden %s' % resolution)
        groupby = 'leiden_%s' % resolution
        for i, fold in enumerate(folds):
            print('\tFold', i)

            ### Train set.
            failed = False
            try:
                adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                h5_path = os.path.join(main_cluster_path, adata_name) + '.h5ad'
                if os.path.isfile(h5_path):
                    adata_train = anndata.read_h5ad(h5_path)
                else:
                    adata_train = anndata.read_h5ad(h5_path.replace('.h5ad', '_subsample.h5ad'))
            except:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                failed = True
            finally:
                gc.collect()

            # Do not even try if train failed.
            if failed:
                continue

            try:
                adata_name = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                assign_clusters(additional_frame, additional_dims, additional_rest, groupby, adata_train, main_cluster_path, adata_name, include_connections=include_connections,
                                save_adata=save_adata)
            except:
                print('\t\tIssue running Leiden %s on fold %s Additinal Set' % (resolution, i))

            del adata_train
            gc.collect()


# Full flow to cluster based on leiden resolutions and folds.
def run_leiden(meta_field, matching_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions, n_neighbors=250, subsample=200000, include_connections=False, save_adata=False):
    # Get folds from existing split.
    folds = load_existing_split(folds_pickle)

    complete_frame, complete_dims, complete_rest = representations_to_frame(h5_complete_path, meta_field=meta_field, rep_key=rep_key)
    additional_frame, additional_dims, additional_rest = representations_to_frame(h5_additional_path, meta_field=meta_field, rep_key=rep_key)

    # Setup folder esqueme
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_field)
    main_cluster_path = os.path.join(main_cluster_path, 'adatas')

    if not os.path.isdir(main_cluster_path):
        os.makedirs(main_cluster_path)

    print()
    for resolution in resolutions:
        print('Leiden %s' % resolution)
        groupby = 'leiden_%s' % resolution
        for i, fold in enumerate(folds):
            print('\tFold', i)

            # Fold split.
            train_samples, valid_samples, test_samples = fold

            ### Train set.
            failed = False
            try:
                train_frame = complete_frame[complete_frame[matching_field].isin(train_samples)]
                adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                adata_train, subsample = run_clustering(train_frame, complete_dims, complete_rest, resolution, groupby, n_neighbors, main_cluster_path, '%s_subsample' % adata_name,
                                                        subsample=subsample, include_connections=include_connections, save_adata=True)
                if subsample is not None:
                    assign_clusters(train_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, adata_name, include_connections=include_connections, save_adata=save_adata)
            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                failed = True
                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)
            finally:
                gc.collect()

            # Do not even try if train failed.
            if failed:
                continue

            ### Validation set.
            try:
                if len(valid_samples) > 0:
                    valid_frame = complete_frame[complete_frame[matching_field].isin(valid_samples)]
                    assign_clusters(valid_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_valid' % adata_name, include_connections=include_connections,
                                    save_adata=save_adata)
            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)

            ### Test set.
            try:
                test_frame = complete_frame[complete_frame[matching_field].isin(test_samples)]
                assign_clusters(test_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_test' % adata_name, include_connections=include_connections,
                                save_adata=save_adata)
            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)

            ### Additional set.
            if additional_frame is not None:
                try:
                    adata_name = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                    assign_clusters(additional_frame, additional_dims, additional_rest, groupby, adata_train, main_cluster_path, adata_name, include_connections=include_connections,
                                    save_adata=save_adata)
                except Exception as ex:
                    print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                    if hasattr(ex, 'message'):
                        print('\t\tException', ex.message)
                    else:
                        print('\t\tException', ex)

            del adata_train
            gc.collect()
