from scipy.stats import spearmanr, pearsonr, hypergeom
import statsmodels.stats.multitest as smm
import pandas as pd
import numpy as np
import os


''' ############ Cluster correlations with annotations - Slide Representations ############ '''
def correlations(complete_df, clusters, anno_fields, corr_method, method):
    all_data = np.zeros((len(anno_fields), len(clusters), 2))
    for i, annotation in enumerate(anno_fields):
        for j, cluster in enumerate(clusters):
            corr_df = complete_df[[cluster, annotation]]
            corr_df = corr_df.dropna()
            if corr_method == 'spearman':
                rho, pval = spearmanr(corr_df[cluster], corr_df[annotation])
            elif corr_method == 'pearson':
                rho, pval = pearsonr(corr_df[cluster], corr_df[annotation])
            all_data[i,j, 0] = rho
            all_data[i,j, 1] = pval

    all_data_rho = pd.DataFrame(all_data[:,:,0], columns=clusters)
    all_data_rho.index = anno_fields

    all_data_pval = pd.DataFrame(all_data[:,:,1], columns=clusters)
    all_data_pval.index = anno_fields

    # p-value correction.
    shape = all_data_pval.values.shape
    flat_pvalues = all_data_pval.values.flatten()
    reject, pvals_corrected, _, _ = smm.multipletests(pvals=flat_pvalues, method=method)

    # Wrap into a dataframe.
    all_data_pval = pd.DataFrame(pvals_corrected.reshape(shape), columns=clusters)
    all_data_pval.index = anno_fields

    return all_data_rho, all_data_pval

def correlate_clusters_annotation(slide_rep_df, annotations_df, matching_field, purity_field, groupby, fold_number, directory, file_name,
                                  corr_method='spearman', method_correction='fdr_bh', field_th=0, pval_th=0.01):
    run_path          = os.path.join(directory, '%s_fold%s' % (groupby.replace('.','p'), fold_number))
    correlations_path = os.path.join(run_path, 'correlations')
    if not os.path.isdir(correlations_path):
        os.makedirs(correlations_path)

    # Grab clusters.
    clusters    = [str(cluster) for cluster in slide_rep_df.columns if cluster not in [purity_field, 'slides', 'samples']]
    # Grab annotations.
    anno_fields = [annotation for annotation in annotations_df.columns if annotation not in [purity_field, 'slides', 'samples']]
    # Combine into one data frame
    complete_df    = slide_rep_df.merge(annotations_df, how='inner', left_on=matching_field, right_on=matching_field)
    # Correlations.
    all_data_rho, all_data_pval = correlations(complete_df, clusters, anno_fields, corr_method=corr_method, method=method_correction)
    # Mask for p-value threshold.
    mask = (all_data_pval.values > pval_th)
    mask = pd.DataFrame(mask, columns=clusters)
    mask.index = anno_fields

    remove_fields = mask[np.logical_not(mask).sum(axis=1)<=field_th].index
    all_data_pval = all_data_pval.drop(index=remove_fields)
    all_data_rho  = all_data_rho.drop(index=remove_fields)
    mask          = mask.drop(index=remove_fields)

    all_data_rho.to_csv(os.path.join(correlations_path,  file_name+'_coef.csv'))
    all_data_pval.to_csv(os.path.join(correlations_path, file_name+'_pval.csv'))

    return all_data_rho, all_data_pval, mask, complete_df

def mask_cox_clusters(mask, cox_clusters):
    for cluster in mask.columns:
        if cluster not in cox_clusters:
            mask[cluster] = [True]*mask.shape[0]
    return mask

def correlate_clusters_occurrance_annotation(slide_rep_df, purity_field, groupby, fold_number, directory, file_name,
                                             corr_method='spearman', method_correction='fdr_bh', pval_th=0.01):
    def correlations(complete_df, clusters, anno_fields, corr_method, method):
        all_data = np.zeros((len(anno_fields), len(clusters), 2))
        for i, annotation in enumerate(anno_fields):
            for j, cluster in enumerate(clusters):
                corr_df = complete_df[[cluster, annotation]]
                corr_df = corr_df.dropna()
                if corr_method == 'spearman':
                    rho, pval = spearmanr(corr_df[cluster], corr_df[annotation])
                elif corr_method == 'pearson':
                    rho, pval = pearsonr(corr_df[cluster], corr_df[annotation])
                if len(rho.shape) > 0:
                    rho  = rho[0,0]
                    pval = pval[0,0]
                all_data[i,j, 0] = rho
                all_data[i,j, 1] = pval

        all_data_rho = pd.DataFrame(all_data[:,:,0], columns=clusters)
        all_data_rho.index = anno_fields

        all_data_pval = pd.DataFrame(all_data[:,:,1], columns=clusters)
        all_data_pval.index = anno_fields

        # p-value correction.
        shape = all_data_pval.values.shape
        flat_pvalues = all_data_pval.values.flatten()
        reject, pvals_corrected, _, _ = smm.multipletests(pvals=flat_pvalues, method=method)

        # Wrap into a dataframe.
        all_data_pval = pd.DataFrame(pvals_corrected.reshape(shape), columns=clusters)
        all_data_pval.index = anno_fields

        return all_data_rho, all_data_pval
    run_path          = os.path.join(directory, '%s_fold%s' % (groupby.replace('.','p'), fold_number))
    correlations_path = os.path.join(run_path, 'correlations')
    if not os.path.isdir(correlations_path):
        os.makedirs(correlations_path)

    # Grab clusters.
    clusters    = [str(cluster) for cluster in slide_rep_df.columns if cluster not in [purity_field, 'slides', 'samples']]

    # Correlations.
    all_data_rho, all_data_pval = correlations(slide_rep_df[1:][clusters].astype(float), clusters, clusters, corr_method=corr_method, method=method_correction)

    # Mask for p-value threshold.
    mask = (all_data_pval.values > pval_th)
    mask = pd.DataFrame(mask, columns=clusters)
    mask.index = clusters

    all_data_rho.to_csv(os.path.join(correlations_path,  file_name+'_coef.csv'))
    all_data_pval.to_csv(os.path.join(correlations_path, file_name+'_pval.csv'))

    return all_data_rho, all_data_pval, mask

''' ############ Cluster purity with cell annotations - Tiles ############ '''
def ks_test_cluster_purities(cluster_anno_df, fields, groupby, fold_number, directory, file_name, p_th=0.01, critical_values_flag=True, method_correction='fdr_bh'):
    from scipy.stats import ks_2samp
    c_alpha = {0.05:1.36, 0.01:1.63, 0.005:1.73}

    run_path          = os.path.join(directory, '%s_fold%s' % (groupby.replace('.','p'), fold_number))
    correlations_path = os.path.join(run_path, 'correlations')
    if not os.path.isdir(correlations_path):
        os.makedirs(correlations_path)

    cluster_ids = np.unique(cluster_anno_df[groupby])
    critical_coef    = np.zeros((len(cluster_ids), len(fields)))
    critical_ref     = np.zeros((len(cluster_ids), len(fields)))
    p_values         = np.zeros((len(cluster_ids), len(fields)))
    for i, field in enumerate(fields):
        population_counts = cluster_anno_df[field].values
        ss_population    = population_counts.shape[0]

        for cluster_id in cluster_ids:
            cluster_counts = cluster_anno_df[cluster_anno_df[groupby]==cluster_id][field].values
            ss_cluster     = cluster_counts.shape[0]
            d_alpha        = c_alpha[p_th]*np.sqrt((ss_population+ss_cluster)/(ss_population*ss_cluster))

            critical_ref[cluster_id,i] = d_alpha
            # F(x)=G(x) for all x as null hypothesis; the alternative is that they are not identical.
            # Check if there's a deviation.
            critical_value, p_value = ks_2samp(data1=population_counts, data2=cluster_counts, alternative='two-sided')
            if (critical_values_flag and (critical_value > d_alpha)) or ((not critical_values_flag) and (p_value < p_th)):
                for alternative in ['less', 'greater']:
                    critical_value, p_value = ks_2samp(data1=population_counts, data2=cluster_counts, alternative=alternative)
                    if (critical_values_flag and (critical_value > d_alpha)) or ((not critical_values_flag) and (p_value < p_th)):
                        break
            #
            # F(x) >= G(x) for all x as null hypothesis; the alternative is that F(x) < G(x)
            if alternative=='less':
                critical_coef[cluster_id,i] = -critical_value
                p_values[cluster_id,i]      = p_value
            # F(x) <= G(x) for all x as null hypothesis; the alternative is that F(x) > G(x)
            else:
                critical_coef[cluster_id,i] = +critical_value
                p_values[cluster_id,i]      = p_value

    critical_coef = pd.DataFrame(critical_coef, columns=fields, index=cluster_ids.astype(str)).transpose()
    critical_ref  = pd.DataFrame(critical_ref,  columns=fields, index=cluster_ids.astype(str)).transpose()
    p_values      = pd.DataFrame(p_values,      columns=fields, index=cluster_ids.astype(str)).transpose()

    # p-value correction.
    shape = p_values.values.shape
    flat_pvalues = p_values.values.flatten()
    reject, pvals_corrected, _, _ = smm.multipletests(pvals=flat_pvalues, method=method_correction)

    # Wrap into a dataframe.
    p_values = pd.DataFrame(pvals_corrected.reshape(shape), columns=cluster_ids.astype(str), index=fields)

    critical_coef.to_csv(os.path.join(correlations_path,  file_name+'_critical_coef.csv'))
    critical_ref.to_csv(os.path.join(correlations_path,   file_name+'_critical_ref.csv'))
    p_values.to_csv(os.path.join(correlations_path,       file_name+'_pval.csv'))

    if critical_values_flag:
        mask  = (np.abs(critical_coef)<=critical_ref)
    else:
        # Mask for p-value threshold.
        mask = (p_values.values > p_th)
        mask = pd.DataFrame(mask, columns=p_values.columns)
        mask.index = p_values.index

    return critical_coef, critical_ref, p_values, mask

''' ############ Cluster purity with hypergeometric test ############ '''
# Get counts for the different classes.
def get_counts(frame, meta_field, classes_reference):
    # Discrete Distribution PMF
    classes, counts = np.unique(frame[meta_field].values, return_counts=True)
    total_counts = np.zeros((len(classes_reference)), dtype=int)
    for class_, count in zip(classes, counts):
        index = np.argwhere(classes_reference==class_)
        total_counts[index] = count
    sample_size = np.sum(total_counts)
    return total_counts, sample_size

# Hyper-geometric test.
def perform_hypergeometric(k, n, k_cap, n_cap, pvalue_as_strengh, pvalue_min=1e-300):
    expectation = n*(k_cap/n_cap)
    overrep = +1
    if k >= expectation:
        p_value = hypergeom.sf(k-1, n_cap, k_cap, n)
    else:
        p_value = hypergeom.cdf(k, n_cap, k_cap, n)
        overrep = -1
    fold = k/expectation
    if pvalue_as_strengh:
        p_value_adj = p_value
        if p_value==0: p_value_adj += pvalue_min
        fold = overrep*(-np.log(p_value_adj))
    return  p_value, fold, overrep

# Perform purity for all types in df using hyper-geometric.
def cluster_purity_hypergeom(frame, frame_clusters, groupby, meta_field, pval_th=0.01, pvalue_as_strengh=False, method_correction='fdr_bh'):
    from scipy.stats import hypergeom

    cluster_ids = np.unique(frame_clusters[groupby].values.astype(int))
    classes_reference = np.unique(frame[meta_field].values)
    population_counts, population_ss = get_counts(frame, meta_field, classes_reference)

    p_values = np.ones((len(cluster_ids),  len(classes_reference)))
    strength = np.zeros((len(cluster_ids), len(classes_reference)))
    for cluster_id in cluster_ids:
        frame_cluster = frame[frame[groupby].values.astype(int)==cluster_id]
        cluster_counts, cluster_ss = get_counts(frame_cluster, meta_field, classes_reference)
        for i, class_ in enumerate(classes_reference):
            p_value = 1
            fold    = 1
            if cluster_ss != 0:
                p_value, fold, rep = perform_hypergeometric(cluster_counts[i], cluster_ss, population_counts[i], population_ss, pvalue_as_strengh)
            p_values[cluster_id, i] = p_value
            strength[cluster_id, i] = fold

    p_values = pd.DataFrame(p_values, columns=classes_reference, index=cluster_ids)
    strength = pd.DataFrame(strength, columns=classes_reference, index=cluster_ids)

    # p-value correction.
    shape = p_values.values.shape
    flat_pvalues = p_values.values.flatten()
    reject, pvals_corrected, _, _ = smm.multipletests(pvals=flat_pvalues, method=method_correction)

    # Wrap into a dataframe.
    p_values = pd.DataFrame(pvals_corrected.reshape(shape), columns=classes_reference, index=cluster_ids)

    mask = p_values > pval_th
    return p_values, strength, mask

''' ############ Cluster spatial interaction correlations ############ '''
# Run spatial correlations between clusters.
def spatial_correlations_all(frame, leiden_clusters, groupby, normalize='cluster', include_background=True, own_corr=True):
    # Type of normalization.
    if normalize not in ['cluster', 'all']:
        print('Normalization option not contemplated. Options: \'cluster\' or \'all\'')
        return

    # Dimensionality of connection vector.
    dimensions = len(leiden_clusters)
    if include_background:
        dimensions += 1

    heatmap = list()
    for cluster_id in leiden_clusters:
        all_conn = [0]*dimensions
        if frame[frame[groupby]==cluster_id].shape[0]!=0:
            cluster_conn, counts_conn = np.unique(frame[frame[groupby]==cluster_id][['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']], return_counts=True)
            # Handle background interactions.
            if np.argwhere(cluster_conn.astype(str)=='nan').shape[0] != 0:
                idx = np.argwhere(cluster_conn.astype(str)=='nan')[0,0]
                if include_background:
                    all_conn[-1] = counts_conn[idx]
                cluster_conn = np.delete(cluster_conn, idx, axis=0)
                counts_conn  = np.delete(counts_conn, idx, axis=0)

            # Cluster to cluster interactions.
            for cluster_id_conn in leiden_clusters:
                if cluster_id == cluster_id_conn:
                    idx = np.argwhere(cluster_conn==cluster_id_conn)[0,0]
                    conn = counts_conn[idx]
                elif np.argwhere(cluster_conn==cluster_id_conn)[:,0].shape[0] == 0:
                    conn = 0
                else:
                    idx = np.argwhere(cluster_conn==cluster_id_conn)[0,0]
                    conn = counts_conn[idx]
                all_conn[cluster_id_conn] = conn

            if normalize == 'cluster':
                all_conn = (np.array(all_conn)/np.sum(all_conn)).tolist()

        heatmap.append([cluster_id]+all_conn)

    heatmap = np.array(heatmap, dtype=float)
    if normalize=='all':
        total_interactions = np.triu(heatmap[:, 1:]).sum()
        heatmap[:, 1:] /= float(total_interactions)

    columns = ['Cluster ID'] + leiden_clusters.tolist()
    if include_background:
        columns.append('Background')
    heatmap = pd.DataFrame(heatmap, columns=columns)

    return heatmap