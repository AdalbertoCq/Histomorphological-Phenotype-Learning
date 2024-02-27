# Python libs.
from sklearn.metrics        import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from scipy.spatial          import distance_matrix
import pandas as pd
import copy

# Own libs.
from models.evaluation.folds import load_existing_split
from models.clustering.data_processing import *


''' ############ Summary Figure ############ '''
def summarize_cluster_evalutation(data_csv, meta_folder, metrics):
    from kneed import KneeLocator

    cluster_evaluations = pd.read_csv(data_csv)

    # Combine a given metric with their ability to generalize across institutions.
    metrics_add   = list()
    metrics_add_2 = list()
    for metric_show, curve, direction in metrics:
        if curve == 'convex':
            values_1 = (1 - cluster_evaluations[metric_show]/cluster_evaluations[metric_show].max())
        else:
            values_1 = cluster_evaluations[metric_show]
        cluster_evaluations['%s_institutions' % metric_show] = values_1*cluster_evaluations['insitution_precense']
        cluster_evaluations['%s_institutions_w' % metric_show] = values_1*cluster_evaluations['insitution_precense_size_weighted']
        metrics_add.append(('%s_institutions' % metric_show, 'concave', 'increasing'))
        metrics_add_2.append(('%s_institutions_w' % metric_show, 'concave', 'increasing'))

    metric_display = metrics + metrics_add + metrics_add_2

    # Find knee from mean value of resolution.
    performance_mean = cluster_evaluations.groupby('resolution').mean()

    # Plot metrics.
    knee_values = list()
    fig = plt.figure(figsize=(30,16))
    for i, values in enumerate(metric_display):
        metric_show, curve, direction = values
        ax  = fig.add_subplot(4, 3, i+1)
        sns.pointplot(data=cluster_evaluations, y=metric_show, x='resolution', ax=ax,  label='metric', errorbar='se')
        kn = KneeLocator(performance_mean.index, performance_mean[metric_show], curve=curve, direction=direction)
        value_index = np.where(performance_mean.index==kn.knee)[0]
        if len(value_index)!=0:
            ax.axvline(value_index, linestyle='--')
            knee_values.append(value_index)
        
        if i==0:
            ax.set_title(meta_folder, fontsize=24)
    
    ax  = fig.add_subplot(4, 3, i+2)
    sns.pointplot(data=cluster_evaluations, y='num_clusters', x='resolution', ax=ax, errorbar='se')
    for value_index in knee_values:
        ax.axvline(value_index, linestyle='--')
    ax  = fig.add_subplot(4, 3, i+3)
    sns.pointplot(data=cluster_evaluations, y='insitution_precense', x='resolution', ax=ax, color='green', errorbar='se')
    ax.set_ylim([0.0,1.0])
    # sns.pointplot(data=cluster_evaluations, y='patient_precense',    x='resolution', ax=ax2, color='orange', errorbar='se')

    ax  = fig.add_subplot(4, 3, i+4)
    sns.pointplot(data=cluster_evaluations, y='insitution_precense_size_weighted', x='resolution', ax=ax, color='green', errorbar=('ci', 95))
    ax.set_ylim([0.0,1.0])
    
    plt.savefig(data_csv.replace('.csv', '.jpg'))
    plt.close()

def summarize_institution_patient_distribution(data_res, main_cluster_path):
    resolutions = list(data_res.keys())
    folds       = list(data_res[resolutions[0]].keys())

    for fold in folds:
        fig   = plt.figure(figsize=(30*3,7*len(resolutions)))
        i = 1
        for resolution in resolutions:
            tiles_df      = data_res[resolution][fold]['adata_train']
            data_hpc_inst = data_res[resolution][fold]['institution']
            hpc_pat_th    = data_res[resolution][fold]['patient']
            
            ax    = fig.add_subplot(len(resolutions), 3, i)
            plot_cluster_size_ax(tiles_df, resolution, ax)
            i+=1
            ax    = fig.add_subplot(len(resolutions), 3, i)
            plot_institution_distribution_ax(data_hpc_inst, field='Percentage of Institutions in HPC', title='Percentage of total institutions present in the HPC\nResolution %s' % resolution, 
                                             ax=ax, fontsize_labels=22, fontsize_legend=20)
            i+=1
            ax    = fig.add_subplot(len(resolutions), 3, i)
            plot_institution_distribution_ax(hpc_pat_th, field='samples', title='Percentage of total patient present in the HPC\nResolution %s' % resolution, 
                                             ax=ax, fontsize_labels=22, fontsize_legend=20)
            i+=1
        plt.tight_layout()
        plt.savefig(os.path.join(main_cluster_path, 'cluster_evalutation_inst_pat_distribution_fold%s.jpg' % fold))


def plot_institution_distribution_ax(data_hpc_inst, field, title, ax, fontsize_labels=22, fontsize_legend=20, show_max_min=False):
    def colors_from_values(values, palette_name, normalize=False):
        # normalize the values to range [0, 1]
        if normalize:
            normalized = (values - min(values)) / (max(values) - min(values))
        else:
            normalized = values
        # convert to indices
        indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
        # use the indices to get the colors
        palette = sns.color_palette(palette_name, int(1.5*len(values)))
        return np.array(palette).take(indices, axis=0)

    y = data_hpc_inst[field].values
    mean_y = np.mean(y)
    sns.barplot(data=data_hpc_inst, x='HPC', y=field, palette=colors_from_values(y, "Greens_d"), ax=ax)

    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim([0.0,1.0])
    yticks = (np.array(range(0,11,1))/10).tolist()
    ax.set_yticks(yticks, yticks)

    ax.set_title(title,  fontsize=fontsize_labels*1.3, fontweight='bold')
    ax.set_xlabel('\nHistomorphological Phenotype Cluster (HPC)', fontsize=fontsize_labels,     fontweight='bold')
    ax.set_ylabel(' ', fontsize=fontsize_labels, fontweight='bold')
    if show_max_min:
        max_val = np.max(data_hpc_inst[field].values)
        min_val = np.min(data_hpc_inst[field].values)
        ax.axhline(max_val, linestyle='--')
        ax.axhline(min_val, linestyle='--')
    ax.axhline(0.50, linestyle='--', color='black')
    ax.axhline(0.25, linestyle='--', color='black')
    ax.axhline(mean_y, linestyle='--', color='red')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_labels)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_labels)
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)

def plot_cluster_size_ax(tiles_df, resolution, ax, fontsize_labels=22):
    groupby = 'leiden_%s' % resolution
    tiles_df[groupby] = tiles_df[groupby].astype(int)
    tiles_df = tiles_df.sort_values(by=groupby)
    leiden_clusters = np.unique(tiles_df[groupby].values)
    
    min_val = tiles_df[groupby].min()
    max_val = tiles_df[groupby].max()
    val_width = max_val - min_val
    n_bins = len(leiden_clusters)
    bin_width = val_width/n_bins

    ticks      = np.arange(min_val-bin_width/2, max_val+bin_width/2, bin_width)
    ticklabels = ['']+leiden_clusters.tolist()
    if len(ticks) != len(ticklabels):
        ticklabels.append('')

    sns.histplot(data=tiles_df, stat='percent', x=groupby, bins=n_bins, binrange=(min_val,max_val), fill=False, common_norm=True, ax=ax, shrink=0.8, linewidth=3)
    ax.set_ylabel('Relative size of HPC\nPercentage of total tiles', fontweight='bold', fontsize=fontsize_labels)
    ax.set_xlabel('Histomorphological Phenotype Cluster (HPC)'     , fontweight='bold', fontsize=fontsize_labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_labels)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize_labels)
        tick.label1.set_fontweight('bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)

    ax.tick_params(axis='x', rotation=90)

def display_metric_per_point(metric_values, leiden_assignations, metric_name, fig_path):

    import matplotlib.cm as cm

    fig = plt.figure(figsize=(5,15))
    ax1  = fig.add_subplot(1, 1, 1)
    leiden_assignations = leiden_assignations.astype(int)
    n_clusters = len(np.unique(leiden_assignations))

    metric_values_avg = np.mean(metric_values)

    y_lower = 1
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_metric_values_values = metric_values[leiden_assignations == i]

        ith_cluster_metric_values_values.sort()

        size_cluster_i = ith_cluster_metric_values_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_metric_values_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontweight='bold')

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title('%s Score per Data Point' % metric_name, fontweight='bold')
    ax1.set_xlabel('%s Coefficient Values' % metric_name, fontweight='bold')
    ax1.set_ylabel('HPC', fontweight='bold')

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=metric_values_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    plt.savefig(fig_path)
    plt.close()

''' ############ Metrics ############ '''

def run_silhoutte_NNgraph(adata_train, groupby, h5ad_path):

    if os.path.isfile(h5ad_path.replace('.h5ad', '_silhoutte_nngraph.npz')):
        content = np.load(h5ad_path.replace('.h5ad', '_silhoutte_nngraph.npz'))
        silhoutte        = content['silhoutte']

    else:
        # Pre-computer distance matrix from NN graph.
        distance_matrix_csr   = copy.deepcopy(adata_train.obsp['nn_leiden_distances'])
        representation_matrix = copy.deepcopy(adata_train.X)

        # Leiden assignations. 
        leiden_clusters  = adata_train.obs[groupby].unique()
        leiden_assignations = adata_train.obs[groupby].values

        # Compute mean distance from each point to others per cluster. 
        mean_distances = np.zeros((distance_matrix_csr.shape[0],len(leiden_clusters), 2))
        rows, columns = distance_matrix_csr.nonzero()
        for data_point_index, neighbor_index in zip(rows, columns):
            neighbor_leiden   = leiden_assignations[neighbor_index]
            mean_distances[data_point_index,int(neighbor_leiden),0] += distance_matrix_csr[data_point_index, neighbor_index]
            mean_distances[data_point_index,int(neighbor_leiden),1] += 1

        # Normalize per connection.
        mean_distances_f = np.zeros((distance_matrix_csr.shape[0],len(leiden_clusters)))
        mean_distances_f = mean_distances[:,:,0]/mean_distances[:,:,1]

        # Silhoutte score per point.
        silhoutte = list()
        for data_point_index in range(mean_distances_f.shape[0]):
            if np.sum(np.isnan(mean_distances_f[data_point_index, :])*1)==len(leiden_clusters)-1:
                score = 1
            else:
                hpc_order = np.argsort(mean_distances_f[data_point_index, :])
                if int(hpc_order[0]) == int(leiden_assignations[data_point_index]):
                    a_i = mean_distances_f[data_point_index, hpc_order[0]]
                    b_i = mean_distances_f[data_point_index, hpc_order[1]]
                else:
                    a_i = mean_distances_f[data_point_index, hpc_order[int(leiden_assignations[data_point_index])]]
                    b_i = mean_distances_f[data_point_index, hpc_order[0]]
                if a_i < b_i:
                    score = 1 - (a_i/b_i)
                elif a_i > b_i:
                    score = (b_i/a_i) - 1
            silhoutte.append(score)
        silhoutte = np.array(silhoutte)

        # Save matrices.
        np.savez(h5ad_path.replace('.h5ad', '_silhoutte_nngraph.npz'), mean_distances=mean_distances, mean_distances_f=mean_distances_f, silhoutte=silhoutte)

    return np.mean(silhoutte)

def run_simplified_silhoutte(adata_train, groupby, h5ad_path, fold, cluster_evaluation_path):
    leiden_assignations = adata_train.obs[groupby].values
    if os.path.isfile(h5ad_path.replace('.h5ad', '_simplified_silhoutte.npz')):
        content = np.load(h5ad_path.replace('.h5ad', '_simplified_silhoutte.npz'))
        leiden_centroids = content['leiden_centroids']
        mean_distances_f = content['mean_distances_f']
        silhoutte        = content['silhoutte']

    else:
        # Pre-computer distance matrix from NN graph.
        distance_matrix_csr   = copy.deepcopy(adata_train.obsp['nn_leiden_distances'])
        representation_matrix = copy.deepcopy(adata_train.X)

        # Leiden assignations
        leiden_clusters  = adata_train.obs[groupby].unique()
        leiden_assignations = adata_train.obs[groupby].values
        
        # Leiden Centroids
        leiden_centroids = np.zeros((len(leiden_clusters), representation_matrix.shape[1]))
        for hpc in leiden_clusters:
            hpc_indexes = np.where(leiden_assignations==hpc)[0]
            centroid = np.mean(representation_matrix[hpc_indexes,:], axis=0)
            leiden_centroids[int(hpc), :] = centroid

        # Compute distance from point to centroids.
        mean_distances_f = np.zeros((distance_matrix_csr.shape[0],len(leiden_clusters)))
        for data_point_index in range(distance_matrix_csr.shape[0]):
            for hpc in leiden_clusters.astype(int):
                mean_distances_f[data_point_index, hpc] = euclidean(representation_matrix[data_point_index,:], leiden_centroids[hpc, :])

        # Silhoutte score per point.
        silhoutte = list()
        for data_point_index in range(mean_distances_f.shape[0]):
            data_point_leiden = leiden_assignations[data_point_index]
            hpc_order = np.argsort(mean_distances_f[data_point_index, :])
            if int(hpc_order[0]) == int(data_point_leiden):
                a_i = mean_distances_f[data_point_index, hpc_order[0]]
                b_i = mean_distances_f[data_point_index, hpc_order[1]]
            else:
                a_i = mean_distances_f[data_point_index, hpc_order[int(data_point_leiden)]]
                b_i = mean_distances_f[data_point_index, hpc_order[0]]
            if a_i < b_i:
                score = 1 - (a_i/b_i)
            elif a_i > b_i:
                score = (b_i/a_i) - 1
            elif a_i == b_i:
                score = 0
            silhoutte.append(score)
        silhoutte = np.array(silhoutte)

        # Save matrices.
        np.savez(h5ad_path.replace('.h5ad', '_simplified_silhoutte.npz'), leiden_centroids=leiden_centroids, mean_distances_f=mean_distances_f, silhoutte=silhoutte)
    
    fig_path = os.path.join(cluster_evaluation_path, 'silhoutte_%s_fold%s.jpg' % (groupby, fold))
    display_metric_per_point(silhoutte, leiden_assignations, metric_name='Silhoutte', fig_path=fig_path)

    return np.mean(silhoutte)

def disruption_metric(adata_train, groupby, h5ad_path, fold, cluster_evaluation_path):
    leiden_assignations = adata_train.obs[groupby].values
    if os.path.isfile(h5ad_path.replace('.h5ad', '_disruption.npz')):
        content = np.load(h5ad_path.replace('.h5ad', '_disruption.npz'))
        leiden_centroids = content['leiden_centroids']
        distances_centroid = content['distances_centroid']

    else:
        # Pre-computer distance matrix from NN graph.
        distance_matrix_csr   = copy.deepcopy(adata_train.obsp['nn_leiden_distances'])
        representation_matrix = copy.deepcopy(adata_train.X)

        # Leiden assignations
        leiden_clusters  = adata_train.obs[groupby].unique()
        leiden_assignations = adata_train.obs[groupby].values
        
        # Leiden Centroids
        leiden_centroids = np.zeros((len(leiden_clusters), representation_matrix.shape[1]))
        for hpc in leiden_clusters:
            hpc_indexes = np.where(leiden_assignations==hpc)[0]
            centroid = np.mean(representation_matrix[hpc_indexes,:], axis=0)
            leiden_centroids[int(hpc), :] = centroid

        # Compute distance from point to centroids.
        distances_centroid = np.zeros((distance_matrix_csr.shape[0]))
        for data_point_index in range(distance_matrix_csr.shape[0]):
            data_point_leiden = leiden_assignations[data_point_index]
            distances_centroid[data_point_index] = euclidean(representation_matrix[data_point_index,:], leiden_centroids[int(data_point_leiden), :])

        np.savez(h5ad_path.replace('.h5ad', '_disruption.npz'), leiden_centroids=leiden_centroids, distances_centroid=distances_centroid)

    fig_path = os.path.join(cluster_evaluation_path, 'disruption_%s_fold%s.jpg' % (groupby, fold))
    display_metric_per_point(distances_centroid, leiden_assignations, metric_name='Disruption', fig_path=fig_path)

    disruption = np.mean(distances_centroid)
    return disruption

def variance_ratio_criterion(adata_train, groupby, h5ad_path):
    representation_matrix = copy.deepcopy(adata_train.X)
    leiden_assignations   = adata_train.obs[groupby].values
    vrc = calinski_harabasz_score(representation_matrix, leiden_assignations)
    return vrc

def davies_bouldin_index(adata_train, groupby, h5ad_path):
    representation_matrix = copy.deepcopy(adata_train.X)
    leiden_assignations   = adata_train.obs[groupby].values
    dbi = davies_bouldin_score(representation_matrix, leiden_assignations)
    return dbi

def patient_precense(adata_train, groupby, matching_field='samples', threshold=0.005):
    tiles_df = adata_train.obs
    tiles_df['samples']  = tiles_df['slides'].apply(lambda x: '-'.join(x.split('-')[:3]))
    tiles_df['TSS Code'] = tiles_df['samples'].apply(lambda x: x.split('-')[1]).values.astype(str)

    # Include counts per samples and HPC
    for name, field in [('sample',matching_field), ('hpc', groupby)]:
        counts_per_field = tiles_df.groupby(field).count()
        counts_per_field = counts_per_field.reset_index()
        counts_per_field = counts_per_field.rename(columns={'tiles':'nt_per_%s'%name})
        tiles_df = tiles_df.merge(counts_per_field[[field, 'nt_per_%s'%name]], on=field)

    # Normalized values of percentage of total tiles in HPC
    tiles_df.insert(loc=len(tiles_df.columns), column='nt_per_hpc_norm', value=tiles_df['nt_per_hpc'].values/tiles_df.shape[0])

    # Normalize contribution of HPC in patient.
    hpc_pat = tiles_df[[matching_field, groupby, 'tiles']].groupby([matching_field, groupby]).count()
    hpc_pat = hpc_pat.reset_index()
    hpc_pat = hpc_pat.rename(columns={'tiles':'nt_per_sample_hpc'})
    hpc_pat = hpc_pat.merge(tiles_df[[matching_field, 'nt_per_sample']].drop_duplicates(), on=matching_field)
    hpc_pat = hpc_pat.drop_duplicates()
    hpc_pat['nt_per_sample_hpc_norm'] = np.divide(hpc_pat['nt_per_sample_hpc'].values.astype(float), hpc_pat['nt_per_sample'].values.astype(float))

    hpc_pat_th = hpc_pat[hpc_pat.nt_per_sample_hpc_norm >= threshold]
    hpc_pat_th = hpc_pat_th.groupby([groupby]).count()['samples']/len(np.unique(hpc_pat[matching_field]))
    hpc_pat_th = hpc_pat_th.reset_index()
    hpc_pat_th = hpc_pat_th.rename(columns={groupby:'HPC'})

    return np.mean(hpc_pat_th['samples']), hpc_pat_th

def institution_precense(adata_train, groupby, threshold=0.005):
    # Threshold out institution with not enough contribution at least 0.5% (default).
    tiles_df = adata_train.obs
    tiles_df['samples']  = tiles_df['slides'].apply(lambda x: '-'.join(x.split('-')[:3]))
    tiles_df['TSS Code'] = tiles_df['samples'].apply(lambda x: x.split('-')[1]).values.astype(str)
    tss_hpc = tiles_df[['TSS Code', 'tiles']].groupby('TSS Code').count()
    tss_hpc = tss_hpc.reset_index()
    tss_hpc = tss_hpc.rename(columns={'tiles':'total_tiles'})

    hpc_sizes = tiles_df.groupby(groupby).count()['tiles']/tiles_df.shape[0]

    data_hpc_inst = list()
    weight_size   = list()
    for hpc in np.unique(tiles_df[groupby]):
        hpc_df    = tiles_df[tiles_df[groupby]==hpc]
        hpc_df    = hpc_df.groupby('TSS Code').count()
        hpc_df    = hpc_df.reset_index()[['TSS Code', 'tiles']]
        hpc_df    = hpc_df.merge(tss_hpc, on='TSS Code', how='inner')
        hpc_df.insert(len(hpc_df.columns), 'tiles_norm', np.divide(hpc_df['tiles'].values,hpc_df['total_tiles'].values))
        hpc_df    = hpc_df[hpc_df['tiles_norm']>=threshold]
        rel_size  = hpc_sizes.loc[hpc]/hpc_sizes.max()
        weight_size.append(rel_size)
        data_hpc_inst.append((hpc, hpc_df.shape[0]/tss_hpc.shape[0]))
    data_hpc_inst = pd.DataFrame(data_hpc_inst, columns=['HPC', 'Percentage of Institutions in HPC'])
    data_hpc_inst['HPC'] = data_hpc_inst['HPC'].astype(int)
    data_hpc_inst = data_hpc_inst.sort_values(by='HPC')

    inst_avg          = np.mean(data_hpc_inst['Percentage of Institutions in HPC'])
    inst_weighted_avg = np.average(data_hpc_inst['Percentage of Institutions in HPC'], weights=weight_size)

    return inst_avg, inst_weighted_avg, data_hpc_inst

''' ############ Main method comparison ############ '''

def evaluate_cluster_configurations(h5_complete_path, meta_folder, folds_pickle, resolutions, threshold_inst=0.005, include_nngraph=False):

    # Get folds from existing split.
    folds = load_existing_split(folds_pickle)

    # Path to save files.
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    main_cluster_path = os.path.join(main_cluster_path, 'cluster_evaluations')
    if not os.path.isdir(main_cluster_path):
        os.makedirs(main_cluster_path)
    if include_nngraph:
        data_csv = os.path.join(main_cluster_path, 'cluster_evaluation_metrics_nngraph.csv')
    else:
        data_csv = os.path.join(main_cluster_path, 'cluster_evaluation_metrics.csv')

    print()
    data = list()
    data_res = dict()
    remove_res = list()
    for resolution in resolutions:
        print('Leiden %s' % resolution)
        groupby = 'leiden_%s' % resolution
        data_res[resolution] = dict()
        for i, fold in enumerate(folds):
            print('\tFold %s' % i)
            try:
                adata_train, h5ad_path = read_h5ad_reference(h5_complete_path, meta_folder, groupby, i)
            except:
                print('\t\tIssue reading resolution/fold: %s/%s' % (resolution, i))
                remove_res.append(resolution)
                continue
            data_res[resolution][i] = dict()
            
            # Leiden clusters.
            leiden_clusters = adata_train.obs[groupby].unique()
            fields = ['resolution', 'fold', 'num_clusters']
            values = [resolution, i, len(leiden_clusters)]
            
            # Metrics.
            print('\t\tDisruption Metric')
            disruption_avg = disruption_metric(adata_train, groupby, h5ad_path, i, main_cluster_path)
            fields.append('disruption')
            values.append(disruption_avg)

            print('\t\tSimplified Silhoutte Score')
            silhoutte = run_simplified_silhoutte(adata_train, groupby, h5ad_path, i, main_cluster_path)
            fields.append('silhoutte')
            values.append(silhoutte)

            if include_nngraph:
                print('\t\tSilhoutte score NN Graph')
                silhoutte_graph = run_silhoutte_NNgraph(adata_train, groupby, h5ad_path)
                fields.append('silhoutte_nngraph')
                values.append(silhoutte_graph)

            print('\t\tVariance Ratio Criterion')
            vrc = variance_ratio_criterion(adata_train, groupby, h5ad_path)    
            fields.append('variance_ratio_criterion')
            values.append(vrc)    

            print('\t\tDavies-Bouldin Index')
            dbi = davies_bouldin_index(adata_train, groupby, h5ad_path)
            fields.append('davies_bouldin_index')
            values.append(dbi)

            print('\t\tInstitution Precense')
            inst_avg, inst_weighted_avg, data_hpc_inst = institution_precense(adata_train, groupby, threshold=threshold_inst)
            fields.append('insitution_precense')
            fields.append('insitution_precense_size_weighted')
            values.append(inst_avg)
            values.append(inst_weighted_avg)
            data_res[resolution][i]['institution'] = data_hpc_inst

            print('\t\tPatient Precense')
            pat_avg, hpc_pat_th = patient_precense(adata_train, groupby, threshold=0.005)
            fields.append('patient_precense')
            values.append(pat_avg)
            data_res[resolution][i]['patient'] = hpc_pat_th
            
            # Keep track of 
            data_res[resolution][i]['adata_train'] = adata_train.obs

            data.append(values)

            data_df = pd.DataFrame(data, columns=fields)
            data_df.to_csv(data_csv, index=False)

        print()
    
    # remove resolutions with no data.
    remove_res = list(set(remove_res))
    for res in remove_res:
        data_res.pop(res, None)

    # Figure summary.
    metrics = [('disruption', 'convex', 'decreasing'), ('silhoutte', 'concave', 'increasing'), ('davies_bouldin_index', 'convex', 'decreasing')]
    summarize_cluster_evalutation(data_csv, meta_folder, metrics)
    
    # Figure summary for institution and patient distribution per resolution.
    summarize_institution_patient_distribution(data_res, main_cluster_path)


