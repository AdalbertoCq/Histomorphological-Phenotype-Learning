from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import csv
import os
import gc

# Own libs
from models.clustering.data_processing import *
from models.visualization.attention_maps import *
from models.clustering.leiden_representations import include_tile_connections_frame

''' ######  Cluster Summary CircularPlot ######### '''
def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(x=angle, y=value + padding, s=label, ha=alignment, va="center", rotation=rotation, rotation_mode="anchor") 


def circular_barplot(ax, VALUES, LABELS, GROUP, GROUPS_NAME, GROUPS_SIZE, figure_name, ref_mark=None):

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)
    OFFSET = np.pi / 2

    # The index contains non-empty bards
    offset = 0
    IDXS = []

    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    ax.set_title(figure_name, fontweight='bold', fontsize=18)
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-100, 100)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if ref_mark is not None:
        for value in ref_mark:
            theta = np.arange(0, 2*np.pi, 0.01)
            r     = np.ones(theta.shape)*value
            ax.plot(theta, r, c='black', linestyle='--')

    # Use different colors for each group!
    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    # And finally add the bars. 
    # Note again the `ANGLES[IDXS]` to drop some angles that leave the space between bars.
    ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    offset = 0 
    for group, size in zip(GROUPS_NAME, GROUPS_SIZE):
        # Add line below bars
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#333333")

        # Add text to indicate group
        ax.text(
            np.mean(x1), -20, group, color="#333333", fontsize=14, 
            fontweight="bold", ha="center", va="center"
        )

        # Add reference lines at 20, 40, 60, and 80
        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)

        offset += size + PAD
        

def cluster_circular(frame_clusters, groupby, meta_field, file_path, rescale=False):
    df_sorted = (frame_clusters.groupby(['Subtype']).apply(lambda x: x.sort_values(['Subtype Counts'], ascending = True)).reset_index(drop=True))

    # Same layout as above
    fig, ax = plt.subplots(figsize=(30, 12), nrows=1, ncols=3, subplot_kw={"projection": "polar"})      

    GROUPS_NAME, GROUPS_SIZE = np.unique(df_sorted['Subtype'], return_counts=True)
    
    if rescale:
        ref_mark = None
        VALUES   = (df_sorted['Subtype Purity(%)'].values-50)*2
    else:
        ref_mark = [50]
        VALUES   = df_sorted['Subtype Purity(%)'].values
    LABELS = df_sorted[groupby].values

    GROUP  = df_sorted['Subtype'].values

    circular_barplot(ax[0], VALUES, LABELS, GROUP, GROUPS_NAME, GROUPS_SIZE, figure_name='Cluster # %s' % meta_field, ref_mark=ref_mark)
    
    if rescale:
        ref_mark = None
        LABELS = ((df_sorted['Subtype Purity(%)'].values-50)*2).astype(int)
    else:
        ref_mark = [50]
        LABELS   = df_sorted['Subtype Purity(%)'].values.astype(int)
    circular_barplot(ax[1], VALUES, LABELS, GROUP, GROUPS_NAME, GROUPS_SIZE, figure_name='Cluster %s Purity' % meta_field, ref_mark=ref_mark)

    LABELS = df_sorted['Subtype Counts'].astype(int).values
    VALUES = df_sorted['Subtype Counts'].values/np.max(df_sorted['Subtype Counts'].values) * 100

    circular_barplot(ax[2], VALUES, LABELS, GROUP, GROUPS_NAME, GROUPS_SIZE, figure_name='Cluster %s Tile Counts' % meta_field)

    plt.tight_layout()
    plt.savefig(file_path.replace('.csv', '_circularplot.jpg'))
    plt.close(fig)


''' ###### Visualize Cluster Samples ######### '''
def create_annotation_file(leiden_path, frame, groupby, value_cluster_ids=None):
    rel_cluster_ids = []
    [rel_cluster_ids.extend(a) for a in value_cluster_ids.values()]

    cluster_ids = np.unique(frame[groupby].values.astype(int))
    all_data_csv = []
    for cluster_id in cluster_ids:
        if cluster_id in rel_cluster_ids:
            if cluster_id in value_cluster_ids[1]:
                all_data_csv.append((cluster_id, 'Favors', '', '', '', ''))
            else:
                all_data_csv.append((cluster_id, 'Against', '', '', '', ''))
        else:
            all_data_csv.append((cluster_id, 'N/A', '', '', '', ''))
    all_data_csv = pd.DataFrame(all_data_csv, columns=['Cluster', 'Event Indicator', 'LUSC/LUAD Dominant', 'Histological Subtypes', 'Other features', 'Comments'])
    all_data_csv = all_data_csv.sort_values(by='Event Indicator', ascending=True)
    all_data_csv.to_csv(os.path.join(leiden_path, 'cluster_annotations.csv'), index=False)


def plot_cluster_images(groupby, meta_folder, data, fold, h5_complete_path, dpi, value_cluster_ids=None, extensive=False):
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    adatas_path       = os.path.join(main_cluster_path, 'adatas')
    leiden_path       = os.path.join(main_cluster_path, '%s_fold%s' % (groupby.replace('.', 'p'), fold))
    if not os.path.isdir(leiden_path):
        os.makedirs(leiden_path)
        
    # Data Class with all h5, these contain the images.
    data_dicts = dict()
    data_dicts['train'] = data.training.images
    data_dicts['valid'] = None
    if data.validation is not None:
        data_dicts['valid'] = data.validation.images
    data_dicts['test'] = None
    if data.test is not None:
        data_dicts['test'] = data.test.images

    # Base name for each set representations.
    adata_name     = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold)

    # Train set.
    train_fold_csv = os.path.join(adatas_path, '%s_train.csv' % adata_name)
    if not os.path.isfile(train_fold_csv):
        train_fold_csv = os.path.join(adatas_path, '%s.csv' % adata_name)
    frame_train    = pd.read_csv(train_fold_csv)
    # Annotations csv.
    create_annotation_file(leiden_path, frame_train, groupby, value_cluster_ids)
    cluster_set_images(frame_train, data_dicts, groupby, 'train', leiden_path, dpi)

    # Test set.
    if extensive:
        test_fold_csv = os.path.join(adatas_path, '%s_test.csv' % adata_name)
        frame_test    = pd.read_csv(test_fold_csv)
        cluster_set_images(frame_test, data_dicts, groupby, 'test', leiden_path, dpi)


def cluster_set_images(frame, data_dicts, groupby, set_, leiden_path, dpi):
    images_path    = os.path.join(leiden_path, 'images')
    backtrack_path = os.path.join(leiden_path, 'backtrack')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)
        os.makedirs(backtrack_path)

    for cluster_id in pd.unique(frame[groupby].values):
        indexes       = frame[frame[groupby]==cluster_id]['indexes'].values.tolist()
        original_sets = frame[frame[groupby]==cluster_id]['original_set'].values.tolist()
        combined      = list(zip(indexes, original_sets))
        random.shuffle(combined)

        csv_information = list()
        images_cluster = list()
        i = 0
        for index, original_set in combined:
            images_cluster.append(data_dicts[original_set][int(index)]/255.)
            csv_information.append(frame[(frame.indexes==index)&(frame.original_set==original_set)].to_dict('index'))
            i += 1
            if i==100:
                break

        sns.set_theme(style='white')
        fig = plt.figure(figsize=(40, 8))
        fig.suptitle('Cluster %s' % (cluster_id), fontsize=18, fontweight='bold')
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 20), axes_pad=0.1,)

        for ax, im in zip(grid, images_cluster):
            ax.imshow(im)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticks([])

        plt.savefig(os.path.join(images_path, 'cluster_%s_%s.jpg' % (cluster_id, set_)), dpi=dpi)
        plt.close(fig)
        sns.set_theme(style='darkgrid')

        # Tracking file for selected images.
        with open(os.path.join(backtrack_path, 'set_%s_%s.csv' % (cluster_id, set_)), 'w') as content:
            w = csv.DictWriter(content, frame.columns.to_list())
            w.writeheader()
            for element in csv_information:
                for index in element:
                    w.writerow(element[index])


''' ###### Visualize WSI Cluster Samples ######### '''
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def get_wsi_arrays(frame, groupby, slide, img_size, downsample, img_dicts, colors, pad_pixels=50, legend_margin=1000):
    slide_indices  = frame[frame.slides==slide].indexes.values.tolist()
    slide_tiles    = frame[frame.slides==slide].tiles.values.tolist()
    slide_sets     = frame[frame.slides==slide].original_set.values.tolist()
    slide_clusters = frame[frame.slides==slide][groupby].values.tolist()

    # Get size of the WSI.
    y,x = get_x_y(slide_tiles[0])
    x_min = x
    x_max = x
    y_min = y
    y_max = y
    for i in slide_tiles:
        y_i, x_i  = get_x_y(i)
        x_min = min(x_min, x_i)
        y_min = min(y_min, y_i)
        x_max = max(x_max, x_i)
        y_max = max(y_max, y_i)
    x_max += 1
    y_max += 1

    wsi_x = int(x_max*img_size//downsample)
    wsi_y = int(y_max*img_size//downsample)

    # Original 5x.
    wsi   = np.ones((wsi_x, wsi_y, 3), dtype=np.uint8)*255
    wsi_c = np.ones((wsi_x, wsi_y, 3), dtype=np.uint8)*255
    print('\t\tWhole Slide Image Resolution %s: (%s, %s)' % (slide, wsi_x, wsi_y))

    for index, tile, original_set, cluster in zip(slide_indices, slide_tiles, slide_sets, slide_clusters):
        y_i, x_i  = get_x_y(tile)
        x_i *= img_size//downsample
        y_i *= img_size//downsample
        tile_img = img_dicts[original_set][int(index)]
        tile_img = np.array(resize(tile_img, (tile_img.shape[0]//downsample, tile_img.shape[1]//downsample), anti_aliasing=True), dtype=float)
        tile_img = (tile_img*255).astype(np.uint8)

        color = colors[int(cluster)]
        mask  = np.ones((img_size//downsample,img_size//downsample))

        wsi[x_i:x_i+(img_size//downsample), y_i:y_i+(img_size//downsample), :]   = tile_img
        wsi_c[x_i:x_i+(img_size//downsample), y_i:y_i+(img_size//downsample), :] = apply_mask(tile_img, mask, color, alpha=0.5)

    wsi_padded = np.pad(wsi[:, :, 0], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')
    wsi_padded_total = np.zeros(list(wsi_padded.shape) + [3])
    wsi_padded_total[:, :, 0] = wsi_padded
    wsi_padded_total[:, :, 1] = np.pad(wsi[:, :, 1], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')
    wsi_padded_total[:, :, 2] = np.pad(wsi[:, :, 2], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')

    wsi_padded = np.pad(wsi_c[:, :, 0], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')
    wsi_c_padded_total = np.zeros(list(wsi_padded.shape) + [3])
    wsi_c_padded_total[:, :, 0] = wsi_padded
    wsi_c_padded_total[:, :, 1] = np.pad(wsi_c[:, :, 1], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')
    wsi_c_padded_total[:, :, 2] = np.pad(wsi_c[:, :, 2], ((pad_pixels, pad_pixels), (pad_pixels, pad_pixels+legend_margin)), 'maximum')
    return wsi_padded_total, wsi_c_padded_total, slide_clusters

# Save WSI with current resolution.
def save_wsi(file_path, wsi, slide_clusters=None, colors=None, dpi=100):
    if slide_clusters is not None:
        from matplotlib.lines import Line2D
        image_clusters, counts = np.unique(slide_clusters, return_counts=True)
        custom_lines = [Line2D([0], [0], color=colors[image_clusters[index]], lw=2.5) for index in np.argsort(-counts)]
        names_lines  = ['Cluster %s' % str(image_clusters[index]) for index in np.argsort(-counts)]

    height, width, _ = wsi.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    ax.imshow(wsi/255.)
    ax.axis('off')

    if slide_clusters is not None:
        legend = ax.legend(custom_lines, names_lines, title='Leiden Clusters', loc='upper right', prop={'size': 4})
        legend.get_title().set_fontsize('5')
        legend.get_frame().set_linewidth(1)
    fig.savefig(file_path, dpi=dpi, transparent=True)
    plt.close(fig)

def get_slides_wsi_overlay(all_slides, slides_rep_df, cluster_ids, only_id, n_wsi_samples=3):
    random.shuffle(all_slides)
    selected_slides = dict()
    if not only_id:
        for slide in all_slides[-10:]:
            cluster_id = slides_rep_df.loc[slides_rep_df.index==slide].idxmax(axis=1).values[0]
            if cluster_id not in selected_slides:
                selected_slides[cluster_id] = list()

    for cluster_id in cluster_ids:
        if slides_rep_df.sort_values(by=cluster_id).index.values.shape[0]==0: continue
        if cluster_id not in selected_slides:
            selected_slides[cluster_id] = list()
        selected_slides[cluster_id].extend(slides_rep_df.sort_values(by=cluster_id).index.values[-n_wsi_samples:])

    return selected_slides

def plot_wsi_clusters(groupby, meta_folder, matching_field, meta_field, data, fold, h5_complete_path, h5_additional_path, additional_as_fold, dpi, min_tiles, manifest_csv, value_cluster_ids,
                      type_='percent', only_id=False, n_wsi_samples=3):
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    adatas_path       = os.path.join(main_cluster_path, 'adatas')
    leiden_path       = os.path.join(main_cluster_path, '%s_fold%s' % (groupby.replace('.', 'p'), fold))
    wsi_path          = os.path.join(leiden_path, 'wsi_clusters')
    if not os.path.isdir(wsi_path):
        os.makedirs(wsi_path)

    # Data Class with all h5, these contain the images.
    data_dicts = dict()
    data_dicts['train'] = data.training.images
    data_dicts['valid'] = data.validation.images
    data_dicts['test']  = data.test.images

    # Base name for each set representations.
    adata_name     = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold)

    # Read frames.
    _, frame_complete, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, fold, [[],[],[]], h5_complete_path, h5_additional_path, additional_as_fold, force_fold=None)
    colors = sns.color_palette('tab20', len(leiden_clusters))

    # Read GDC Manifest
    gdc_frame = pd.read_csv(manifest_csv, delimiter='\t')
    gdc_frame['filename'] = gdc_frame['filename'].apply(lambda p: p.split('.')[0])

    # Get sample representations
    slide_rep_df = prepare_set_representation(frame_complete, matching_field, meta_field, groupby, leiden_clusters, type_=type_, min_tiles=min_tiles)

    dropped_slides = list()
    # Save WSI with cluster overlay.
    for value in np.unique(slide_rep_df[meta_field].values):
        print('Meta Field %s Value: %s' % (meta_field, value))
        all_value_slides = slide_rep_df[slide_rep_df[meta_field]==value].index.tolist()
        selected_slides = get_slides_wsi_overlay(all_value_slides, slide_rep_df[leiden_clusters], value_cluster_ids[value], only_id=only_id, n_wsi_samples=n_wsi_samples)
        for cluster_id in selected_slides:
            print('\tCluster %s' % cluster_id)
            for slide in selected_slides[cluster_id]:
                wsi, wsi_c, slide_clusters = get_wsi_arrays(frame_complete, groupby, slide, img_size=224, downsample=2, img_dicts=data_dicts, colors=colors, pad_pixels=0, legend_margin=1000)
                try:
                    save_wsi(os.path.join(wsi_path, 'DominantCluster_%s-%s-Original.jpg' % (cluster_id, slide)), wsi, slide_clusters=None, colors=None, dpi=dpi)
                    dropped_slides.append((slide, value, cluster_id))
                except:
                    print('Issue with slide:', slide)
                finally:
                    del wsi
                    gc.collect()

                try:
                    save_wsi(os.path.join(wsi_path, 'DominantCluster_%s-%s-Clusters.jpg' % (cluster_id, slide)), wsi_c, slide_clusters, colors, dpi=dpi)
                except:
                    print('Issue with slide cluster overlay:', slide)
                finally:
                    del wsi_c
                    gc.collect()

    # Drop annotations file for WSI.
    all_data_csv = list()
    for slide, value, cluster_id in dropped_slides:
        percentage = np.round(slide_rep_df[slide_rep_df.index==slide][cluster_id].values[0]*100,2)
        if value:
            value = 'Favors'
        else:
            value = 'Against'
        try:
            wsi_link = 'https://portal.gdc.cancer.gov/files/%s' % gdc_frame[gdc_frame['filename']==slide]['id'].values[0]
        finally:
            wsi_link = None
        all_data_csv.append((slide, cluster_id, percentage, value, '', wsi_link))
    all_data_csv = pd.DataFrame(all_data_csv, columns=['Slide', 'Dominant Cluster', 'Percentage Dominant', 'Event Indicator', 'Annotations', 'Link'])
    all_data_csv.to_csv(os.path.join(leiden_path, 'wsi_annotations.csv'), index=False)

def plot_wsi_clusters_interactions(groupby, meta_folder, matching_field, meta_field, data, fold, h5_complete_path, h5_additional_path, additional_as_fold, dpi, min_tiles, manifest_csv,
                                   inter_dict, type_='percent', only_id=False, n_wsi_samples=3):
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_folder)
    adatas_path       = os.path.join(main_cluster_path, 'adatas')
    leiden_path       = os.path.join(main_cluster_path, '%s_fold%s' % (groupby.replace('.', 'p'), fold))
    wsi_path          = os.path.join(leiden_path, 'wsi_clusters_interactions')
    interactions_path = os.path.join(leiden_path, 'interactions')
    if not os.path.isdir(wsi_path):
        os.makedirs(wsi_path)

    # Data Class with all h5, these contain the images.
    data_dicts = dict()
    data_dicts['train'] = data.training.images
    data_dicts['valid'] = data.validation.images
    data_dicts['test']  = data.test.images

    # Base name for each set representations.
    adata_name     = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), fold)

    # Read frames.
    _, frame_complete, leiden_clusters = read_csvs(adatas_path, matching_field, groupby, fold, [[],[],[]], h5_complete_path, h5_additional_path, additional_as_fold, force_fold=None)
    colors = sns.color_palette('tab20', len(leiden_clusters))

    # Read GDC Manifest
    gdc_frame = pd.read_csv(manifest_csv, delimiter='\t')
    gdc_frame['filename'] = gdc_frame['filename'].apply(lambda p: p.split('.')[0])

    # Get cluster interactions. If not create file.
    file_name     = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s_%s_cluster_interactions' % (groupby.replace('.', 'p'), fold, meta_folder)
    file_path     = os.path.join(interactions_path, file_name + '.csv')
    if os.path.isfile(file_path):
        frame_conn = pd.read_csv(file_path)
    else:
        frame_conn = include_tile_connections_frame(frame=frame_complete, groupby=groupby)
        frame_conn.to_csv(file_path, index=False)

    # Get cluster interactions per matching field.
    cluster_interactions = cluster_conn_slides(frame_conn, leiden_clusters, groupby, type='slide', own_corr=True, min_tiles=min_tiles, type_=type_)

    # Gather slides per interaction.
    wsi_dict = dict()
    for out_type in inter_dict:
        wsi_dict[out_type] = dict()
        for interaction in inter_dict[out_type]:
            values = [int(val) for val in interaction.split('_')]
            values.sort()
            values = [str(val) for val in values]
            interaction = '_'.join(values)
            wsi_dict[out_type][interaction] = list()
            sorted_frame = cluster_interactions[0].copy(deep=True).sort_values(by=interaction, ascending=False)
            slides = sorted_frame[matching_field].values.tolist()[:n_wsi_samples]
            wsi_dict[out_type][interaction].extend(slides)

    dropped_slides = list()
    # Save WSI with cluster overlay.
    for out_type in wsi_dict:
        print('Outcome type %s:' % (out_type))
        for interaction in wsi_dict[out_type]:
            print('\tInteraction %s' % interaction)
            for slide in wsi_dict[out_type][interaction]:
                wsi, wsi_c, slide_clusters = get_wsi_arrays(frame_complete, groupby, slide, img_size=224, downsample=2, img_dicts=data_dicts, colors=colors, pad_pixels=0, legend_margin=1000)
                try:
                    save_wsi(os.path.join(wsi_path, 'DominantCluster_%s-%s-Original.jpg' % (interaction, slide)), wsi, slide_clusters=None, colors=None, dpi=dpi)
                    dropped_slides.append((slide, out_type, interaction))
                except:
                    print('\t\tIssue with slide                :', slide)
                finally:
                    del wsi
                    gc.collect()

                try:
                    save_wsi(os.path.join(wsi_path, 'DominantCluster_%s-%s-Clusters.jpg' % (interaction, slide)), wsi_c, slide_clusters, colors, dpi=dpi)
                except:
                    print('\t\tIssue with slide cluster overlay:', slide)
                finally:
                    del wsi_c
                    gc.collect()

    # Drop annotations file for WSI.
    all_data_csv = list()
    for slide, value, interaction in dropped_slides:
        percentage = cluster_interactions[0][cluster_interactions[0][matching_field]==slide][interaction].values[0]
        try:
            wsi_link = 'https://portal.gdc.cancer.gov/files/%s' % gdc_frame[gdc_frame['filename']==slide]['id'].values[0]
        finally:
            wsi_link = None
        all_data_csv.append((slide, interaction, percentage, '', wsi_link))
    all_data_csv = pd.DataFrame(all_data_csv, columns=['Slide', 'Dominant Interaction', 'Percentage Interaction Slide', 'Annotations', 'Link'])
    all_data_csv.to_csv(os.path.join(leiden_path, 'wsi_annotations_interactions.csv'), index=False)


''' ###### Logistic Regression Performances ###### '''
def plot_confusion_matrix_lr(cms, directory, file_name):
    fig_path = os.path.join(directory, 'confusion_matrices')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    for label in cms:
        for cm, name in zip(cms[label], ['test', 'additional']):
            if cm is None:
                continue
            file_path = os.path.join(fig_path, file_name.replace('.jpg', '_label%s_%s.jpg' % (label, name)))
            print(cm)
            cm  = pd.DataFrame(cm, dtype=int)
            fig = plt.figure(figsize=(10,10))
            ax  = fig.add_subplot(1, 1, 1)
            sns.heatmap(cm, annot=True, ax=ax, cmap='rocket_r')
            plt.savefig(file_path)
            plt.close(fig)
