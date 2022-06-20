from matplotlib.patches import Ellipse
from collections import OrderedDict
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from sklearn import mixture
import numpy as np
import math


def draw_ellipse(position, covariance, cm, label, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        nsig = 2
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs, facecolor=cm(label)))
        
def plot_gmm(gmm, X, label=True, elip=True):
    cm = plt.cm.get_cmap('viridis_r', 7)
    fig, big_axes = plt.subplots(figsize=(10,10), nrows=1, ncols=1, sharex=True, sharey=True)
    labels = gmm.fit(X).predict(X)
    if label:
        big_axes.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap=cm, zorder=2)
    else:
        big_axes.scatter(X[:, 0], X[:, 1], s=5, zorder=2)

    if elip:
        w_factor = 0.2 / gmm.weights_.max()
        for label, data in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
            pos, covar, w = data
            draw_ellipse(pos, covar, alpha=w*w_factor, cm=cm, label=label)
    
    return labels

def combine_images(imgs, spacing=0.07):
    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :, :, :]
    num_images, height, width, channels = imgs.shape
    offset_h = int(spacing*height)
    offset_w = int(spacing*width)
    
    grid = np.ones((2*height+offset_h, 2*width+offset_w, channels))
    for i in range(0,num_images):
        img_s = imgs[i, :, :, :]/255.
        if i == 0:
            grid[0:height, 0:width, :] = img_s
        elif i ==1:
            grid[0:height, width+offset_w:, :] = img_s
        elif i ==2:
            grid[height+offset_h:, :width] = img_s
        else:
            grid[height+offset_h:, width+offset_w:, :] = img_s
    return grid

def get_rois(c=(0,0), r=15, points=12):
    rois = list()
    for phi in reversed(range(150, 510, int(360/points))):
        x = r*math.cos(phi/180*math.pi) + c[0]
        y = r*math.sin(phi/180*math.pi) + c[1]
        rois.append((x,y))
    return rois

def get_images_gmm(labels):
    labels_u = np.unique(labels)
    l_ind = OrderedDict()
    for l in labels_u:
        l_ind[l] = np.argwhere(labels==l)
    return l_ind

def get_images_label(l_ind, label, imgs):
    indeces = l_ind[label][:4]
    if len(indeces.shape) > 1:
        indeces = np.ravel(indeces)
    return imgs[indeces, :, :, :]

def git_gmm_roi_images(gmm, embedding, images, rois):
    # Prediction for all points, assignations to mixture.
    gmm_labels = gmm.predict(embedding)
    # Per Mixture, indeces.
    l_ind = get_images_gmm(gmm_labels)
    rois_img = list()
    taken = list()
    for roi in rois:
        mean, label = find_gaussian(gmm, roi, l_ind, taken)
        taken.append(label)
        label_img = get_images_label(l_ind, label, images)
        rois_img.append((label_img, mean))
    return rois_img

def find_gaussian(gmm, xy, l_ind, taken=None):
    xy = np.array(xy)
    distance = np.Infinity
    min_mean = None
    min_label = None
    for label, mean in enumerate(gmm.means_):
        new = np.sqrt(np.sum(np.square(mean-xy)))
        if new < distance and label not in taken and label in l_ind: 
            distance = new
            min_mean = mean
            min_label = label
    return min_mean, min_label

def find_linear_interpolations(start, end, gmm, n_points):
    distance_start = np.Infinity
    distance_end = np.Infinity
    min_start_mean = None
    min_end_mean = None
    
    for label, mean in enumerate(gmm.means_):
        start_new = np.sqrt(np.sum(np.square(mean-start)))
        end_new = np.sqrt(np.sum(np.square(mean-end)))
        if start_new < distance_start: 
            distance_start = start_new
            min_start_mean = mean
            start_label = label
        if end_new < distance_end: 
            distance_end = end_new
            min_end_mean = mean
            end_label = label
            
    return np.linspace(min_start_mean, min_end_mean, n_points)

def find_closest_real(interpolation_points, gmm_embedding, dataset_img):
    inter_indeces_real = list()
    inter_imgs_real = list()
    for point in interpolation_points:        
        distance = np.Infinity
        for real_index in range(gmm_embedding.shape[0]):
            real_point = gmm_embedding[real_index]
            new = np.sqrt(np.sum(np.square(point-real_point)))
            if new < distance:
                distance = new
                closes_point = real_index
        inter_indeces_real.append(closes_point)
        inter_imgs_real.append(dataset_img[closes_point])
    return inter_indeces_real, inter_imgs_real

def plot_latent_space(embedding, images, img_path, field_names, file_name, labels=None, legend_title=None, radius_rate=0.8, scatter_plot=True, scatter_size=1, n_cells=4, size=20, figsize=(20,20), 
                      gmm_components=1000, x_lim=None, y_lim=None, cmap='viridis'):
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn import mixture
    
    # CMAP for coloring.
    cm = plt.cm.get_cmap(cmap, 7)
    
    # Figure ratios and total size.
    widths = list(np.ones((n_cells,))*2)
    heights = list(np.ones((n_cells,))*2)
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    plt.ioff()
    fig6, f6_axes = plt.subplots(figsize=figsize, ncols=n_cells, nrows=n_cells, gridspec_kw=gs_kw)
    
    # Remove label ticks from image subplots.
    for ax_row in f6_axes:
        for ax in ax_row:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
    
    # Combine four subplots for big plot on latent space.
    for ax_i in f6_axes[1:3, 1:3]:
        for ax in ax_i:
            ax.remove()
    gs = f6_axes[2,2].get_gridspec()
    main_ax = fig6.add_subplot(gs[1:3, 1:3])
    main_ax.axis('on')
    main_ax.set_xlabel('UMAP dimension 1', size=size)
    main_ax.set_ylabel('UMAP dimension 2', size=size)
    main_ax.tick_params(labelsize=size-2)

    
    if legend_title is None: 
        legend_title = 'Classes'
    
    # Scatter plot.
    if scatter_plot:
        # Plot latent space on big axes.
        if labels is not None:
            scatter1 = main_ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=scatter_size, cmap=cm)
            if field_names is not None:
                legend1 = main_ax.legend(scatter1.legend_elements()[0], field_names, loc="best", title=legend_title, prop={'size': size-8})
            else:
                legend1 = main_ax.legend(*scatter1.legend_elements(), loc="best", title=legend_title, prop={'size': size-8})
            plt.setp(legend1.get_title(),fontsize=size)
        else:
            scatter1 = main_ax.scatter(embedding[:, 0], embedding[:, 1], s=scatter_size, cmap=cm)

        if x_lim is not None and y_lim is not None:
            main_ax.set_xlim(x_lim[0], x_lim[1])
            main_ax.set_ylim(y_lim[0], y_lim[1])
        else:
            x_lim = plt.xlim()
            y_lim = plt.ylim()

    # Density plot
    else:
        x = embedding[:, 0]
        y = embedding[:, 1]
        nbins=150
        k = kde([x,y])
        xi, yi = np.mgrid[x_lim[0]:x_lim[1]:nbins*1j, y_lim[0]:y_lim[1]:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Make the plot
        a = main_ax.pcolormesh(xi, yi, zi.reshape(xi.shape), label=labels)
        divider = make_axes_locatable(main_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(a, cax)
        cbar.ax.tick_params(labelsize=size-8) 

    # Regions of interest for images.
    x_c = (x_lim[0] + x_lim[1])/2.
    y_c = (y_lim[0] + y_lim[1])/2.
    x_r = x_lim[1] - x_c
    y_r = y_lim[1] - y_c
    r = x_r
    if y_r < r:
        r = y_r
    rois = get_rois(c=(x_c,y_c), r=r*radius_rate, points=12)
    
    # Fit GMM to select images
    gmm = mixture.GaussianMixture(n_components=gmm_components, covariance_type='full')
    gmm.fit(embedding)
    
    # Pull closest representations to points in rois.
    rois_img = git_gmm_roi_images(gmm, embedding, images, rois)
    
    # Plot images of regions of interest.
    for roi_i, m_img in enumerate(rois_img):
        img, mean = m_img

        # Row indexing.
        if roi_i in [0, 1, 2, 3]: row_index = 0
        elif roi_i in [4, 11]: row_index = 1
        elif roi_i in [5, 10]: row_index = 2
        else: row_index = 3

        # Column indexing.
        if roi_i in [0, 11, 10, 9]: col_index = 0
        elif roi_i in [1, 8]: col_index = 1
        elif roi_i in [2, 7]: col_index = 2
        else: col_index = 3
        
        if len(img.shape) == 5:
            img = img[:, 0, :, :, :]
        if np.max(img)< 2: img = img*255.
        
        # Plot combined images.
        combined_img = combine_images(img)
        if combined_img.shape[-1] == 1:
            combined_img = combined_img[:,:,0]
            height, width = combined_img.shape
        else:
            height, width, channels = combined_img.shape
        f6_axes[row_index, col_index].imshow(combined_img)
        

        if roi_i == 0: box_coor = [height, width]
        elif roi_i ==1 or roi_i ==2: box_coor = [int(height/2), width]
        elif roi_i ==3: box_coor = [0, width]
        elif roi_i ==4 or roi_i ==5: box_coor = [0, int(height/2)]
        elif roi_i ==6: box_coor = [0, 0]
        elif roi_i ==7 or roi_i ==8: box_coor = [int(height/2), 0]
        elif roi_i ==9: box_coor = [height, 0]
        elif roi_i ==10 or roi_i ==11: box_coor = [width, int(height/2)]
        
        axesA = f6_axes[row_index, col_index]

        color="black"
        con = ConnectionPatch(xyA=mean, xyB=box_coor, coordsA="data", coordsB="data", axesA=main_ax, axesB=axesA, color=color, arrowstyle="-", zorder=0)
        con = ConnectionPatch(xyA=mean, xyB=box_coor, coordsA="data", coordsB="data", axesA=main_ax, axesB=axesA, color=color, arrowstyle="-", zorder=0)
        con = ConnectionPatch(xyA=mean, xyB=box_coor, coordsA="data", coordsB="data", axesA=main_ax, axesB=axesA, color=color, arrowstyle="-", zorder=0)
        main_ax.add_artist(con)
        
    plt.tight_layout()
    image_path = '%s/%s.jpg' % (img_path, file_name)
    plt.savefig(image_path)
    plt.close(fig6)
    
    if scatter_plot:
        return image_path, rois, x_lim, y_lim
    return image_path, rois

def report_progress_latent(epoch, w_samples, img_samples, img_path, label_samples=None, storage_name='w_lat', metric='euclidean'):
    from sklearn import mixture
    import umap
    import random

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42, low_memory=True, metric=metric)
    embedding_umap = reducer.fit_transform(w_samples)
    
    for radius in [0.11, 0.25, 0.51, 0.61, 0.75, 1.01, 1.25, 1.51]:
        if label_samples is not None:
            labels = label_samples[:, 0]
        else:
            labels = None
        n_components = 1000
        image_path, rois, x_lim, y_lim = plot_latent_space(embedding=embedding_umap, images=img_samples, img_path=img_path, field_names=None, file_name='%s_epoch_%s_images_latent_space_%s'% (storage_name, epoch, str(radius).replace('.', 'p')), 
                                                   labels=labels, gmm_components=n_components, legend_title=None, radius_rate=radius, scatter_plot=True, size=20, figsize=(20,20), x_lim=None, y_lim=None, cmap='Spectral')

    label_image_path = None
    if label_samples is not None:
        clusters = list(range(np.max(label_samples)+1))
        random.shuffle(clusters)
        for i in clusters[:10]:
            try:
                radius = 0.8
                class_indxs = np.argwhere(label_samples[:,0]==i)[:,0]
                class_emb = embedding_umap[class_indxs, :]
                class_labels = np.ones((class_indxs.shape[0]))*i
                class_img = img_samples[class_indxs,:,:,:]
                n_components = 20
                if class_indxs.shape[0] < 20:
                    n_components = class_indxs.shape[0]
                label_image_path, rois = plot_latent_space(embedding=class_emb, images=class_img, img_path=img_path, field_names=None, 
                                  file_name='class_%s_%s_epoch_%s_images_latent_space_%s'% (i, storage_name, epoch, str(radius).replace('.', 'p')), gmm_components=n_components,
                                  labels=class_labels, legend_title=None, radius_rate=radius, scatter_plot=True, size=20, figsize=(20,20), x_lim=x_lim, y_lim=y_lim, cmap='Spectral')
            except:
                print('Issue with cluster %s in epoch %s' % (i, epoch))
        
    return image_path, label_image_path

