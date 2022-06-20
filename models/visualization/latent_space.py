from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np
import umap

from models.evaluation.latent_space import *
from models.utils import *
from models.visualization.utils import *


def process_clusters_top(fold_cls_top_tiles_weighted, top_num=15):
	cluster_names, counts = np.unique(fold_cls_top_tiles_weighted, return_counts=True)
	indices       = list(reversed(np.argsort(counts).tolist()))[:top_num]
	top_clusters  = cluster_names[indices]
	legend_labels = ['%-6s - %s' % (int(cluster), count) for cluster, count in zip(cluster_names[indices], counts[indices])]
	
	cluster_assignations = np.zeros_like(fold_cls_top_tiles_weighted)
	for i in range(cluster_assignations.shape[0]):
		cluster_instance = fold_cls_top_tiles_weighted[i, 0]
		if cluster_instance in top_clusters:
			cluster_assignations[i] = int(cluster_instance)
		else:
			cluster_assignations[i] = 9999
			
	count = np.argwhere(cluster_assignations==9999)[:,0].shape[0]
	legend_labels.append('Others - %s' % (count))
		
	return cluster_assignations, legend_labels


def plot_latent_space(label, gmm, embedding, images, img_path, file_name, labels, legend_title=None, radius_rate=0.8, scatter_plot=True, scatter_size=1, n_cells=4, size=20, figsize=(20,20), 
					  rois=None, x_lim=None, y_lim=None, cmap='viridis', cls_top_tiles_weighted=None, ins_top_tiles_weighted=None):
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
	
	if cls_top_tiles_weighted is None and ins_top_tiles_weighted is None:
		if 'LUAD' in label:
			indices_inst = np.argwhere(labels==1)    
			scatter1 = main_ax.scatter(embedding[indices_inst, 0], embedding[indices_inst, 1], marker='o', s=scatter_size, color='blue', label='LUAD')
			indices_inst = np.argwhere(labels==0)    
			scatter1 = main_ax.scatter(embedding[indices_inst, 0], embedding[indices_inst, 1], marker='o', s=scatter_size, color='orange', label='LUSC')  
			# indices_inst = np.argwhere(labels==2)    
			# scatter1 = main_ax.scatter(embedding[indices_inst, 0], embedding[indices_inst, 1], s=scatter_size, color='red', label='Missclassified')  
			legend=main_ax.legend(loc="best", title='Cancer Subtype', prop={'size': size-12})
			plt.setp(legend.get_title(),fontsize=size-10)
		else:
			indices_inst = np.argwhere(labels==0)    
			scatter1 = main_ax.scatter(embedding[indices_inst, 0], embedding[indices_inst, 1], marker='o', s=scatter_size, color='orange', label='Other')  
			indices_inst = np.argwhere(labels==1)    
			scatter1 = main_ax.scatter(embedding[indices_inst, 0], embedding[indices_inst, 1], marker='o', s=scatter_size, color='blue', label=legend_title)
			legend=main_ax.legend(loc="best", title='Histological Subtype', prop={'size': size-12})
			plt.setp(legend.get_title(),fontsize=size-10)
		
	elif ins_top_tiles_weighted is not None:
		for inst_name in np.unique(ins_top_tiles_weighted):
			indices_inst = np.argwhere(ins_top_tiles_weighted==inst_name)    
			main_ax.scatter(embedding[indices_inst,0], embedding[indices_inst,1], marker='o', s=scatter_size, alpha=1, label=inst_name)
		# legend=main_ax.legend(loc="best", title='Institutions', prop={'size': size-12})
		# plt.setp(legend.get_title(),fontsize=size-10)
	else:
		cluster_assignations, legend_labels = process_clusters_top(cls_top_tiles_weighted)
		for string_cl in legend_labels:
			cluster = string_cl.split(' - ')[0]
			if 'Others' in cluster: 
				cluster = 9999
				indices = np.argwhere(cluster_assignations==int(cluster))[:,0]
				scatter1 = main_ax.scatter(embedding[indices, 0], embedding[indices, 1], marker='o', s=scatter_size, label=string_cl, color='black')
			else:
				indices = np.argwhere(cluster_assignations==int(cluster))[:,0]
				scatter1 = main_ax.scatter(embedding[indices, 0], embedding[indices, 1], marker='o', s=scatter_size, label=string_cl)
		legend = main_ax.legend(loc="best", title='Cluster - Counts', prop={'size': size-12})
		plt.setp(legend.get_title(),fontsize=size-10)
		
	if ins_top_tiles_weighted is None:
		for legend_handle in legend.legendHandles:
			legend_handle._sizes = [30]
		
	if x_lim is not None and y_lim is not None:
		main_ax.set_xlim(x_lim[0], x_lim[1])
		main_ax.set_ylim(y_lim[0], y_lim[1])
	else:
		x_lim = plt.xlim()
		y_lim = plt.ylim()

	# Regions of interest for images.
	x_c = (x_lim[0] + x_lim[1])/2.
	y_c = (y_lim[0] + y_lim[1])/2.
	x_r = x_lim[1] - x_c
	y_r = y_lim[1] - y_c
	r = x_r
	if y_r < r:
		r = y_r
		
	if rois is None:
		rois = get_rois(c=(x_c,y_c), r=r*radius_rate, points=12)
	
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
		
	fig6.tight_layout()
	plt.savefig('%s/%s' % (img_path, file_name))
	plt.close(fig6)
	
	if scatter_plot:
		return rois, x_lim, y_lim
	return rois


def top_tiles(label, img_top_tiles_weighted, lat_top_tiles_weighted, pat_top_tiles_weighted, dataset, weight_threshold, tiles_img_mil_path, magnification_name, scatter_size, \
			  cls_top_tiles_weighted=None, ins_top_tiles_weighted=None):
	
	if lat_top_tiles_weighted.shape[0] < 50:
		print('\tLess than 100 samples:', lat_top_tiles_weighted.shape[0])
		return
	
	umap_reducer = umap.UMAP(n_neighbors=500, min_dist=0.0, n_components=2, random_state=42, metric='euclidean', low_memory=True, verbose=False, densmap=True)
	umap_fitted  = umap_reducer.fit(lat_top_tiles_weighted)
	sel_emb      = umap_fitted.embedding_
	
	# Fit GMM to select images
	n_components = int(sel_emb.shape[0]/10)
	gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
	gmm.fit(sel_emb)
	
	for radius in [1.51, 1.01, 0.75, 0.61, 0.51, 0.25]:
		# High/Low Risk Plot
		file_name = '%s_indmag_%s_top_weighted_wthrs_%s_representations_radius_%s.jpg' % (dataset, magnification_name, weight_threshold, str(radius).replace('.', 'p'))
		plot_latent_space(label=label, embedding=sel_emb, images=img_top_tiles_weighted, img_path=tiles_img_mil_path, file_name=file_name, 
						  labels=pat_top_tiles_weighted, legend_title='Lung Subtype', radius_rate=radius, scatter_plot=True, scatter_size=scatter_size, n_cells=4, size=30, figsize=(20,20), 
						  rois=None, gmm=gmm, x_lim=None, y_lim=None, cmap='viridis', cls_top_tiles_weighted=None)
		
		if cls_top_tiles_weighted is not None:
			# Clusters Plot
			file_name = '%s_indmag_%s_top_weighted_wthrs_%s_sphericalcluster_radius_%s.jpg' % (dataset, magnification_name, weight_threshold, str(radius).replace('.', 'p'))
			plot_latent_space(label=label, embedding=sel_emb, images=img_top_tiles_weighted, img_path=tiles_img_mil_path, file_name=file_name, 
							  labels=pat_top_tiles_weighted, legend_title=None, radius_rate=radius, scatter_plot=True, scatter_size=scatter_size, n_cells=4, size=30, figsize=(20,20), 
							  rois=None, gmm=gmm, x_lim=None, y_lim=None, cmap='viridis', cls_top_tiles_weighted=cls_top_tiles_weighted)

		if ins_top_tiles_weighted is not None:
			# Clusters Plot
			file_name = '%s_indmag_%s_top_weighted_wthrs_%s_institutions_radius_%s.jpg' % (dataset, magnification_name, weight_threshold, str(radius).replace('.', 'p'))
			plot_latent_space(label=label, embedding=sel_emb, images=img_top_tiles_weighted, img_path=tiles_img_mil_path, file_name=file_name, 
							  labels=ins_top_tiles_weighted, legend_title=None, radius_rate=radius, scatter_plot=True, scatter_size=scatter_size, n_cells=4, size=25, figsize=(20,20), 
							  rois=None, gmm=gmm, x_lim=None, y_lim=None, cmap='viridis', ins_top_tiles_weighted=ins_top_tiles_weighted)


def gather_latent_img_label(label, orig_part, weight_threshold, uniq_slides, slides, patterns, weights_5x, latent_5x, train_img_5x, valid_img_5x, test_img_5x, oriind_5x, \
							orig_set, cluster_labels_5x=None, inst_labels=None, max_per_slide=200, display=True, grab_images=True):
	
	# Original Images 5x
	train_img_magnifications = train_img_5x    
	valid_img_magnifications = valid_img_5x    
	test_img_magnifications  = test_img_5x
	# Original Indices.
	oriind_magnifications    = oriind_5x
	# Latent representations.
	latent_magnitications    = latent_5x
	
	orig_part_train, orig_part_valid, orig_part_test = orig_part
	
	# Iterate through fold run slides 
	processed_slides = 0
	top_weighted_indeces  = list()
	top_weighted_patterns = list()
	for slide in uniq_slides:
		
		# This is 5x
		slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
		
		# Slide label.
		class_ = 0
		if label in patterns[slide_indices[0],0]:
			class_ = 1
			
		label_latent = class_
			
		processed_slides += 1
			
		if not (weights_5x[slide_indices,0] == weights_5x[slide_indices,0][0]).all():
			slide_weights_5x = weights_5x[slide_indices,0]
			slide_weights    = (slide_weights_5x - np.min(slide_weights_5x))/(np.max(slide_weights_5x)-np.min(slide_weights_5x))
		else:
			slide_weights = weights_5x[slide_indices,0]/weights_5x[slide_indices,0][0]
		
		# Threshold for weighted latent to pick
		slide_samples      = 0
		# Select top weighted first.
		top_weighted_slide = np.argwhere(slide_weights>weight_threshold)[:,0]
        # top_weighted_slide = list(reversed(np.argsort(slide_weights)))
		for index in top_weighted_slide:
			total_index = slide_indices[index]
			top_weighted_indeces.append(total_index)
			top_weighted_patterns.append(label_latent)
			slide_samples += 1
			if (max_per_slide == slide_samples) or (slide_weights[index]<weight_threshold):
				break
				
		if display: print('\t\tSlide',  slide, class_, 'samples:', slide_samples, 'institution:', inst_labels[total_index])
	if display: print('\tProcessed Slides:', processed_slides)
	
	# We need to relate now to the original dataset, and filter out the 5x that don't have a latent. 
	img_top_tiles_weighted = list()
	lat_top_tiles_weighted = list()
	pat_top_tiles_weighted = list()   
	ind_top_tiles_weighted = list()    
	cls_top_tiles_weighted = None  
	ins_top_tiles_weighted = None
	if cluster_labels_5x is not None:
		cls_top_tiles_weighted = list()
	if inst_labels is not None:
		ins_top_tiles_weighted = list()
	# Iterate
	for i, ind_i in enumerate(top_weighted_indeces):
		orig_ind    = oriind_magnifications[ind_i].astype(np.int32)
		latent_ind  = latent_magnitications[ind_i] 
		
		if orig_ind == 0:
			continue
		
		original_set = None
		if slides[ind_i] in orig_part_train:
			original_set = 'train'
		if slides[ind_i] in orig_part_valid:
			if original_set is not None:
				print('\t\t\tMultiple')
				continue
			original_set = 'valid'
		if slides[ind_i] in orig_part_test:
			if original_set is not None:
				print('\t\t\tMultiple')
				continue
			original_set = 'test'
			
		if original_set is None:
			print(slide)
			return
			
		if orig_set[ind_i] != original_set:
			print('\t\t\t%s' % slides[ind_i, 0], 'Wrong match original set:', original_set, 'orig_set', orig_set[ind_i, 0], ind_i)
			continue
			
		# Image handling.
		if 'train' in orig_set[ind_i,0]:
			img = train_img_magnifications[orig_ind,:,:,:]
		elif 'valid' in orig_set[ind_i,0]:
			img = valid_img_magnifications[orig_ind,:,:,:]
		elif 'test' in orig_set[ind_i,0]:
			img = test_img_magnifications[orig_ind,:,:,:]
			
		if grab_images: 
			img_top_tiles_weighted.append(img)
		lat_top_tiles_weighted.append(latent_ind)
		pat_top_tiles_weighted.append(top_weighted_patterns[i])
		ind_top_tiles_weighted.append(ind_i)
		if cluster_labels_5x is not None:
			cls_top_tiles_weighted.append(cluster_labels_5x[ind_i])
		if inst_labels is not None:
			ins_top_tiles_weighted.append(inst_labels[ind_i])
			
	if len(lat_top_tiles_weighted) != 0:
		if grab_images: img_top_tiles_weighted = np.vstack(img_top_tiles_weighted)
		lat_top_tiles_weighted = np.vstack(lat_top_tiles_weighted)
		pat_top_tiles_weighted = np.vstack(pat_top_tiles_weighted)
		ind_top_tiles_weighted = np.vstack(ind_top_tiles_weighted)
		if cluster_labels_5x is not None:
			cls_top_tiles_weighted = np.vstack(cls_top_tiles_weighted)
		if inst_labels is not None:
			ins_top_tiles_weighted = np.vstack(ins_top_tiles_weighted)
	else:
		img_top_tiles_weighted = np.zeros((1,1))
		lat_top_tiles_weighted = np.zeros((1,1))
		pat_top_tiles_weighted = np.zeros((1,1))
		cls_top_tiles_weighted = np.zeros((1,1))
		ins_top_tiles_weighted = np.zeros((1,1))
		ind_top_tiles_weighted = np.zeros((1,1))
		
	return img_top_tiles_weighted, lat_top_tiles_weighted, pat_top_tiles_weighted, cls_top_tiles_weighted, ins_top_tiles_weighted, ind_top_tiles_weighted


def gather_latent_img_label_other(label, weight_threshold, uniq_slides, magnification_name, scatter_size, probs, slides, patterns, orig_set, weights_5x, weights_otherx, latent_otherx, \
								  train_img_otherx, valid_img_otherx, test_img_otherx, oriind_otherx, cluster_labels_otherx=None, inst_labels=None, img_size=224, max_per_slide=200, \
								  display=True, grab_images=True):
	
	# Original Images 5x
	train_img_magnifications = train_img_otherx    
	valid_img_magnifications = valid_img_otherx
	test_img_magnifications  = test_img_otherx
	# Latent representations.
	latent_magnitications = latent_otherx
	# Original Indices.
	oriind_magnifications = oriind_otherx
	
	# Reformat 20x latent and indices.
	if '20x' in magnification_name:
		latent_magnitications = np.reshape(latent_magnitications, (latent_magnitications.shape[0], -1, latent_magnitications.shape[-1]))
		oriind_magnifications = np.reshape(oriind_magnifications, (oriind_magnifications.shape[0], -1, oriind_magnifications.shape[-1]))
		cluster_labels_otherx = np.reshape(cluster_labels_otherx, (cluster_labels_otherx.shape[0], -1, cluster_labels_otherx.shape[-1]))
	
	processed_slides = 0
	img_top_tiles_weighted = list()
	lat_top_tiles_weighted = list()
	pat_top_tiles_weighted = list() 
	ind_top_tiles_weighted = list() 
	cls_top_tiles_weighted = None  
	ins_top_tiles_weighted = None
	if cluster_labels_otherx is not None:
		cls_top_tiles_weighted = list()
	if inst_labels is not None:
		ins_top_tiles_weighted = list()
	for slide in uniq_slides:
		# This is 5x
		slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
		
		# Slide label.
		class_ = 0
		if label in patterns[slide_indices[0],0]:
			class_ = 1
		
		processed_slides += 1
							   
		site = inst_labels[slide_indices[0]]
		slide_latents    = latent_magnitications[slide_indices]
		slide_oriind     = oriind_magnifications[slide_indices]
		if cluster_labels_otherx is not None: slide_clusters = cluster_labels_otherx[slide_indices]
		
		slide_weights_other = weights_otherx[slide_indices, :, 0]*np.reshape(weights_5x[slide_indices,0], (-1, 1))
		slide_weights_other = np.reshape(slide_weights_other, (-1,1))
		slide_weights       = (slide_weights_other - np.min(slide_weights_other))/(np.max(slide_weights_other)-np.min(slide_weights_other))
		slide_weights       = np.reshape(slide_weights, (-1, weights_otherx.shape[1], 1))
		
		flag_max_slide = False
		slide_samples = 0
		# Threshold for weighted latent to pick
		top_weighted_slide   = slide_weights>weight_threshold
		for i_5x in range(top_weighted_slide.shape[0]):
			for k in range(top_weighted_slide.shape[1]):
				flag = top_weighted_slide[i_5x, k, 0]
				if flag:
					orig_ind   = slide_oriind[i_5x, k, 0]
					latent_ind = slide_latents[i_5x, k, :]
					orig_set_st = orig_set[slide_indices[i_5x]]
					if orig_ind == 0:
						continue
					if 'train' in orig_set_st:
						img = train_img_magnifications[orig_ind, :, :, :].reshape(1, img_size, img_size, 3)
					elif 'valid' in orig_set_st:
						img = valid_img_magnifications[orig_ind, :, :, :].reshape(1, img_size, img_size, 3)
					elif 'test' in orig_set_st:
						img = test_img_magnifications[orig_ind, :, :, :].reshape(1, img_size, img_size, 3)
					if grab_images:
						img_top_tiles_weighted.append(img)
					lat_top_tiles_weighted.append(slide_latents[i_5x, k, :])
					pat_top_tiles_weighted.append(class_)
					ind_top_tiles_weighted.append((i_5x, k))
					if cluster_labels_otherx is not None:
						cls_top_tiles_weighted.append(slide_clusters[i_5x, k])
					if inst_labels is not None:
						ins_top_tiles_weighted.append(site)
					slide_samples += 1
					if max_per_slide == slide_samples:
						flag_max_slide = True
						break
			if flag_max_slide:
				break
		if display: print('\t\tSlide',  slide, class_, 'samples:', slide_samples, 'institutions:', site)                    
	if display: print('\tProcessed Slides:', processed_slides)
	
	if len(lat_top_tiles_weighted) != 0:
		if grab_images:
			img_top_tiles_weighted = np.vstack(img_top_tiles_weighted)
		lat_top_tiles_weighted = np.vstack(lat_top_tiles_weighted)
		pat_top_tiles_weighted = np.vstack(pat_top_tiles_weighted)
		ind_top_tiles_weighted = np.vstack(ind_top_tiles_weighted)
		if cluster_labels_otherx is not None:
			cls_top_tiles_weighted = np.vstack(cls_top_tiles_weighted)
		if inst_labels is not None:
			ins_top_tiles_weighted = np.vstack(ins_top_tiles_weighted)
	else:
		img_top_tiles_weighted = np.zeros((1,1))
		lat_top_tiles_weighted = np.zeros((1,1))
		pat_top_tiles_weighted = np.zeros((1,1))
		cls_top_tiles_weighted = np.zeros((1,1))
		ins_top_tiles_weighted = np.zeros((1,1))
		ind_top_tiles_weighted = np.zeros((1,1))
		
	return [img_top_tiles_weighted, lat_top_tiles_weighted, pat_top_tiles_weighted, cls_top_tiles_weighted, ins_top_tiles_weighted, ind_top_tiles_weighted]


def append_top_weighted(otw, fold_top_weighted, fold_number, num_runs):
	img_top_tiles_weighted, lat_top_tiles_weighted, pat_top_tiles_weighted, cls_top_tiles_weighted, ins_top_tiles_weighted, ind_top_tiles_weighted = otw
	fold_img_top_tiles_weighted, fold_lat_top_tiles_weighted, fold_pat_top_tiles_weighted, fold_cls_top_tiles_weighted, fold_ins_top_tiles_weighted, fold_ind_top_tiles_weighted = fold_top_weighted
	
	if lat_top_tiles_weighted.shape == (1,1): 
			print('Issue with fold')
			return fold_top_weighted
	
	fold_img_top_tiles_weighted.extend(img_top_tiles_weighted)
	fold_lat_top_tiles_weighted.extend(lat_top_tiles_weighted)
	fold_pat_top_tiles_weighted.extend(pat_top_tiles_weighted)
	fold_cls_top_tiles_weighted.extend(cls_top_tiles_weighted)
	fold_ins_top_tiles_weighted.extend(ins_top_tiles_weighted)
	fold_ind_top_tiles_weighted.extend(ind_top_tiles_weighted)
	
	if fold_number == num_runs -1:
		if len(fold_img_top_tiles_weighted) > 0:
			fold_img_top_tiles_weighted = np.array(fold_img_top_tiles_weighted)
		fold_lat_top_tiles_weighted = np.array(fold_lat_top_tiles_weighted)
		fold_pat_top_tiles_weighted = np.array(fold_pat_top_tiles_weighted)
		fold_cls_top_tiles_weighted = np.array(fold_cls_top_tiles_weighted)
		fold_ins_top_tiles_weighted = np.array(fold_ins_top_tiles_weighted)
		fold_ind_top_tiles_weighted = np.array(fold_ind_top_tiles_weighted)
	
	return [fold_img_top_tiles_weighted, fold_lat_top_tiles_weighted, fold_pat_top_tiles_weighted, fold_cls_top_tiles_weighted, fold_ins_top_tiles_weighted, fold_ind_top_tiles_weighted]


def plot_latent_space_5x(label, slides, patterns, img_5x, latent_5x, oriind_5x, orig_set, cluster_labels_5x, institutions, orig_part, dataset, directories, fold_path, max_per_slide, num_folds, h5_file_name, display=False):
	histograms_img_mil_path, latent_paths, wsi_paths, miss_wsi_paths = directories
	train_img_5x,  valid_img_5x,  test_img_5x = img_5x

	ths_5X     = [0.9, 0.8, 0.7] 
	ths_5X     = [0.989, 0.95, 0.9] 

	for th_5X in ths_5X:        
		weight_threshold = th_5X
		print('\t\tWeight Threshold', weight_threshold)

		fold_top_weighted = [list(), list(), list(), list(), list(), list()]

		for fold_number in range(num_folds):
			print('\t\t\tFold', fold_number)

			# Construct H5 for fold.
			hdf5_path_weights_comb  = '%s/fold_%s/results/%s' % (fold_path, fold_number, h5_file_name)

			# Gather fold weights, predictions, labels, and others.
			weights_20x, weights_10x, weights_5x, probs, slides_metrics, train_slides, valid_slides, test_slides = gather_attention_results(hdf5_path_weights_comb)

			# Weights distribution per patient.
			otw = gather_latent_img_label(label, orig_part, weight_threshold, test_slides, slides, patterns, weights_5x, latent_5x, train_img_5x, valid_img_5x, test_img_5x, oriind_5x, orig_set, cluster_labels_5x, institutions[:,0], max_per_slide=max_per_slide, display=display, grab_images=True)

			# Append top weighted information
			fold_top_weighted = append_top_weighted(otw, fold_top_weighted, fold_number, num_folds)

		print('\t\tNumber of tiles:', fold_top_weighted[1].shape[0], 'Max Tiles per Slide:', max_per_slide)		
		top_tiles(label, fold_top_weighted[0], fold_top_weighted[1], fold_top_weighted[2], dataset, weight_threshold, latent_paths[0], magnification_name='5x', scatter_size=2, \
				  cls_top_tiles_weighted=fold_top_weighted[3], ins_top_tiles_weighted=fold_top_weighted[4])

		# Delete information from memory.
		del fold_top_weighted
		

def plot_latent_space_10x(label, slides, patterns, img_10x, latent_10x, oriind_10x, orig_set, cluster_labels_10x, institutions, orig_part, dataset, directories, fold_path, img_size, max_per_slide, num_folds, h5_file_name, display=False):
	histograms_img_mil_path, latent_paths, wsi_paths, miss_wsi_paths = directories
	train_img_10x,  valid_img_10x,  test_img_10x = img_10x

	ths_otherX = [0.9, 0.8, 0.7, 0.6] 
	ths_otherX = [0.9, 0.8] 
	for th_otherX in ths_otherX:

		weight_threshold = th_otherX
		print('\t\tWeight Threshold', weight_threshold)
		fold_top_weighted = [list(), list(), list(), list(), list(), list()]

		for fold_number in range(num_folds):
			print('\t\t\tFold', fold_number)

			# Construct H5 for fold.
			hdf5_path_weights_comb  = '%s/fold_%s/results/%s' % (fold_path, fold_number, h5_file_name)

			# Gather fold weights, predictions, labels, and others.
			weights_20x, weights_10x, weights_5x, probs, slides_metrics, train_slides, valid_slides, test_slides = gather_attention_results(hdf5_path_weights_comb)

			# Weights distribution per patient.
			otw = gather_latent_img_label_other(label, weight_threshold, test_slides, '10x', scatter_size=5, probs=probs, slides=slides, patterns=patterns, orig_set=orig_set, weights_5x=weights_5x, \
												weights_otherx=weights_10x, latent_otherx=latent_10x, train_img_otherx=train_img_10x, valid_img_otherx=valid_img_10x, test_img_otherx=test_img_10x, \
												oriind_otherx=oriind_10x, cluster_labels_otherx=cluster_labels_10x, inst_labels=institutions[:,0], max_per_slide=max_per_slide, \
												img_size=img_size, display=display, grab_images=True)

			# Append top weighted information
			fold_top_weighted = append_top_weighted(otw, fold_top_weighted, fold_number, num_folds)

		print('\t\tNumber of tiles:', fold_top_weighted[1].shape[0], 'Max Tiles per Slide:', max_per_slide)	
		top_tiles(label, fold_top_weighted[0], fold_top_weighted[1], fold_top_weighted[2], dataset, weight_threshold, latent_paths[1], magnification_name='10x', scatter_size=2, \
				  cls_top_tiles_weighted=fold_top_weighted[3], ins_top_tiles_weighted=fold_top_weighted[4])

		# Delete information from memory.
		del fold_top_weighted
	

def plot_latent_space_20x(label, slides, patterns, img_20x, latent_20x, oriind_20x, orig_set, cluster_labels_20x, institutions, orig_part, dataset, directories, fold_path, img_size, max_per_slide, num_folds, h5_file_name, display=False):
	histograms_img_mil_path, latent_paths, wsi_paths, miss_wsi_paths = directories
	train_img_20x,  valid_img_20x,  test_img_20x = img_20x

	ths_otherX = [0.9, 0.8, 0.7, 0.6] 
	ths_otherX = [0.9, 0.8] 
	for th_otherX in ths_otherX:

		weight_threshold = th_otherX
		print('\t\tWeight Threshold', weight_threshold)
		fold_top_weighted = [list(), list(), list(), list(), list(), list()]

		for fold_number in range(num_folds):
			print('\t\t\tFold', fold_number)

			# Construct H5 for fold.
			hdf5_path_weights_comb  = '%s/fold_%s/results/%s' % (fold_path, fold_number, h5_file_name)

			# Gather fold weights, predictions, labels, and others.
			weights_20x, weights_10x, weights_5x, probs, slides_metrics, train_slides, valid_slides, test_slides = gather_attention_results(hdf5_path_weights_comb)

			# Weights distribution per patient.
			otw = gather_latent_img_label_other(label, weight_threshold, test_slides, '20x', scatter_size=5, probs=probs, slides=slides, patterns=patterns, orig_set=orig_set, weights_5x=weights_5x, \
												weights_otherx=weights_20x, latent_otherx=latent_20x, train_img_otherx=train_img_20x, valid_img_otherx=valid_img_20x, test_img_otherx=test_img_20x, \
												oriind_otherx=oriind_20x, cluster_labels_otherx=cluster_labels_20x, inst_labels=institutions[:,0], max_per_slide=max_per_slide, \
												img_size=img_size, display=display, grab_images=True)

			# Append top weighted information
			fold_top_weighted = append_top_weighted(otw, fold_top_weighted, fold_number, num_folds)

		print('\t\tNumber of tiles:', fold_top_weighted[1].shape, 'Max Tiles per Slide:', max_per_slide)	
		top_tiles(label, fold_top_weighted[0], fold_top_weighted[1], fold_top_weighted[2], dataset, weight_threshold, latent_paths[2], magnification_name='20x', scatter_size=2, \
				  cls_top_tiles_weighted=fold_top_weighted[3], ins_top_tiles_weighted=fold_top_weighted[4])

		# Delete information from memory.
		del fold_top_weighted
	
