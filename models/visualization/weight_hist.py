import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os

from models.evaluation.latent_space import *
from models.utils import *
from models.visualization.utils import *


def show_weight_distributions(test_slides, weights_5x, weights_10x, weights_20x, slides, patterns, inst_labels, orig_set, fold_number, label, hist_img_mil_path, show_distribution=True):
    for slide, prob, class_ in test_slides:
        # Slide indices
        slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
        
        site = inst_labels[slide_indices[0]]

        # Slide weights 5x.
        slide_weights_5x       = weights_5x[slide_indices,0]
        slide_weights_5x_norm  = (slide_weights_5x - np.min(slide_weights_5x))/(np.max(slide_weights_5x)-np.min(slide_weights_5x))

        # Slide weights 10x.
        slide_weights_10x      = weights_10x[slide_indices, :, 0]*np.reshape(weights_5x[slide_indices,0], (-1, 1))
        slide_weights_10x      = np.reshape(slide_weights_10x, (-1,1))
        slide_weights_10x_norm = (slide_weights_10x - np.min(slide_weights_10x))/(np.max(slide_weights_10x)-np.min(slide_weights_10x))

        # Slide weights 20x.    
        slide_weights_20x      = weights_20x[slide_indices, :, 0]*np.reshape(weights_5x[slide_indices,0], (-1, 1))
        slide_weights_20x      = np.reshape(slide_weights_20x, (-1,1))
        slide_weights_20x_norm = (slide_weights_20x - np.min(slide_weights_20x))/(np.max(slide_weights_20x)-np.min(slide_weights_20x))

        fig, axes = plt.subplots(figsize=(50,15), ncols=1, nrows=3)
        for i, values in enumerate([(slide_weights_5x_norm, '5x'), (slide_weights_10x_norm, '10x'), (slide_weights_20x_norm, '20x')]):
            weights_slide, magnification = values
            axes[i].set_title(magnification, fontsize=20)
            axes[i].hist(weights_slide, bins=200, log=True)
        plt.suptitle('%s - %s - %s - Predict %s - %s' % (slide, magnification, patterns[slide_indices[0]][0], np.round(prob,4), site), fontsize=24)
        plt.savefig(os.path.join(hist_img_mil_path, '%s_prob%s_class%s.jpg' % (slide, np.round(prob,4), class_)))
        if show_distribution:
            plt.show()
        plt.close(fig)


def get_weight_distributions(slides, patterns, institutions, orig_set, label, fold_path, directories, num_folds, h5_file_name):
    histograms_img_mil_path, latent_paths, wsi_paths, miss_wsi_paths = directories
    for fold_number in range(num_folds): 
        print('\tFold', fold_number)
        
        hdf5_path_weights_comb  = '%s/fold_%s/results/%s' % (fold_path, fold_number, h5_file_name)

        ### Attention runs
        weights_20x, weights_10x, weights_5x, probs, slides_metrics, train_slides, valid_slides, test_slides = gather_attention_results(hdf5_path_weights_comb)

        top_slides, wrt_slides = pull_top_missclassified(test_slides, slides, slides_metrics, probs, patterns, label, top_percent=0.10)

        top_slides = top_slides[:10]
        wrt_slides = wrt_slides[-10:]

        ### Histogram of weight distributions
        show_weight_distributions(top_slides, weights_5x, weights_10x, weights_20x, slides, patterns, institutions, orig_set, fold_number, label, histograms_img_mil_path, show_distribution=False)
        show_weight_distributions(wrt_slides, weights_5x, weights_10x, weights_20x, slides, patterns, institutions, orig_set, fold_number, label, histograms_img_mil_path, show_distribution=False)


