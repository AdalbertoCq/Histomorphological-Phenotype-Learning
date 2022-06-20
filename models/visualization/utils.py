from sklearn.metrics import *
import numpy as np
import h5py
import os 

from models.utils import *


def setup_esq_folder(hdf5_projections_path, folder_type, att_model, attention_run, label):
    img_path             = hdf5_projections_path.split('hdf5_')[0]
    img_path             = os.path.join(img_path, 'Representations')
    img_path             = os.path.join(img_path, folder_type)
    att_img_path         = os.path.join(img_path, '%s_%s' % (att_model, attention_run))
    img_path             = os.path.join(att_img_path, label) 
    histograms_img_mil_path   = os.path.join(att_img_path, 'weight_histograms')

    latents_5x_img_mil_path   = os.path.join(att_img_path, 'latents_5x')
    latents_10x_img_mil_path  = os.path.join(att_img_path, 'latents_10x')
    latents_20x_img_mil_path  = os.path.join(att_img_path, 'latents_20x')

    slides_5x_img_mil_path   = os.path.join(att_img_path, 'WSI_5x')
    slides_10x_img_mil_path  = os.path.join(att_img_path, 'WSI_10x')
    slides_20x_img_mil_path  = os.path.join(att_img_path, 'WSI_20x')

    miss_slides_5x_img_mil_path  = os.path.join(att_img_path, 'WSI_5x_miss')
    miss_slides_10x_img_mil_path = os.path.join(att_img_path, 'WSI_10x_miss')
    miss_slides_20x_img_mil_path = os.path.join(att_img_path, 'WSI_20x_miss')

    paths = [histograms_img_mil_path, latents_5x_img_mil_path, latents_10x_img_mil_path, latents_20x_img_mil_path, \
             miss_slides_5x_img_mil_path, miss_slides_10x_img_mil_path, miss_slides_20x_img_mil_path, slides_5x_img_mil_path, slides_10x_img_mil_path, slides_20x_img_mil_path]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    return histograms_img_mil_path, [latents_5x_img_mil_path, latents_10x_img_mil_path, latents_20x_img_mil_path], \
           [slides_5x_img_mil_path, slides_10x_img_mil_path, slides_20x_img_mil_path], [miss_slides_5x_img_mil_path, miss_slides_10x_img_mil_path, miss_slides_20x_img_mil_path]


def gather_projection_var_multimag(hdf5_projections_path):
    all_output         = gather_content_multi_magnification(hdf5_projections_path, set_type='combined', h_latent=False)
    projection_content = h5py.File(hdf5_projections_path, mode='r')
    orig_set           = projection_content['combined_set_original']
    cluster_labels_20x = None
    cluster_labels_10x = None
    cluster_labels_5x  = None
    if 'cluster_labels_20x' in projection_content:
            cluster_labels_20x = projection_content['cluster_labels_20x']
            cluster_labels_10x = projection_content['cluster_labels_10x']
            cluster_labels_5x  = projection_content['cluster_labels_5x']

    all_output = list(all_output)
    all_output.extend([orig_set, cluster_labels_20x, cluster_labels_10x, cluster_labels_5x])
    return all_output


def read_images(hdf5_path_img, set_name):
    content = h5py.File(hdf5_path_img, mode='r')
    img     = content['%s_img' % set_name]
    sld     = content['%s_slides' % set_name]
    pat     = content['%s_patterns' % set_name]
    return img, sld, pat


def get_all_magnification_references(main_path, img_size, dataset_5x, dataset_10x, dataset_20x):
    # Test images magnifications.
    h5_test_mag             = []
    hdf5_path_img_test_5x   = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_test.h5' % (main_path, dataset_5x, img_size, img_size, dataset_5x)
    hdf5_path_img_test_10x  = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_test.h5' % (main_path, dataset_10x, img_size, img_size, dataset_10x)
    hdf5_path_img_test_20x  = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_test.h5' % (main_path, dataset_20x, img_size, img_size, dataset_20x)

    # Validation images magnifications.
    h5_valid_mag             = []
    hdf5_path_img_valid_5x   = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_validation.h5' % (main_path, dataset_5x, img_size, img_size, dataset_5x)
    hdf5_path_img_valid_10x  = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_validation.h5' % (main_path, dataset_10x, img_size, img_size, dataset_10x)
    hdf5_path_img_valid_20x  = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_validation.h5' % (main_path, dataset_20x, img_size, img_size, dataset_20x)

    # Train images magnifications.
    h5_train_mag            = []
    hdf5_path_img_train_5x  = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_train.h5' % (main_path, dataset_5x, img_size, img_size, dataset_5x)
    hdf5_path_img_train_10x = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_train.h5' % (main_path, dataset_10x, img_size, img_size, dataset_10x)
    hdf5_path_img_train_20x = '%s/datasets/%s/he/patches_h%s_w%s/hdf5_%s_he_train.h5' % (main_path, dataset_20x, img_size, img_size, dataset_20x)

    h5_test_mag             = [hdf5_path_img_test_5x, hdf5_path_img_test_10x, hdf5_path_img_test_20x]
    h5_valid_mag            = [hdf5_path_img_valid_5x, hdf5_path_img_valid_10x, hdf5_path_img_valid_20x]
    h5_train_mag            = [hdf5_path_img_train_5x, hdf5_path_img_train_10x, hdf5_path_img_train_20x]

    # Verify all files are there.
    flag = False 
    for h5_file in h5_test_mag + h5_valid_mag + h5_train_mag:
        if not os.path.isfile(h5_file):
            print('Image H5 file not found:', h5_file)
            flag = True
    if flag:
        print()
        print('Missing H5 files with images: Break #1 - Look at Dataset image variables x5/x10/x20. Files could be missing too.')
        exit()

    return h5_train_mag, h5_valid_mag, h5_test_mag


def gather_original_partition(train_sld, valid_sld, test_sld):
    def gather_slides(slides):
        unique_slides = np.unique(slides[:].astype(str)).tolist()
    #     unique_partic = ['-'.join(slide.split('-')[:3]) for slide in unique_slides]
        return unique_slides

    def gather_patients(slides):
        unique_slides = np.unique(slides[:].astype(str)).tolist()
        unique_partic = ['-'.join(slide.split('-')[:3]) for slide in unique_slides]
        return unique_partic

    train_sld_20x, train_sld_10x, train_sld_5x = train_sld
    valid_sld_20x, valid_sld_10x, valid_sld_5x = valid_sld
    test_sld_20x,  test_sld_10x,  test_sld_5x  = test_sld

    part_train_20x  = gather_slides(train_sld_20x)
    part_train_10x  = gather_slides(train_sld_10x)
    part_train_5x   = gather_slides(train_sld_5x)
    orig_part_train = list(set(part_train_20x + part_train_10x + part_train_5x))

    part_valid_20x  = gather_slides(valid_sld_20x)
    part_valid_10x  = gather_slides(valid_sld_10x)
    part_valid_5x   = gather_slides(valid_sld_5x)
    orig_part_valid = list(set(part_valid_20x + part_valid_10x + part_valid_5x))

    part_test_20x   = gather_slides(test_sld_20x)
    part_test_10x   = gather_slides(test_sld_10x)
    part_test_5x    = gather_slides(test_sld_5x)
    orig_part_test  = list(set(part_test_20x + part_test_10x + part_test_5x))

    orig_part = [orig_part_train, orig_part_valid, orig_part_test]

    print('Intersection Slides and sets:')
    print('Train/Valid:', set(orig_part_train).intersection(set(orig_part_valid)))
    print('Train/Test: ', set(orig_part_test).intersection(set(orig_part_valid)))
    print('Valid/Test: ', set(orig_part_train).intersection(set(orig_part_test)))

    return orig_part


def pull_top_missclassified(test_slides, slides, slides_metrics, probs, patterns, label, top_percent=0.1):
    all_slides = list()
    all_diff   = list()
    all_class  = list()
    all_probs  = list()
    all_preds  = list()

    for slide in test_slides:
        inds   = np.argwhere(slides_metrics==slide)[0,0]
        inds_p = np.argwhere(slides==slide)[0,0]
        if label in patterns[inds_p,0]:
            class_slide = 1
        else:
            class_slide = 0

        diff = np.abs(class_slide - probs[inds,1])
        all_slides.append(slide)
        all_diff.append(diff)
        all_class.append(class_slide)
        all_probs.append(probs[inds,1])
        all_preds.append(np.argmax(probs[inds]))

    all_slides = np.vstack(all_slides)
    all_diff   = np.vstack(all_diff)
    all_class  = np.vstack(all_class)
    all_probs  = np.vstack(all_probs)
    all_preds  = np.vstack(all_preds)

    inds         = np.argsort(all_diff[:,0])
    inds_match   = np.argwhere(all_diff[:,0]<0.5)
    inds_not     = np.argwhere(all_diff[:,0]>=0.5)
    inds_match   = np.intersect1d(inds_match, inds)
    inds_not     = np.intersect1d(inds_not, inds)

    top_nsamples = math.ceil(top_percent*all_diff.shape[0])
    wrt_ind      = inds_not
    top_ind      = inds_match[:top_nsamples]

    wrt_prob     = all_probs[wrt_ind,0]
    wrt_slides   = all_slides[wrt_ind,0]
    wrt_class    = all_class[wrt_ind,0]

    top_prob     = all_probs[top_ind,0]
    top_slides   = all_slides[top_ind,0]
    top_class    = all_class[top_ind,0]

    top_sl  = list(zip(top_slides, top_prob, top_class))
    wrt_sl  = list(zip(wrt_slides, wrt_prob, wrt_class))

    return top_sl, wrt_sl
