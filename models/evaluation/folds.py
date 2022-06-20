# Import packages
import numpy as np
import random
import os

# Own libs
from data_manipulation.utils import load_data


def load_existing_split(pickle_file):
    folds = load_data(pickle_file)
    slide_folds = list()
    for fold in folds:
        train_i = folds[fold]['train']
        valid_i = folds[fold]['valid']
        test_i  = folds[fold]['test']
        train_slides = [set_[0] for set_ in train_i if set_[0] is not None]
        valid_slides = [set_[0] for set_ in valid_i if set_[0] is not None]
        test_slides  = [set_[0] for set_ in test_i   if set_[0] is not None]
        slide_folds.append((train_slides, valid_slides, test_slides))
    return slide_folds

def split_folds(model, total_slides, total_patterns, num_folds=5, val_split=False, file_path=''):
    # If split already exist, pull it.
    if os.path.isfile(file_path):
        print('Loading existing fold cross-validation:', file_path)
        folds = load_existing_split(file_path)
        return folds

    all_slides      = np.unique(total_slides)
    random.shuffle(all_slides)
    positive_slides = list()
    negative_slides = list()
    for slide in all_slides:
        indxs = np.argwhere(total_slides[:]==slide)[:,0]
        # Get label str and transform into int.
        label_instances = total_patterns[indxs[0]]
        label_batch = model.process_label(label_instances)
        if label_batch == 1:
            positive_slides.append(slide)
        else:
            negative_slides.append(slide)

    perct = 1.0/num_folds
    positive_test_size  = int(len(positive_slides)*perct)
    negative_test_size  = int(len(negative_slides)*perct)
    positive_rem        = len(positive_slides)-positive_test_size
    negative_rem        = len(negative_slides)-negative_test_size
    positive_valid_size = int(positive_rem*perct)
    negative_valid_size = int(negative_rem*perct)
    positive_set = set(positive_slides)
    negative_set = set(negative_slides)

    i = 0
    pos_folds = list()
    for i in range(num_folds):
        if i == num_folds-1:
            i_pos_test_folds  = positive_slides[i*positive_test_size:]
            i_pos_valid_folds = []
            if val_split:
                i_pos_valid_folds = positive_slides[i*positive_test_size-positive_valid_size:i*positive_test_size]
        else:
            i_pos_test_folds  = positive_slides[i*positive_test_size:i*positive_test_size+positive_test_size]
            i_pos_valid_folds = []
            if val_split:
                i_pos_valid_folds = positive_slides[i*positive_test_size+positive_test_size:i*positive_test_size+positive_test_size+positive_valid_size]
        pos_folds.append((i_pos_test_folds, i_pos_valid_folds))

    i = 0
    neg_folds = list()
    for i in range(num_folds):
        if i == num_folds-1:
            i_neg_test_folds  = negative_slides[i*negative_test_size:]
            i_neg_valid_folds = []
            if val_split:
                i_neg_valid_folds = negative_slides[i*negative_test_size-negative_valid_size:i*negative_test_size]
        else:
            i_neg_test_folds  = negative_slides[i*negative_test_size:i*negative_test_size+negative_test_size]
            i_neg_valid_folds = []
            if val_split:
                i_neg_valid_folds = negative_slides[i*negative_test_size+negative_test_size:i*negative_test_size+negative_test_size+negative_valid_size]
        neg_folds.append((i_neg_test_folds, i_neg_valid_folds))

    folds = list()
    for i, pos_sets in enumerate(pos_folds):
        i_pos_test_folds, i_pos_valid_folds = pos_sets
        i_neg_test_folds, i_neg_valid_folds = neg_folds[i]
        train_set = list()
        valid_set = list()
        test_set  = list()
        i_pos_train_set = list(positive_set.difference(set(i_pos_test_folds+i_pos_valid_folds)))
        i_neg_train_set = list(negative_set.difference(set(i_neg_test_folds+i_neg_valid_folds)))
        train_set.extend(i_pos_train_set)
        train_set.extend(i_neg_train_set)
        valid_set.extend(i_pos_valid_folds)
        valid_set.extend(i_neg_valid_folds)
        test_set.extend(i_pos_test_folds)
        test_set.extend(i_neg_test_folds)

        folds.append((np.array(train_set), np.array(valid_set), np.array(test_set)))

    return folds

# Get positive samples per institutions.
def get_counts_institutions(model, slides, patterns, institutions):
    inst_pos = dict()
    all_pos = 0
    for institution in np.unique(institutions):
        inst_indices = np.argwhere(institutions[:]==institution)[:,0]
        labels = list()
        for slide in np.unique(slides[inst_indices]):
            slide_ind = np.argwhere(slides==slide)[0,0]
            if len(patterns.shape) == 2:
                pattern_slide = patterns[slide_ind]
            else:
                pattern_slide = patterns[slide_ind]
            label     = model.process_label(pattern_slide)
            labels.append(label)
        all_pos += np.sum(labels)
        inst_pos[institution] = (np.sum(labels), len(labels))
    return inst_pos, all_pos


# Get a fold that is somewhat balanced in the number of positive samples, no-overlapping institutions.
def get_reference_fold_institutions(institutions, inst_pos, num_folds, all_pos, random_shuffles=20):
	# Number of positive per test set.
    split_count = (1/(num_folds+1))*all_pos

    fold_ref = None
    diff_ref = np.inf
    for i in range(random_shuffles):
        institutions_uniq = list(np.unique(institutions))
        random.shuffle(institutions_uniq)
        acc_split = 0
        all_split = 0

        fold_split = list()
        split_ins = list()
        for institution in institutions_uniq:
            acc_split += inst_pos[institution][0]
            all_split += inst_pos[institution][1]
            split_ins.append(institution)
            if acc_split > split_count:
                fold_split.append((split_ins, acc_split, all_split-acc_split))
                acc_split = 0
                all_split = 0
                split_ins = list()
        fold_split.append((split_ins, acc_split, all_split-acc_split))
        if len(fold_split) != num_folds:
        	continue
        if diff_ref > abs(split_count-acc_split):
            fold_ref = fold_split
            diff_ref = split_count-acc_split

    return fold_ref

def get_train_val_institutions(model, train_institutions, institutions, slides, patterns, random_shuffles=20):
    all_pos = 0
    all_inst = list()
    for train_inst in train_institutions:
        inst_pos    = 0
        inst_ind    = np.argwhere(institutions[:]==train_inst)[:,0]
        if len(slides.shape) == 2:
            slide_i = slides[inst_ind, 0]
        else:
            slide_i = slides[inst_ind]
        inst_slides = np.unique(slide_i)
        for slide in inst_slides:
            ind_slide = np.argwhere(slides[:]==slide)[:,0]
            if len(patterns.shape) == 2:
                pattern_i = patterns[ind_slide[0], 0]
            else:
                pattern_i = patterns[ind_slide[0]]
            label     = model.process_label(pattern_i)
            if label:
                inst_pos += 1
        all_pos += inst_pos
        all_inst.append((train_inst, inst_pos))

    val_pos = int(all_pos*0.2)
    train_institutions = list(train_institutions)

    ref_fold = list()
    min_dist = np.inf
    for i in range(random_shuffles):
        val_fold = list()
        acc_pos = 0
        for inst, pos_numb in all_inst:
            acc_pos += pos_numb
            val_fold.append(inst)
            if acc_pos > val_pos:
                break
        if abs(acc_pos - val_pos) < min_dist:
            ref_fold = val_fold

    train_institutions = list(set(train_institutions).difference(set(ref_fold)))
    return train_institutions, ref_fold


def get_slides(set_institutions, institutions, slides):
    set_slides = list()
    for institution in set_institutions:
        inst_ind = np.argwhere(institutions[:]==institution)[:,0]
        if len(slides.shape) == 2:
            slide_i = slides[inst_ind, 0]
        else:
            slide_i = slides[inst_ind]
        set_slides.extend(np.unique(slide_i))
    return set_slides


def get_final_split_institutions(model, fold_ref, institutions, slides, patterns, val_split=False):
    folds = list()
    uniq_institutions  = np.unique(institutions)
    for test_split_inst, _, _ in fold_ref:
        train_institutions = set(uniq_institutions).difference(set(test_split_inst))
        if val_split:
            train_institutions, valid_institutions = get_train_val_institutions(model, train_institutions, institutions, slides, patterns, random_shuffles=20)
        train_slides = get_slides(train_institutions, institutions, slides)
        test_slides  = get_slides(test_split_inst, institutions, slides)
        if val_split:
            valid_slides = get_slides(valid_institutions, institutions, slides)
            folds.append((train_slides, valid_slides, test_slides))
        else:
            folds.append((train_slides, [], test_slides))
    return folds


def split_folds_institutions(model, slides, patterns, institutions, val_split=False, num_folds=5, random_shuffles=20, file_path=''):

    if os.path.isfile(file_path):
        print('Loading existing fold cross-validation:', file_path)
        folds = load_existing_split(file_path)

    else:
        # Get number of positive samples per institution.
        inst_pos, all_pos = get_counts_institutions(model, slides, patterns, institutions)

        # Get a fold that is somewhat balanced in the number of positive samples, no-overlapping institutions.
        fold_ref = get_reference_fold_institutions(institutions, inst_pos, num_folds, all_pos, random_shuffles=random_shuffles)

        # Construct folds
        folds = get_final_split_institutions(model, fold_ref, institutions, slides, patterns, val_split=val_split)

    return folds
