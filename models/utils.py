# Imports.
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np
import shutil
import random
import h5py
import json
import math
import csv
import os


# Simple function to plot number images.
def plot_images(plt_num, images, dim1=None, dim2=None, wspace=None, title=None, axis='off', plt_save=None):
    # Standard parameters for the plot.
    if dim1 is not None and dim2 is not None:
        fig = plt.figure(figsize=(dim1, dim2))
    else:
        fig = plt.figure()
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace)
    if title is not None:
        fig.suptitle(title)
    for i in range(0, plt_num):
        fig.add_subplot(1, 10, i+1)
        img = images[i, :, :, :]
        plt.imshow(img)
        plt.axis(axis)
    if plt_save is not None:
        plt.savefig(plt_save)
    plt.show()


# Plot and save figure of losses.
def save_loss(losses, data_out_path, dim):    
    mpl.rcParams["figure.figsize"] = dim, dim
    plt.rcParams.update({'font.size': 22})
    losses = np.array(losses)
    num_loss = losses.shape[1]
    for _ in range(num_loss):
        if _ == 0:
            label = 'Generator'
        elif _ == 1:
            label = 'Discriminator'
        else:
            label = 'Mutual Information'
        plt.plot(losses[:, _], label=label, alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('%s/training_loss.png' % data_out_path)


def get_checkpoint(data_out_path, which=0):
    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, 'checkpoint')
    index = 0
    with open(checkpoints, 'r') as f:
        for line in reversed(f.readlines()):
            if index == which:
                return line.split('"')[1]
    print('No model to restore')
    exit()


def update_csv(model, file, variables, epoch, iteration, losses):
    with open(file, 'a') as csv_file:
        if 'loss' in file: 
            header = ['Epoch', 'Iteration']
            header.extend(losses)
            writer = csv.DictWriter(csv_file, fieldnames = header)
            line = dict()
            line['Epoch'] = epoch
            line['Iteration'] = iteration
            for ind, val in enumerate(losses):
                line[val] = variables[ind]
        elif 'filter' in file:
            header = ['Epoch', 'Iteration']
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.gen_filters])
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.dis_filters])
            writer = csv.DictWriter(csv_file, fieldnames = header)
            line = dict()
            line['Epoch'] = epoch
            line['Iteration'] = iteration
            for var in variables[0]:
                line[var] = variables[0][var]
            for var in variables[1]:
                line[var] = variables[1][var]
        elif 'jacobian' in file:
            writer = csv.writer(csv_file)
            line = [epoch, iteration]
            line.extend(variables)
        elif 'hessian' in file:
            writer = csv.writer(csv_file)
            line = [epoch, iteration]
            line.extend(variables)
        writer.writerow(line)


def setup_csvs(csvs, model, losses, restore=False):
    loss_csv = csvs[0]
    if not restore:
        header = ['Epoch', 'Iteration']
        header.extend(losses)
        print(loss_csv)
        with open(loss_csv, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()

        if len(csvs) > 1: 

            filters_s_csv, jacob_s_csv, hessian_s_csv = csvs[1:]

            header = ['Epoch', 'Iteration']
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.gen_filters])
            header.extend([str(v.name.split(':')[0].replace('/', '_')) for v in model.dis_filters])
            with open(filters_s_csv, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=header)
                writer.writeheader()

            header = ['Epoch', 'Iteration', 'Jacobian Max Singular', 'Jacobian Min Singular']
            with open(jacob_s_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)

            header = ['Epoch', 'Iteration']
            with open(hessian_s_csv, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
            

# Setup output folder.
def setup_output(data_out_path, model_name, restore, additional_loss=False):
    os.umask(0o002)
    results_path = os.path.join(data_out_path, 'results')
    checkpoints_path = os.path.join(data_out_path, 'checkpoints')
    checkpoints = os.path.join(checkpoints_path, '%s.ckt' % model_name)
    gen_images_path = os.path.join(data_out_path, 'images')
    
    if not restore:
        if os.path.isdir(checkpoints_path):
            shutil.rmtree(checkpoints_path)
        os.makedirs(checkpoints_path)
        if os.path.isdir(gen_images_path):
            shutil.rmtree(gen_images_path)
        os.makedirs(gen_images_path)
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)
    
    loss_csv = os.path.join(data_out_path, 'loss.csv')
    if additional_loss:
        loss_csv_2 = os.path.join(data_out_path, 'loss_add.csv')
        return checkpoints, [loss_csv, loss_csv_2]

    return checkpoints, [loss_csv]

# Run session to generate output samples.
def show_generated(session, z_input, z_dim, output_fake, n_images, label_input=None, labels=None, c_input=None, c_dim=None, dim=20, show=True):
    gen_samples = list()
    sample_z = list()
    batch_sample = 20
    for x in range(n_images):
        rand_sample = random.randint(0,batch_sample-1)
        
        z_batch = np.random.uniform(low=-1., high=1., size=(batch_sample, z_dim))
        feed_dict = {z_input:z_batch}
        if c_input is not None:
            c_batch = np.random.normal(loc=0.0, scale=1.0, size=(batch_sample, c_dim))
            feed_dict[c_input] = c_batch
        elif label_input is not None:
            feed_dict[label_input] = labels[:batch_sample, :]
        gen_batch = session.run(output_fake, feed_dict=feed_dict)
        gen_samples.append(gen_batch[rand_sample, :, :, :])
        sample_z.append(z_batch[rand_sample, :])
    if show:
        plot_images(plt_num=n_images, images=np.array(gen_samples), dim=dim)    
    return np.array(gen_samples), np.array(sample_z)


# Method to report parameter in the run.
def report_parameters(model, epochs, restore, data_out_path):
    with open('%s/run_parameters.txt' % data_out_path, 'w') as f:
        f.write('Epochs: %s\n' % (epochs))
        f.write('Restore: %s\n' % (restore))
        for attr, value in model.__dict__.items():
            f.write('%s: %s\n' % (attr, value))


def gather_filters():
    gen_filters = list()
    dis_filters = list()
    for v in tf.trainable_variables():
        if 'filter' in v.name:
            if 'generator' in v.name:
                gen_filters.append(v)
            elif 'discriminator' in v.name:
                dis_filters.append(v)
            elif 'encoder' in v.name:
                dis_filters.append(v)
            else:
                print('No contemplated filter: ', v.name)
                print('Review gather_filters()')
    return gen_filters, dis_filters


def retrieve_csv_data(csv_file, limit_head=2, limit_row=None, sing=0):
    dictionary = dict()
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for field in reader.fieldnames:
            dictionary[field] = list()
        ind = 0
        for row in reader:
            ind += 1
            if ind < limit_head:
                continue
            elif limit_row is not None and ind >= limit_row:
                break
            for field in reader.fieldnames:
                value = row[field].replace('[', '')
                value = value.replace(']', '')
                if ' ' in value and 'j' in value:
                    value = value.replace('j ', 'j_')
                    value = value.replace(' ', '')
                    value = value.replace('j_', 'j ')
                    value = [complex(val).real for val in value.split(' ')]
                    if sing is None:
                        value = value[0]/value[1]
                    else:
                        value = value[sing]
                elif 'j' in value:
                    value = complex(value)
                    if value.imag > 1e-4:
                        print('[Warning] Imaginary part of singular value larget than 1e-4:', value)
                    value = value.real
                    if value == 0.0:
                        print('[Warning] Min Singular Value Jacobian: [0.0] ', json.dumps(row))
                        value = float(1e-3)
                elif value == '':
                    print('[Warning] Min Singular Value Jacobian: [None]', json.dumps(row))
                    value = float(1e-3)
                else:
                    value = float(value)
                dictionary[field].append(value)

    if 'jacobian' in  csv_file:
        dictionary['Ratio Max/Min'] = list()
        for p in [i for i in range(len(dictionary['Iteration']))]:
            dictionary['Ratio Max/Min'].append(np.log(dictionary['Jacobian Max Singular'][p]/dictionary['Jacobian Min Singular'][p]))

    return dictionary


def plot_data(data1, data2=None, filter1=[], filter2=[], dim=20, total_axis=20, same=False):
    mpl.rcParams['figure.figsize'] = dim, dim
    exclude_b = ['Epoch', 'Iteration']
    fig, ax1 = plt.subplots()
    points = [i for i in range(len(data1['data']['Iteration']))]

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]
    random.shuffle(colors)
    ind = 0

    # First data plot
    exclude1 = list()
    exclude1.extend(exclude_b)
    exclude1.extend(filter1)
    ax1.set_xlabel('Iterations (Batch size)')
    ax1.set_ylabel(data1['name'])
    # ax1.set_color_cycle(['red', 'black', 'yellow'])
    for field in data1['data']:
        flag = False
        for exclude in exclude1:
            if exclude in field:
                flag=True
                break
        if flag: continue
        ax1.plot(points, data1['data'][field], label='%s %s' %(data1['name'].split(' ')[1],field), color=colors[ind])
        ind += 1

    every = int(len(points)/total_axis)
    if every == 0: every =1
    plt.xticks(points[0::every], data1['data']['Iteration'][0::every], rotation=45)
    plt.legend(loc='upper left')

    if data2 is not None:
        # Second data plot
        exclude2 = list()
        exclude2.extend(exclude_b)
        exclude2.extend(filter2)
        if not same:   
            ax2 = ax1.twinx()  
            ax2.set_ylabel(data2['name']) 
            plot = ax2
        else:
            plot = ax1
        for field in data2['data']:
            flag = False
            for exclude in exclude2:
                if exclude in field:
                    flag=True
                    break
            if flag: continue
            plot.plot(points, data2['data'][field], label='%s %s' %(data2['name'].split(' ')[1],field), color=colors[ind])
            ind += 1
        plt.xticks(points[0::every], data2['data']['Iteration'][0::every], rotation=45)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='upper right')
    plt.show()

def display_activations(layer_activations, image, images_row, dim=None):
    if dim is not None:
        import matplotlib as mpl
        mpl.rcParams['figure.figsize'] = dim, dim
    num_channels = layer_activations.shape[-1]
    img_width = layer_activations.shape[2]
    img_height = layer_activations.shape[1]
    rows = math.ceil(num_channels/images_row)
    grid = np.zeros((img_height*rows, img_width*images_row))
    
    print('Number of Channels:', num_channels)
    print('Number of Rows:', rows)
    for channel in range(num_channels):
        channel_image = layer_activations[image, :, :, channel]
        channel_image -= channel_image.mean() 
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        grid_row = int(channel/images_row)
        grid_col = channel%images_row
        grid[grid_row*img_height : grid_row*img_height + img_height, grid_col*img_width: grid_col*img_width + img_width] = channel_image

    scale = 1. / num_channels
    plt.figure(figsize=(scale * grid.shape[1], scale * grid.shape[0]))
    plt.matshow(grid)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


### Multiple Instance Learning.

def save_fold_performance(data_out_path, fold_losses, folds_metrics, file_name):
    file_path = os.path.join(data_out_path, file_name)

    print('folds_metrics', len(folds_metrics))

    with open(file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fold_losses)
        writer.writeheader()

        # Keep track of train, valid, and test Accuracy and AUC for mean and std calculation
        metrics_ms = np.zeros((len(folds_metrics), 6))
        for i, fold_metrics in enumerate(folds_metrics):
            line = dict()
            line['Fold'] = i
            train_metrics, valid_metrics, test_metrics = fold_metrics
            train_accuracy, train_recall, train_precision, train_auc, train_class_set, train_pred_set, train_prob_set, _, top_w_train = train_metrics
            train_accuracy, train_recall, train_precision, train_auc, _, _, _, _, top_w_train = train_metrics
            valid_accuracy, valid_recall, valid_precision, valid_auc, _, _, _, _, top_w_valid = valid_metrics
            test_accuracy,  test_recall,  test_precision,  test_auc,  _, _, _, _, top_w_test = test_metrics
            metrics = [train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc, train_recall, valid_recall, test_recall, train_precision, valid_precision, test_precision]
            for ind, val in enumerate(fold_losses[1:]):
                line[val] = metrics[ind]
                if ind < 6:
                    metrics_ms[i,ind] = metrics[ind][0]
            writer.writerow(line)

        line = dict()
        line['Fold'] = 'Mean'
        means = np.round(np.mean(metrics_ms, axis=0),3)
        for ind, val in enumerate(fold_losses[1:]):
            if ind < 6:
                line[val] = means[ind]
            else:
                line[val] = ''
        writer.writerow(line)

        line = dict()
        line['Fold'] = 'Std'
        std = np.round(np.std(metrics_ms, axis=0),3)
        for ind, val in enumerate(fold_losses[1:]):
            if ind < 6:
                line[val] = std[ind]
            else:
                line[val] = ''
        writer.writerow(line)
    
def save_fold_performance_survival(data_out_path, fold_losses, folds_metrics, file_name):
    file_path = os.path.join(data_out_path, file_name)

    print('folds_metrics', len(folds_metrics))
    fold_losses = ['Epochs', 'Iteration'] + fold_losses
    
    with open(file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Fold'] + fold_losses)
        writer.writeheader()
        
        train_cth_index = list()
        test_cth_index  = list()
        # Keep track of train, valid, and test Accuracy and AUC for mean and std calculation
        metrics_ms = np.zeros((len(folds_metrics), 6))
        for i, fold_metrics in enumerate(folds_metrics):
            line = dict()
            line['Fold'] = i
            for j, loss in enumerate(fold_losses):
                if j < 2:
                    value = int(folds_metrics[i][j])
                else:
                    value = np.round(folds_metrics[i][j],3)
                line[loss] = value
                if j == 5: train_cth_index.append(value)
                elif j == 9: test_cth_index.append(value)
            writer.writerow(line)

        line = dict()
        line['Fold'] = 'Mean'
        for j, loss in enumerate(fold_losses):
            if j == 5: 
                line[loss] = np.round(np.mean(train_cth_index),3)
            elif j == 9: 
                line[loss] = np.round(np.mean(test_cth_index),3)
            else:
                line[loss] = ''
        writer.writerow(line)
        
        
        line = dict()
        line['Fold'] = 'Std'
        for j, loss in enumerate(fold_losses):
            if j == 5: 
                line[loss] = np.round(np.std(train_cth_index),3)
            elif j == 9: 
                line[loss] = np.round(np.std(test_cth_index),3)
            else:
                line[loss] = ''
        writer.writerow(line)
    
def save_unique_samples(data_out_path, train_class_set, valid_class_set, test_class_set, file_name):
    file_path = os.path.join(data_out_path, file_name)
    with open(file_path, 'w') as content:
        for name, class_set in ('Train:', train_class_set), ('Valid:', valid_class_set), ('Test: ', test_class_set):
            content.write('%s \n' % name)
            uniq, counts = np.unique(class_set, return_counts=True)
            content.write('\tLabels: %s \n' % str(uniq))
            content.write('\tCounts: %s \n' % str(counts))

# Save relevant tiles for the outcome, allows to visualize important tiles given attention.
def save_relevant(relevant, output_path, set_type):
    relevant_patches, relevant_labels, relevant_indeces, relevant_slides, relevant_weights = relevant
    dt = h5py.special_dtype(vlen=str)
    hdf5_path = os.path.join(output_path, 'hdf5_relevant_tiles_%s.h5' % set_type)
    with h5py.File(hdf5_path, mode='w') as hdf5_content:   
        latent_storage = hdf5_content.create_dataset(name='latent', shape=relevant_patches.shape,     dtype=np.float32)
        label_storage  = hdf5_content.create_dataset(name='label',  shape=relevant_labels.shape,      dtype=np.float32)
        ind_storage    = hdf5_content.create_dataset(name='indece', shape=relevant_indeces.shape,     dtype=np.float32)
        slide_storage  = hdf5_content.create_dataset(name='slide',  shape=(len(relevant_slides), 1),  dtype=dt)
        weight_storage = hdf5_content.create_dataset(name='weight', shape=relevant_weights.shape,     dtype=np.float32)

        for i in range(relevant_patches.shape[0]):
            latent_storage[i, :] = relevant_patches[i, :]
            label_storage[i]     = relevant_labels[i]
            ind_storage[i]       = relevant_indeces[i]
            slide_storage[i]     = relevant_slides[i]
            weight_storage[i]    = relevant_weights[i]

# Gathers content of H5 files with latent representations.
def gather_content(hdf5_path, set_type, h_latent=True):
    # Open file for data manipulation. 
    hdf5_content = h5py.File(hdf5_path, mode='r')
    if '%s_img_w_latent' % set_type in list(hdf5_content.keys()):
        latent   = hdf5_content['%s_img_w_latent' % set_type]
    else:
        if h_latent:
            latent   = hdf5_content['%s_img_h_latent' % set_type]
        else:
            latent   = hdf5_content['%s_img_z_latent' % set_type]
    patterns = np.array(hdf5_content['%s_patterns' % set_type]).astype(str)
    slides   = np.array(hdf5_content['%s_slides' % set_type]).astype(str)
    tiles    = hdf5_content['%s_tiles' % set_type]

    labels = None
    if '%s_labels' % set_type in hdf5_content.keys():
        labels   = hdf5_content['%s_labels' % set_type]

    if '_histological_subtypes' in hdf5_path:
        patterns = np.array(hdf5_content['combined_hist_subtype']).astype(str)

    institutions = None
    if 'combined_institutions' in hdf5_content.keys():
        institutions = np.array(hdf5_content['combined_institutions']).astype(str)
    return latent, labels, patterns, slides, tiles, institutions


def gather_content_multi_magnification(hdf5_path, set_type=None, h_latent=True):
    # Open file for data manipulation. 
    hdf5_content     = h5py.File(hdf5_path, mode='r')
    if set_type is None:
        for key in hdf5_content.keys():
            if '20x_img_z_latent' in key:
                set_type = key.split('_')[0]

    # Latents
    latent_20x       = hdf5_content['%s_20x_img_z_latent' % set_type]
    latent_10x       = hdf5_content['%s_10x_img_z_latent' % set_type]
    latent_5x        = hdf5_content['%s_5x_img_z_latent'  % set_type]
    # Indices of the original datasets.
    orig_indices_20x = hdf5_content['%s_20x_orig_indices' % set_type]
    orig_indices_10x = hdf5_content['%s_10x_orig_indices' % set_type]
    orig_indices_5x  = hdf5_content['%s_5x_orig_indices'  % set_type]
    # Patterns.
    if '%s_pattern' % set_type in hdf5_content.keys():
        patterns = np.array(hdf5_content['%s_pattern' % set_type]).astype(str)
    else:
        patterns = np.array(hdf5_content['%s_patterns' % set_type]).astype(str)
    # Slides.
    slides   = np.array(hdf5_content['%s_slides' % set_type]).astype(str)
    # Tiles.
    tiles    = hdf5_content['%s_tiles' % set_type]

    if '%s_survival' % set_type in hdf5_content.keys():
        patterns = np.array(hdf5_content['%s_survival' % set_type])
    elif 'combined_hist_subtype' in hdf5_content.keys():
        patterns = np.array(hdf5_content['combined_hist_subtype']).astype(str)

    institutions = None
    if 'combined_institutions' in hdf5_content.keys():
        institutions = hdf5_content['combined_institutions']

    return latent_20x, latent_10x, latent_5x, orig_indices_20x, orig_indices_10x, orig_indices_5x, patterns, slides, tiles, institutions


def gather_attention_results_multimag(hdf5_path_weights_comb):
    content = h5py.File(hdf5_path_weights_comb, mode='r')
    weights_20x = content['weights_20x']
    weights_10x = content['weights_10x']
    weights_5x  = content['weights_5x']
    probs       = content['probabilities']
    labels      = content['labels']
    fold_set    = content['fold_set'][:].astype(str)
    slides_m    = content['slides_metric'][:].astype(str)
    
    train_ind   = np.argwhere(fold_set=='train')[:,0]
    valid_ind   = np.argwhere(fold_set=='valid')[:,0]
    test_ind    = np.argwhere(fold_set=='test')[:,0]

    train_slides = np.unique(slides_m[train_ind,0])
    test_slides  = np.unique(slides_m[test_ind,0])
    
    if valid_ind.shape[0] == 0:
        valid_slides = None
    else:
        valid_slides = np.unique(slides_m[valid_ind,0])

    # Workaround: Real fix should be in save_weights
    filtered_test = list()
    for slide in test_slides:
        inds = np.argwhere(slides_m[:]==slide)
        if inds.shape[0] < 100:
            continue
        filtered_test.append(slide)

    return weights_20x, weights_10x, weights_5x, probs, slides_m, train_slides, valid_slides, filtered_test


def gather_attention_results_indmag(hdf5_path_weights_comb):
    content = h5py.File(hdf5_path_weights_comb, mode='r')
    weights     = content['weights']
    probs       = content['probabilities']
    labels      = content['labels']
    fold_set    = content['fold_set'][:].astype(str)
    slides_m    = content['slides_metric'][:].astype(str)
    
    train_ind   = np.argwhere(fold_set=='train')[:,0]
    valid_ind   = np.argwhere(fold_set=='valid')[:,0]
    test_ind    = np.argwhere(fold_set=='test')[:,0]

    train_slides = np.unique(slides_m[train_ind,0])
    test_slides  = np.unique(slides_m[test_ind,0])
    
    if valid_ind.shape[0] == 0:
        valid_slides = None
    else:
        valid_slides = np.unique(slides_m[valid_ind,0])

    # Workaround: Real fix should be in save_weights
    filtered_test = list()
    for slide in test_slides:
        inds = np.argwhere(slides_m[:]==slide)
        if inds.shape[0] < 100:
            continue
        filtered_test.append(slide)

    return weights, probs, slides_m, train_slides, valid_slides, filtered_test
