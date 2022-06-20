import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import math
import copy
import csv
import sys
import glob
import os


def get_last_saved_epoch(data_out_path):
    generated_images = sorted(glob.glob(os.path.join(data_out_path, 'images/gen_samples_epoch_*.png')), key=os.path.getctime, reverse=True)
    oldest_epoch_image = generated_images[0].split('_')[-1].replace('.png', '')
    return int(oldest_epoch_image)

def save_image(img, job_id, name, train=True):
    import skimage.io
    if train:
        folder = 'img'
    else:
        folder = 'gen'
    if not os.path.isdir('run/%s/%s/' % (job_id, folder)):
        os.makedirs('run/%s/%s/' % (job_id, folder))
    skimage.io.imsave('run/%s/%s/%s.png' % (job_id, folder, name), img)


def store_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def load_csv(file_path):
    with open(file_path, 'r') as file:
        return list(csv.reader(file))


def filter_filenames(filenames, extension):
    return sorted(list(filter(lambda f: f.endswith(extension), filenames)))

def labels_to_binary(labels, n_bits=5, buckets=True):
    if buckets:
        lower = (labels<=5)*1
        upper = (labels>5)*2
        labels = lower + upper

    labels = labels.astype(int)
    batch_size, l_dim = labels.shape
    output_labels =  np.zeros((batch_size, n_bits))
    for b_num in range(batch_size):
        l = labels[b_num, 0]
        binary_l = '{0:b}'.format(l)
        binary_l = list(binary_l)
        binary_l = list(map(int, binary_l))
        n_rem = n_bits - len(binary_l)
        if n_rem > 0:
            pad =  np.zeros((n_rem), dtype=int)
            pad = pad.tolist()
            binary_l = pad + binary_l
        output_labels[b_num, :] = binary_l
    return output_labels

def survival_5(labels):
    new_l = np.zeros_like(labels)
    upper = (labels>5)*1
    new_l += upper

    return new_l

def labels_to_int(labels):
    batch_size, l_dim = labels.shape
    output_labels =  np.zeros((batch_size, 1))
    line = list()
    for ind in range(l_dim):
        line.append(2**ind)
    line = list(reversed(line))
    line = np.array(line)
    for ind in range(batch_size):
        l = labels[ind, :]
        l_int = int(np.sum(np.multiply(l,line)))
        output_labels[ind, :] = l_int
    return output_labels

def labels_normalize(labels, norm_value=50):
    return labels/norm_value


# Gets patch from the original image given the config argument:
# Config: _, y, x, rot, flip
# It will also rotate and flip the patch, and returns depeding on norm/flip.
def get_augmented_patch(path, img_filename, config, patch_h, patch_w, norm=True):
    import skimage.io
    img_path = os.path.join(path, img_filename)
    img = skimage.io.imread(img_path)
    _, y, x, rot, flip = config
    patch = img[y:y+patch_h, x:x+patch_w]
    rotated = np.rot90(patch, rot)
    flipped = np.fliplr(rotated) if flip else rotated
    return flipped / 255.0 if norm else flipped


def get_and_save_patch(augmentations, sets, hdf5_path, dataset_path, train_path, patch_h, patch_w, n_channels, save):
    import skimage.io
    total = len(augmentations)
    hdf5_file = h5py.File(hdf5_path, mode='w')
    img_db_shape = (total, patch_h, patch_w, n_channels)
    _, label_sample = sets[0]
    if not isinstance(label_sample, (list)):
        len_label = 1
    else:
        len_label = len(label_sample)
    labels_db_shape = (total, len_label)
    img_storage = hdf5_file.create_dataset(name='images', shape=img_db_shape, dtype=np.uint8)
    label_storage = hdf5_file.create_dataset(name='labels', shape=labels_db_shape, dtype=np.float32)

    print('\nTotal images: ', total)
    index_patches = 0
    for i, patch_config in enumerate(augmentations):
        # Update on progress.
        if i%100 == 0:
            sys.stdout.write('\r%d%% complete  Images processed: %s' % ((i * 100)/total, i))
            sys.stdout.flush()
        index_set, y, x, rot, flip = patch_config
        file_name, labels = sets[index_set]
        try:
            print(file_name, patch_h, patch_w, patch_config)
            augmented_patch = get_augmented_patch(dataset_path, file_name, patch_config, patch_h, patch_w, norm=False)
        except:
            print('\nCan\'t read image file ', file_name)

        if save:
            label = ''
            if not isinstance(label_sample, (list)):
                label = str(labels)
            else:
                for l in labels:
                    label += '_' + str(l).replace('.', 'p')

            new_file_name = '%s_y%s_x%s_r%s_f%s_label%s.jpg' % (file_name.replace('.jpg', ''), y, x, rot, flip, label)
            new_file_path = os.path.join(train_path, new_file_name)
            skimage.io.imsave(new_file_path, augmented_patch)

        img_storage[index_patches] = augmented_patch
        label_storage[index_patches] = np.array(labels)
        
        index_patches += 1
    hdf5_file.close()
    print()


def make_arrays(train_images, test_images, train_labels, test_labels, patch_h, patch_w, n_channels):
    n_train = len(train_images)
    n_test = len(test_images)
    train_img_data = np.zeros((n_train, patch_h, patch_w, n_channels), dtype=np.uint8)
    train_label_data = np.zeros(n_train, dtype=np.float32)
    test_img_data = np.zeros((n_test, patch_h, patch_w, n_channels), dtype=np.uint8)
    test_label_data = np.zeros(n_test, dtype=np.float32)

    for i in range(n_train):
        train_img_data[i] = train_images[i]
        train_label_data[i] = train_labels[i]
    for i in range(n_test):
        test_img_data[i] = test_images[i]
        test_label_data[i] = test_labels[i]

    return train_img_data, train_label_data, test_img_data, test_label_data


def write_img_data(img_data, patch_h, patch_w, file_name):
    header = np.array([0x0803, len(img_data), patch_h, patch_w], dtype='>i4')
    with open(file_name, "wb") as f:
        f.write(header.tobytes())
        f.write(img_data.tobytes())


def write_label_data(label_data, file_name):
    header = np.array([0x0801, len(label_data)], dtype='>i4')
    with open(file_name, "wb") as f:
        f.write(header.tobytes())
        f.write(label_data.tobytes())


def write_sprite_image(data, filename=None, metadata=True, row_n=None):

    if metadata:
        with open(filename.replace('gen_sprite.png', 'metadata.tsv'),'w') as f:
            f.write("Index\tLabel\n")
            for index in range(data.shape[0]):
                f.write("%d\t%d\n" % (index,index))

    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    
    if data.shape[-1] == 1:
        data = data[:, :, 0]
    if filename is not None:
        plt.imsave(filename, data)

    return data


def read_hdf5(path, dic):
    hdf5_file = h5py.File(path, 'r')
    image_name = dic
    if 'images' == dic and dic not in hdf5_file:
        naming = list(hdf5_file.keys())
        if '_' in naming[0]:
            image_name = '%s_img' % naming[0].split('_')[0]
    if dic not in hdf5_file:
        return None
    return hdf5_file[image_name]


# Method to get ImageNet Inception features and cluster them. 
def inception_feature_labels(hdf5, image_name, patch_h, patch_w, n_channels, num_clusters, clust_percent, batch_size=50, set_type='inception'):
    import tensorflow as tf
    import tensorflow.contrib.gan as tfgan
    import random

    # Verify HDF5 Data file exists.
    if not os.path.isfile(hdf5):
        print('H5 File not found:', hdf5)
        exit()
    print('H5 File found:', hdf5)

    # If we already have the file, return labels.
    hdf5_features = hdf5.replace('.h', '_features_%s_%sclusters.h' % (set_type, num_clusters))
    
    if not os.path.isfile(hdf5_features) and set_type == 'inception':
        # TensorFlow graph
        with tf.Graph().as_default():
            # Resizing and scaling image input to match InceptionV1.
            images_input = tf.placeholder(dtype=tf.float32, shape=[None, patch_h, patch_w, n_channels], name='images')
            images = 2*images_input
            images -= 1
            images = tf.image.resize_bilinear(images, [299, 299])
            out_incept_v3 = tfgan.eval.run_inception(images=images, output_tensor='pool_3:0')
            
            # Start session
            with tf.Session() as sess:
                import umap
                from sklearn.cluster import KMeans
                print('Starting label clustering in Inception Space...')
                with h5py.File(hdf5, mode='r') as hdf5_img_file:
                    with h5py.File(hdf5_features, mode='w') as hdf5_features_file:        

                        # Create storage for features.
                        images_storage = hdf5_img_file[image_name]
                        num_samples = images_storage.shape[0]
                        batches = int(num_samples/batch_size)                        
                        features_storage = hdf5_features_file.create_dataset(name='features', shape=(num_samples, 2048), dtype=np.float32)
                        
                        # Get projections and save them.
                        print('Projecting images...')
                        ind = 0
                        for batch_num in range(batches):
                            batch_images = images_storage[batch_num*batch_size:(batch_num+1)*batch_size]
                            if np.amax(batch_images) > 1.0:
                                batch_images = batch_images/255.
                            activations = sess.run(out_incept_v3, {images_input: batch_images})
                            features_storage[batch_num*batch_size:(batch_num+1)*batch_size] = activations
                            ind += batch_size

                            if ind%10000==0: print('Processed', ind, 'images')
                        print('Processed', ind, 'images')

                        # Grab selected vector for UMAP and clustering.
                        print('Starting label clustering in Inception Space...')
                        all_indx = list(range(num_samples))
                        random.shuffle(all_indx)
                        selected_indx = np.array(sorted(all_indx[:int(num_samples*clust_percent)]))
                        
                        # UMAP
                        print('Running UMAP...')
                        umap_reducer = umap.UMAP(n_components=2, random_state=45)
                        umap_fitted = umap_reducer.fit(features_storage[selected_indx, :])
                        embedding_umap_clustering = umap_fitted.transform(features_storage)
                        print(embedding_umap_clustering.shape)

                        # K-Means
                        print('Running K_means...')
                        feature_labels_storage = hdf5_features_file.create_dataset(name='feat_cluster_labels', shape=[num_samples] + [1], dtype=np.float32)
                        embedding_storage      = hdf5_features_file.create_dataset(name='embedding',           shape=[num_samples] + [2], dtype=np.float32)
                        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10).fit(embedding_umap_clustering)
                        new_classes = kmeans.predict(embedding_umap_clustering)
                        
                        # Save storage for cluster labels.
                        for i in range(num_samples):
                            feature_labels_storage[i, :] = new_classes[i]
                            embedding_storage[i, :]      = embedding_umap_clustering[i, :]

    with h5py.File(hdf5_features, mode='r') as hdf5_features_file: 
        print(hdf5_features)     
        print(hdf5_features_file.keys())     
        new_labels     = np.array(hdf5_features_file['feat_cluster_labels'])
        labels         = copy.deepcopy(np.array(new_labels))
        if 'embedding' not in list(hdf5_features_file.keys()):
            embedding = None
        else:
            new_embedding  = np.array(hdf5_features_file['embedding'])
            embedding      = copy.deepcopy(np.array(new_embedding))
        print('Feature labels collected.')
    return labels, embedding
