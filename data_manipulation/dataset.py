from data_manipulation.utils import inception_feature_labels
import numpy as np
import random
import h5py


class Dataset:
    def __init__(self, hdf5_path, patch_h, patch_w, n_channels, batch_size, thresholds=(), labels=None, empty=False, num_clusters=500, clust_percent=1.0):

        self.i = 0
        self.batch_size = batch_size
        self.done = False
        self.thresholds = thresholds
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels

        # Options for conditional PathologyGAN
        self.num_clusters = num_clusters
        self.clust_percent = clust_percent

        self.labels_name = labels
        if labels is None:
            self.labels_flag = False
        else:
            self.labels_flag = True

        self.hdf5_path = hdf5_path
        # Get images and labels
        self.images = list()
        self.labels = list()
        if not empty:
            self.images, self.labels, self.embedding = self.get_hdf5_data()
        self.size = len(self.images)
        self.iterations = len(self.images)//self.batch_size + 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    @property
    def shape(self):
        return [len(self.images), self.patch_h, self.patch_w, self.n_channels]

    def get_hdf5_data(self):
        hdf5_file = h5py.File(self.hdf5_path, 'r')

        # Legacy code for initial naming of images, label keys.
        labels_name = self.labels_name
        naming = list(hdf5_file.keys())
        if 'images' in naming:
            image_name = 'images'
            if labels_name is None:
                labels_name = 'labels'       
        else:
            for naming in list(hdf5_file.keys()):     
                if 'img' in naming or 'image' in naming:
                    image_name = naming
                elif 'labels' in naming and self.labels_name is None:
                    labels_name = naming

        # Get images, labels, and embeddings if neccesary.
        images    = hdf5_file[image_name]
        embedding = None
        labels    = np.zeros((images.shape[0]))
        if self.labels_flag:
            if self.labels_name == 'inception' or self.labels_name == 'self':
                labels, embedding = inception_feature_labels(self.hdf5_path, image_name, self.patch_h, self.patch_w, self.n_channels, self.num_clusters, self.clust_percent, set_type=self.labels_name)
                labels, embedding = inception_feature_labels(self.hdf5_path, image_name, self.patch_h, self.patch_w, self.n_channels, self.num_clusters, self.clust_percent, set_type=self.labels_name)
            else:
                labels = hdf5_file[labels_name]
        return images, labels, embedding

    def set_pos(self, i):
        self.i = i

    def get_pos(self):
        return self.i

    def reset(self):
        self.set_pos(0)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def adapt_label(self, label):
        thresholds = self.thresholds + (None,)
        adapted = [0.0 for _ in range(len(thresholds))]
        i = None
        for i, threshold in enumerate(thresholds):
            if threshold is None or label < threshold:
                break
        adapted[i] = label if len(adapted) == 1 else 1.0
        return adapted

    def next_batch(self, n):
        if self.done:
            self.done = False
            raise StopIteration
        batch_img = self.images[self.i:self.i + n]
        batch_labels = self.labels[self.i:self.i + n]
        self.i += len(batch_img)
        delta = n - len(batch_img)
        if delta == n:
            raise StopIteration
        if 0 < delta:
            batch_img = np.concatenate((batch_img, self.images[:delta]), axis=0)
            batch_labels = np.concatenate((batch_labels, self.labels[:delta]), axis=0)
            self.i = delta
            self.done = True
        return batch_img/255.0, batch_labels
