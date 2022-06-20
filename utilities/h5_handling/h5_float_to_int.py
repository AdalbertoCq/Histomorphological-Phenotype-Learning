import numpy as np
import argparse
import shutil
import math
import h5py
import glob
import sys
import os


parser = argparse.ArgumentParser(description='Convert H5 file image dataset from float32 to uint8, saving storage.')
parser.add_argument('--dataset',      dest='dataset',      type=str,  default=None,  help='Dataset to use.')
parser.add_argument('--marker',       dest='marker',       type=str,  default='he',  help='Marker of dataset to use.')
parser.add_argument('--img_size',     dest='img_size',     type=int,  default=224,   help='Image size for the model.')
parser.add_argument('--img_ch',       dest='img_ch',       type=int,  default=3,     help='Number of channels for the model.')
parser.add_argument('--dbs_path',     dest='dbs_path',     type=str,  default=None,  help='Path where datasets are stored.')
args     = parser.parse_args()
dataset  = args.dataset
marker   = args.marker
img_size = args.img_size
img_ch   = args.img_ch
dbs_path = args.dbs_path

if dbs_path is None:
    dbs_path = os.path.dirname(os.path.realpath(__file__))
    dbs_path = '/'.join(dbs_path.split('/')[:-2])

sys.path.append(dbs_path)
from data_manipulation.data import Data

print('Dataset:', dataset)
batch_size = 100

data = Data(dataset=dataset, marker=marker, patch_h=img_size, patch_w=img_size, n_channels=img_ch, batch_size=64, project_path=dbs_path, load=False)

# Dataset files.
for hdf5_path in [data.hdf5_train, data.hdf5_validation, data.hdf5_test]:

    hdf5_path_new = hdf5_path + '_int'
    if os.path.isfile(hdf5_path_new):
        os.remove(hdf5_path_new)

    print('Current File:', hdf5_path)
    with h5py.File(hdf5_path, 'r') as original_content:
        with h5py.File(hdf5_path_new, mode='w') as hdf5_content:
            for key in original_content.keys():
                print('\t', key, '-', original_content[key].shape)
                if 'images' in key or 'img' in key:
                    normalized_flag = False
                    if np.amax(original_content[key][:10, :, :, :]) == 1.:
                        normalized_flag = True
                    img_storage = hdf5_content.create_dataset(key, shape=original_content[key].shape, dtype=np.uint8)
                    blocks = math.ceil(original_content[key].shape[0]/batch_size)
                    for i in range(blocks):
                        if normalized_flag:
                            img_storage[i*batch_size:(i+1)*batch_size, :, :, :] = original_content[key][i*batch_size:(i+1)*batch_size, :, :, :]*255
                        else:
                            img_storage[i*batch_size:(i+1)*batch_size, :, :, :] = original_content[key][i*batch_size:(i+1)*batch_size, :, :, :]
                        if i*batch_size%10000==0: print('\t\t', 'Processed', i*batch_size, 'images')
                else:
                    hdf5_content.create_dataset(key,   data=original_content[key])
    os.remove(hdf5_path)
    shutil.move(hdf5_path_new, hdf5_path)
