import numpy as np
import argparse
import h5py
import sys
import os

# Add project path
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


##### Methods #######
# Get set paths.
def representations_h5(main_path, model, dataset, img_size, zdim):
    hdf5_path_train = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_train.h5'      % (main_path, model, dataset, img_size, img_size, zdim, dataset)
    hdf5_path_valid = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_validation.h5' % (main_path, model, dataset, img_size, img_size, zdim, dataset)
    hdf5_path_test  = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_test.h5'       % (main_path, model, dataset, img_size, img_size, zdim, dataset)
    return [hdf5_path_train, hdf5_path_valid, hdf5_path_test]

# Get total number of tiles for all sets.
def get_total_samples(data):
    total_samples = 0
    for h5_path in [data[0], data[1], data[2]]:
        if not os.path.isfile(h5_path):
            print('Warning: H5 file not found', h5_path)
            continue
        with h5py.File(h5_path, 'r') as content:
            key_1 = list(content.keys())[0]
            total_samples += content[key_1].shape[0]
    return total_samples

# Get key_names, shape, and dtype
def data_specs(data):
    key_dict = dict()
    with h5py.File(data[0], 'r') as content:
        for key in content.keys():
            key_dict[key] = dict()
            key_dict[key]['shape'] = content[key].shape[1:]
            key_dict[key]['dtype'] = content[key].dtype

    return key_dict


def create_complete_h5(data, num_tiles, key_dict, override):
    h5_complete_path = data[0].replace('_train.h5', '_complete.h5')
    if override:
        os.remove(h5_complete_path)
    if os.path.isfile(h5_complete_path):
        print('File already exists, if you want to overwrite enable the flag --override')
        print(h5_complete_path)
        print()
        exit()

    storage_dict = dict()
    content      = h5py.File(h5_complete_path, mode='w')
    for key in key_dict:
        shape = [num_tiles] + list(key_dict[key]['shape'])
        dtype = key_dict[key]['dtype']
        storage_dict[key] = content.create_dataset(name=key.replace('train_', ''), shape=shape, dtype=dtype)

    storage_ref_dict = dict()
    dt = h5py.special_dtype(vlen=str)
    storage_ref_dict['indexes']      = content.create_dataset(name='indexes',      shape=[shape[0]], dtype=np.int32)
    storage_ref_dict['original_set'] = content.create_dataset(name='original_set', shape=[shape[0]], dtype=dt)

    index = 0
    for set_path, set_name in [(data[0], 'train'), (data[1], 'valid'), (data[2], 'test')]:
        print('Iterating through %s ...' % set_name)

        with h5py.File(set_path, 'r') as content:
            set_dict = dict()
            for key in storage_dict:
                set_dict[key] = content[key.replace('train_', '%s_'%set_name)]

            for i in range(set_dict[key].shape[0]):
                # Original data.
                for key in storage_dict:
                    storage_dict[key][index] = set_dict[key][i]
                # Back-referencing to sets.
                storage_ref_dict['indexes'][index]      = i
                storage_ref_dict['original_set'][index] = set_name

                # Verbose.
                if i%1e+5==0:   
                    print('\tprocessed %s entries' % i)
                index += 1
        print()

##### Main #######
parser = argparse.ArgumentParser(description='Script to combine all H5 representation file into a \'complete\' one.')
parser.add_argument('--img_size',      dest='img_size',      type=int,            default=224,                    help='Image size for the model.')
parser.add_argument('--z_dim',         dest='z_dim',         type=int,            default=128,                    help='Dimensionality of projections, default is the Z latent of Self-Supervised.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='vgh_nki',              help='Dataset to use.')
parser.add_argument('--model',         dest='model',         type=str,            default='ContrastivePathology', help='Model name, used to select the type of model (SimCLR, BYOL, SwAV).')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,                   help='Path for the output run.')
parser.add_argument('--override',      dest='override',      action='store_true', default=False,                  help='Override \'complete\' H5 file if it already exists.')
args           = parser.parse_args()
img_size       = args.img_size
z_dim          = args.z_dim
dataset        = args.dataset
model          = args.model
main_path      = args.main_path
override       = args.override

if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))
    main_path = '/'.join(main_path.split('/')[:-2])

# Get representations paths.
data = representations_h5(main_path, model, dataset, img_size, z_dim)

# Get total number fo samples.
num_tiles = get_total_samples(data)

# Dictionary with keys, shapes, and dtypes.
key_dict = data_specs(data)

# Combine all H5 into a 'complete' one.
create_complete_h5(data, num_tiles, key_dict, override)
