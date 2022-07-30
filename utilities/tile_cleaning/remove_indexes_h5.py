# Imports.
import argparse
import h5py
import sys
import os

# Add project path.
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

# Own libs.
from data_manipulation.utils import load_data

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


##### Methods #######
# Get key_names, shape, and dtype.
def data_specs(data_path):
    key_dict = dict()
    with h5py.File(data_path, 'r') as content:
        for key in content.keys():
            key_dict[key] = dict()
            key_dict[key]['shape'] = content[key].shape[1:]
            key_dict[key]['dtype'] = content[key].dtype
        h5_samples = content[key].shape[0]

    return key_dict, h5_samples

# Filter out given indexes.
def create_complete_h5(data_path, num_tiles, key_dict, indexes_to_remove, override):
    h5_complete_path = data_path.replace('.h5', '_filtered.h5')
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
        key_  = key.replace('train_', '')
        key_  = key_.replace('valid_', '')
        key_  = key_.replace('test_', '')
        storage_dict[key] = content.create_dataset(name=key_, shape=shape, dtype=dtype)

    index = 0
    print('Iterating through %s ...' % data_path)
    with h5py.File(data_path, 'r') as content:
        set_dict = dict()
        for key in storage_dict:
            set_dict[key] = content[key]

        for i in range(set_dict[key].shape[0]):
            # Original data.
            for key in storage_dict:
                storage_dict[key][index] = set_dict[key][i]
            
            # Check if this is a tile to remove
            if i in indexes_to_remove:
            	indexes_to_remove.remove(i)
            	continue

            if num_tiles == index:
                break

            # Verbose.
            if i%1e+5==0:   
                print('\tprocessed %s entries' % i)
            index += 1
    print()


##### Main #######
parser = argparse.ArgumentParser(description='Script to remove indexes from H5 file.')
parser.add_argument('--h5_file',       dest='h5_file',       type=str,            required=True,  help='Original H5 file to parse.')
parser.add_argument('--pickle_file',   dest='pickle_file',   type=str,            required=True,  help='Pickle file with indexes to remove.')
parser.add_argument('--override',      dest='override',      action='store_true', default=False,  help='Override \'complete\' H5 file if it already exists.')
args        = parser.parse_args()
h5_file     = args.h5_file
pickle_file = args.pickle_file
override    = args.override


# Check if files exist.
for file_path in [h5_file, pickle_file]:
	if not os.path.isfile(file_path):
		print('File not found:', file_path)
		exit()

# Grab indexes.
indexes_to_remove = load_data(pickle_file)

# Information content H5 file.
key_dict, h5_samples = data_specs(h5_file)

# Remove from H5 file.
remain_samples = h5_samples - len(indexes_to_remove)
create_complete_h5(h5_file, remain_samples, key_dict, indexes_to_remove, override)

