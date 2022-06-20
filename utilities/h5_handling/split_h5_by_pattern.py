# Imports.
import argparse
import h5py
import sys
import os

# Add project path.
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

from data_manipulation.utils import load_data

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


##### Methods #######
# Get number of samples that contain pattern in matching field.
def h5_overlap_pattern_individuals(h5_file, matching_field, pattern):
    h5_samples = 0
    with h5py.File(h5_file, 'r') as content:
        for index_h5 in range(content[matching_field].shape[0]):
            field_value = content[matching_field][index_h5].decode("utf-8")
            if pattern in field_value:
                h5_samples +=1

    return h5_samples

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
def create_complete_h5(data_path, num_tiles, key_dict, pattern, matching_field, override):
    h5_complete_path = data_path.replace('.h5', '_%s.h5' % pattern)
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
            # Verbose.
            if i%1e+5==0:   
                print('\tprocessed %s entries' % i)
                
            # Check in pattern is contained in this instance.
            if pattern not in set_dict[matching_field][i].decode("utf-8"):
                continue

            # Original data.
            for key in storage_dict:
                storage_dict[key][index] = set_dict[key][i]
            
            if num_tiles == index:
                break

            index += 1
    print()


##### Main #######
parser = argparse.ArgumentParser(description='Script to create a new H5 file that contains a particular pattern.')
parser.add_argument('--h5_file',         dest='h5_file',         type=str,            required=True,  help='Original H5 file to parse.')
parser.add_argument('--matching_field',  dest='matching_field',  type=str,            required=True,  help='Reference filed to use, cross check between original H5 and meta file.')
parser.add_argument('--pattern',         dest='pattern',         type=str,            required=True,  help='Pattern to search for in the matching_field entries.')
parser.add_argument('--override',        dest='override',        action='store_true', default=False,  help='Override \'complete\' H5 file if it already exists.')
args           = parser.parse_args()
h5_file        = args.h5_file
matching_field = args.matching_field
pattern        = args.pattern
override       = args.override

# Get number of tiles from all individuals in the original H5, <= to the original.
num_tiles = h5_overlap_pattern_individuals(h5_file, matching_field, pattern)

# Information content H5 file.
key_dict, h5_samples = data_specs(h5_file)

# Remove from H5 file.
create_complete_h5(h5_file, num_tiles, key_dict, pattern, matching_field, override)

