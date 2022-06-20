# Imports.
import argparse
import random
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
# Get key_names, shape, and dtype.
def data_specs(data_path, num_samples):
	key_dict = dict()
	with h5py.File(data_path, 'r') as content:
		for key in content.keys():
			key_dict[key] = dict()
			key_dict[key]['shape'] = content[key].shape[1:]
			key_dict[key]['dtype'] = content[key].dtype
		h5_samples = content[key].shape[0]

		if num_samples > content[key].shape[0]:
			print('Number of subsamples %s is smaller than the size of original H5 file %s' % (num_samples,content[key].shape[0]))
			exit()

	return key_dict, h5_samples

# Get subsamples from original H5 file.
def create_complete_h5(data_path, num_samples, key_dict, override):
	h5_complete_path = data_path.replace('.h5', '_%s_subsampled.h5' % num_samples)
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
		shape = [num_samples] + list(key_dict[key]['shape'])
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

		all_indexes = list(range(content[key].shape[0]))
		random.shuffle(all_indexes)
		all_indexes = all_indexes[:num_samples]

		for i, random_index in enumerate(all_indexes):
			# Original data.
			for key in storage_dict:
				storage_dict[key][index] = set_dict[key][random_index]

			# Verbose.
			if i%1e+5==0:   
				print('\tprocessed %s entries' % i)
			index += 1
	print()


##### Main #######
parser = argparse.ArgumentParser(description='Script to subsample indexes from H5 file.')
parser.add_argument('--h5_file',       dest='h5_file',       type=str,            required=True,  help='Original H5 file to parse.')
parser.add_argument('--num_samples',   dest='num_samples',   type=int,            required=True,  help='Number of random subsamples to pick from original H5 file.')
parser.add_argument('--override',      dest='override',      action='store_true', default=False,  help='Override \'complete\' H5 file if it already exists.')
args        = parser.parse_args()
h5_file     = args.h5_file
num_samples = args.num_samples
override    = args.override


# Check if files exist.
if not os.path.isfile(h5_file):
	print('File not found:', h5_file)
	exit()

# Information content H5 file.
key_dict, h5_samples = data_specs(h5_file, num_samples)

# Remove from H5 file.
create_complete_h5(h5_file, num_samples, key_dict, override)

