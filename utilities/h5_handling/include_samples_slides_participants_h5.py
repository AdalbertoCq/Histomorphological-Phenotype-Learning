# Imports.
import argparse
import h5py
import sys
import os

# Add project path
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

# Own libraries
from data_manipulation.data import Data

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

##### Methods #######
# Methods to include slide and participant into H5 file.
def include_slide_participants_h5(h5_path, reference_field):
    # Sample      'TCGA-AA-3812-01'
    # Slide       'TCGA-02-0001-01C-01-TS1'
    # Participant 'TCGA-02-0001'
    slides_h5       = list()
    samples_h5      = list()
    participants_h5 = list()
    with h5py.File(h5_path, 'r+') as content:
        print('Processing file:', h5_path)
        if reference_field not in content.keys():
            print('\tReference field', reference_field, 'not found')
            print('\tH5 Keys:', ', '.join(content.keys()))
            exit()
        file_name_prev = ''
        for index_h5 in range(content[reference_field].shape[0]):
            file_name = content[reference_field][index_h5,0].decode("utf-8")
            try:
                base_name        = file_name.split('.')[0]
                base_name        = base_name.split('_')[1]
                base_name        = base_name.split('-')
                slide_name       = '-'.join(base_name)
                participant_name = '-'.join(base_name[:3])
                sample_name      = '-'.join(base_name[:4])
                slides_h5.append(slide_name)
                samples_h5.append(sample_name)
                participants_h5.append(participant_name)
            except:
                if file_name_prev != file_name:
                    print('\tCorrupted entry, not able to find mapping:', file_name)
                slides_h5.append('None')
                samples_h5.append('None')
                participants_h5.append('None')
            file_name_prev = file_name

        if 'slides' not in content.keys():
            content.create_dataset('slides',       data=slides_h5)
        if 'samples' not in content.keys():
            content.create_dataset('samples',      data=samples_h5)
        if 'participants' not in content.keys():
            content.create_dataset('participants', data=participants_h5)
        print()


##### Main #######
parser = argparse.ArgumentParser(description='Script to include sample, slide, and participant information from a reference field in the H5.')
parser.add_argument('--img_size',      dest='img_size',      type=int,            default=224,                    help='Image size for the model.')
parser.add_argument('--img_ch',        dest='img_ch',        type=int,            default=3,                      help='Number of channels for the model.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='vgh_nki',              help='Dataset to use.')
parser.add_argument('--marker',        dest='marker',        type=str,            default='he',                   help='Marker of dataset to use.')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,                   help='Path for the output run.')
parser.add_argument('--dbs_path',      dest='dbs_path',      type=str,            default=None,                   help='Directory with DBs to use.')
parser.add_argument('--ref_field',     dest='ref_field',     type=str,            default='file_name',            help='Key name that contains slide and participant information.')
args           = parser.parse_args()
image_width    = args.img_size
image_height   = args.img_size
image_channels = args.img_ch
dataset        = args.dataset
marker         = args.marker
main_path      = args.main_path
dbs_path       = args.dbs_path
ref_field      = args.ref_field

if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))
    main_path = '/'.join(main_path.split('/')[:-2])

if dbs_path is None:
    dbs_path = main_path

# Data Class with all h5
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=64, project_path=dbs_path, load=False)

for h5_path in [data.hdf5_train, data.hdf5_validation, data.hdf5_test]:
    if not os.path.isfile(h5_path):
        print('Warning: H5 file not found', h5_path)
        continue
    include_slide_participants_h5(h5_path, ref_field)

#
# def representations_h5(main_path, model, dataset, img_size, zdim):
#     hdf5_path_train = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_train.h5'      % (main_path, model, dataset, img_size, img_size, zdim, dataset)
#     hdf5_path_valid = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_validation.h5' % (main_path, model, dataset, img_size, img_size, zdim, dataset)
#     hdf5_path_test  = '%s/results/%s/%s/h%s_w%s_n3_zdim%s/hdf5_%s_he_test.h5'       % (main_path, model, dataset, img_size, img_size, zdim, dataset)
#     return [hdf5_path_train, hdf5_path_valid, hdf5_path_test]
#
#
# data = representations_h5(main_path, 'ContrastivePathology_BarlowTwins_3', dataset, 224, 128)
# for h5_path in [data[0], data[1], data[2]]:
#     if not os.path.isfile(h5_path):
#         print('Warning: H5 file not found', h5_path)
#         continue
#     include_slide_participants_h5(h5_path, ref_field)
