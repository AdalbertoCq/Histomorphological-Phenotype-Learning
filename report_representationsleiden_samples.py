# Imports
import pandas as pd
import argparse
import os

# Own libraries
from data_manipulation.data import Data
from models.visualization.clusters import plot_cluster_images, plot_wsi_clusters, plot_wsi_clusters_interactions


##### Main #######
parser = argparse.ArgumentParser(description='Report cluster images from a given Leiden cluster configuration.')
parser.add_argument('--meta_folder',         dest='meta_folder',         type=str,            default=None,                   help='Purpose of the clustering, name of folder.')
parser.add_argument('--meta_field',           dest='meta_field',           type=str,            default=None,                   help='Meta field to use for the Logistic Regression or Cox event indicator.')
parser.add_argument('--matching_field',       dest='matching_field',       type=str,            default=None,                   help='Key used to match folds split and H5 representation file.')
parser.add_argument('--resolution',          dest='resolution',          type=float,           default=None,                   help='Minimum number of tiles per matching_field.')
parser.add_argument('--dpi',                 dest='dpi',                 type=int,            default=1000,                   help='Highest quality: 1000.')
parser.add_argument('--fold',                dest='fold',                type=int,            default=0,                      help='Minimum number of tiles per matching_field.')
parser.add_argument('--dataset',             dest='dataset',             type=str,            default='TCGAFFPE_LUADLUSC_5x', help='Dataset to use.')
parser.add_argument('--h5_complete_path',    dest='h5_complete_path',    type=str,            required=True,                  help='H5 file path to run the leiden clustering folds.')
parser.add_argument('--h5_additional_path',  dest='h5_additional_path',  type=str,            default=None,                   help='Additional H5 representation to assign leiden clusters.')
parser.add_argument('--min_tiles',           dest='min_tiles',           type=int,            default=400,                    help='Minimum number of tiles per matching_field.')
parser.add_argument('--dbs_path',            dest='dbs_path',            type=str,            default=None,                   help='Path for the output run.')
parser.add_argument('--img_size',            dest='img_size',            type=int,            default=224,                    help='Image size for the model.')
parser.add_argument('--img_ch',              dest='img_ch',              type=int,            default=3,                      help='Number of channels for the model.')
parser.add_argument('--marker',              dest='marker',              type=str,            default='he',                   help='Marker of dataset to use.')
parser.add_argument('--tile_img',            dest='tile_img',            action='store_true', default=False,                  help='Dump cluster tile images.')
parser.add_argument('--extensive',           dest='extensive',           action='store_true', default=False,                  help='Flag to dump test set cluster images in addition to train.')
parser.add_argument('--additional_as_fold',  dest='additional_as_fold',  action='store_true', default=False,                  help='Flag to specify if additional H5 file will be used for cross-validation.')
args               = parser.parse_args()
meta_folder        = args.meta_folder
meta_field          = args.meta_field
matching_field      = args.matching_field
resolution         = args.resolution
dpi                = args.dpi
fold               = args.fold
min_tiles          = args.min_tiles
image_height       = args.img_size
image_width        = args.img_size
image_channels     = args.img_ch
marker             = args.marker
dataset            = args.dataset
h5_complete_path   = args.h5_complete_path
h5_additional_path = args.h5_additional_path
dbs_path           = args.dbs_path
tile_img           = args.tile_img
extensive          = args.extensive
additional_as_fold = args.additional_as_fold

# Dominating clusters to pull WSI.
only_id = True
value_cluster_ids = dict()
# value_cluster_ids[1] = []
# value_cluster_ids[0] = []

########################################################
############# LUAD vs LUSC #############################
# Leiden_2.0 fold 4.
# value_cluster_ids = dict()
# value_cluster_ids[1] = [11,31,28,36,22,35]
# value_cluster_ids[0] = [5, 45,]
# only_id = False
# ## Leiden_1.0 fold 4.
# value_cluster_ids[1] = [11,12,13]
# value_cluster_ids[0] = [14,26]
# only_id = False

########################################################
############# LUAD OS ##################################
## Leiden 2.0 fold 0.
# value_cluster_ids = dict()
# value_cluster_ids[0] = [31, 1,37, 0,16, 8, 5]
# value_cluster_ids[1] = [15,39,41,22,10,14,27]
# only_id = True
## Leiden 1.0 fold 0.
# value_cluster_ids[0] = [13, 8, 3, 9, 5]
# value_cluster_ids[1] = [14,18, 7,15]
# only_id = True

########################################################
############# LUAD PFS #################################
## Leiden 2.0 fold 0.
# value_cluster_ids = dict()
# value_cluster_ids[0] = [39,45,29,27,22,36,32, 0,37,21]
# value_cluster_ids[1] = [15,11, 6,44, 5,24]
# only_id = True
# Leiden 1.0 fold 0.
# value_cluster_ids[0] = [27,20,22, 0,26, 5,21, 4, 8,11]
# value_cluster_ids[1] = [14, 7,12,1]
# only_id = True

########################################################
############# Interaction OS/PFS #######################
inter_dict = dict()
# inter_dict['value'] = ['4_13','4_2','4_32','4_10','13_2','13_32','13_9','2_9','2_32','2_10','2_3','6_34',
#                       '6_5','6_7','23_3','23_22','16_1','16_0','0_1','0_9','0_25','0_5','0_17','1_9','1_25',
#                       '1_5','1_7','5_15','28_27','28_35','21_37','21_8','21_26','14_7','11_7','11_15','7_15']

# Default path for GDC manifest.
manifest_csv = '%s/utilities/files/LUADLUSC/gdc_manifest.txt' % os.path.dirname(os.path.realpath(__file__))

# Default DBs path. 
if dbs_path is None:
    dbs_path = os.path.dirname(os.path.realpath(__file__))

# Leiden convention name.
groupby = 'leiden_%s' % resolution

# Dataset images.
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=64, project_path=dbs_path, load=True)

# Dump cluster images.
if tile_img:
    plot_cluster_images(groupby, meta_folder, data, fold, h5_complete_path, dpi, value_cluster_ids, extensive=extensive)

# Save WSI overlay with clusters.
plot_wsi_clusters(groupby, meta_folder, matching_field, meta_field, data, fold, h5_complete_path, h5_additional_path, additional_as_fold, dpi, min_tiles, manifest_csv=manifest_csv,
                  value_cluster_ids=value_cluster_ids, type_='percent', only_id=only_id, n_wsi_samples=3)

# Save WSI overlay with clusters.
# plot_wsi_clusters_interactions(groupby, meta_folder, 'slides', meta_field, data, fold, h5_complete_path, h5_additional_path, additional_as_fold, dpi, min_tiles, manifest_csv=manifest_csv,
#                                inter_dict=inter_dict, type_='percent', only_id=only_id, n_wsi_samples=2)
