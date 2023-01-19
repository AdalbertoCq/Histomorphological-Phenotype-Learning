# Imports
import argparse
import os

# Own libs.
from models.clustering.evaluation_metrics import evaluate_cluster_configurations

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


##### Main #######
parser = argparse.ArgumentParser(description='Report classification and cluster performance based on Logistic Regression.')
parser.add_argument('--meta_folder',         dest='meta_folder',         type=str,            default=None,        help='Purpose of the clustering, name of folder.')
parser.add_argument('--folds_pickle',        dest='folds_pickle',        type=str,            default=None,        help='Pickle file with folds information.')
parser.add_argument('--h5_complete_path',    dest='h5_complete_path',    type=str,            required=True,       help='H5 file path to run the leiden clustering folds.')
parser.add_argument('--include_silnngraph',  dest='include_silnngraph',  action='store_true', default=False,       help='Flag to specify if silhoutte based on NN graph is run (comp. costly).')
args               = parser.parse_args()
meta_folder        = args.meta_folder
folds_pickle       = args.folds_pickle
h5_complete_path   = args.h5_complete_path
include_silnngraph = args.include_silnngraph

resolutions = [0.25, 0.4, 0.5, 0.7, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0]
# resolutions = [0.25, 0.4, 0.5, 0.7, 0.75, 1.0]

evaluate_cluster_configurations(h5_complete_path, meta_folder, folds_pickle, resolutions, threshold_inst=0.01, include_nngraph=include_silnngraph)
