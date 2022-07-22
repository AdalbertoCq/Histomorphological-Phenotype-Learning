# Imports
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Own libs.
from models.clustering.leiden_representations import run_leiden

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


##### Main #######
parser = argparse.ArgumentParser(description='Run Leiden Community detection over Self-Supervised representations.')
parser.add_argument('--resolution',          dest='resolution',          type=float,   default=None,        help='Leiden resolution.')
parser.add_argument('--subsample',           dest='subsample',           type=int,    default=None,        help='Number of sample to run Leiden on, default is None, 200000 works well.')
parser.add_argument('--n_neighbors',         dest='n_neighbors',         type=int,    default=250,         help='Number of neighbors to use when creating the graph, default is 250.')
parser.add_argument('--meta_field',           dest='meta_field',           type=str,    default=None,        help='Purpose of the clustering, name of folder.')
parser.add_argument('--matching_field',       dest='matching_field',       type=str,    default=None,        help='Key used to match folds split and H5 representation file.')
parser.add_argument('--rep_key',             dest='rep_key',             type=str,    default='z_latent',  help='Key pattern for representations to grab: z_latent, h_latent.')
parser.add_argument('--folds_pickle',        dest='folds_pickle',        type=str,    default=None,        help='Pickle file with folds information.')
parser.add_argument('--main_path',           dest='main_path',           type=str,    default=None,        help='Workspace main path.')
parser.add_argument('--h5_complete_path',    dest='h5_complete_path',    type=str,    required=True,       help='H5 file path to run the leiden clustering folds.')
parser.add_argument('--h5_additional_path',  dest='h5_additional_path',  type=str,    default=None,        help='Additional H5 representation to assign leiden clusters.')
args               = parser.parse_args()
subsample          = args.subsample
resolution         = args.resolution
n_neighbors        = args.n_neighbors
meta_field          = args.meta_field
matching_field      = args.matching_field
rep_key            = args.rep_key
folds_pickle       = args.folds_pickle
main_path          = args.main_path
h5_complete_path   = args.h5_complete_path
h5_additional_path = args.h5_additional_path

if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))

# Default resolutions.
if resolution is None:
    resolutions = [0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # resolutions.extend([6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
else:
    resolutions = [resolution]

# Include surrounding cluster tile annotations.
include_connections = False

# Run leiden clustering.
run_leiden(meta_field, matching_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions, n_neighbors=n_neighbors, subsample=subsample, include_connections=include_connections)