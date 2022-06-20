# Imports
import argparse
import os

# Own libs.
from models.clustering.logistic_regression_leiden_clusters import run_circular_plots
from models.clustering.cox_proportional_hazard_regression_leiden_clusters import *


##### Main #######
parser = argparse.ArgumentParser(description='Report classification and cluster performance based on Logistic Regression.')
parser.add_argument('--meta_folder',         dest='meta_folder',         type=str,            default=None,        help='Purpose of the clustering, name of folder.')
parser.add_argument('--matching_field',      dest='matching_field',      type=str,            default=None,        help='Key used to match folds split and H5 representation file.')
parser.add_argument('--event_ind_field',     dest='event_ind_field',     type=str,            default=None,        help='Key used to match event indicator field.')
parser.add_argument('--event_data_field',    dest='event_data_field',    type=str,            default=None,        help='Key used to match event data field.')
parser.add_argument('--diversity_key',       dest='diversity_key',       type=str,            default=None,        help='Key use to check diversity within cluster: Slide, Institution, Sample.')
parser.add_argument('--type_composition',    dest='type_composition',    type=str,            default='clr',       help='Space transformation type: percent, clr, ilr, alr.')
parser.add_argument('--min_tiles',           dest='min_tiles',           type=int,            default=100,         help='Minimum number of tiles per matching_field.')
parser.add_argument('--folds_pickle',        dest='folds_pickle',        type=str,            default=None,        help='Pickle file with folds information.')
parser.add_argument('--force_fold',          dest='force_fold',          type=int,            default=None,        help='Force fold of clustering.')
parser.add_argument('--h5_complete_path',    dest='h5_complete_path',    type=str,            required=True,       help='H5 file path to run the leiden clustering folds.')
parser.add_argument('--h5_additional_path',  dest='h5_additional_path',  type=str,            default=None,        help='Additional H5 representation to assign leiden clusters.')
parser.add_argument('--additional_as_fold',  dest='additional_as_fold',  action='store_true', default=False,       help='Flag to specify if additional H5 file will be used for cross-validation.')
parser.add_argument('--report_clusters',     dest='report_clusters',     action='store_true', default=False,       help='Flag to report cluster circular plots.')
args               = parser.parse_args()
meta_folder        = args.meta_folder
matching_field     = args.matching_field
event_ind_field    = args.event_ind_field
event_data_field   = args.event_data_field
diversity_key      = args.diversity_key
type_composition   = args.type_composition
min_tiles          = args.min_tiles
folds_pickle       = args.folds_pickle
force_fold         = args.force_fold
h5_complete_path   = args.h5_complete_path
h5_additional_path = args.h5_additional_path
additional_as_fold = args.additional_as_fold
report_clusters    = args.report_clusters

max_months = 15.0*15.0

# Use connectivity between clusters as features.
use_conn          = False
use_ratio         = False
top_variance_feat = 99

# Alphas and resolutions.
l1_ratios   = [0.0, 0.1, 0.2, 0.5, 0.75, 1.0]
alphas      = 10. ** np.linspace(-4, 4, 50)

# resolutions = [0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
resolutions = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
resolutions = [1.0, 2.0]

# Report figures for clusters.
if report_clusters: run_circular_plots(resolutions, meta_folder, event_ind_field, matching_field, folds_pickle, h5_complete_path, h5_additional_path, diversity_key)

# Run Cox Proportional Hazard Regression with L1/L2 Penalties.
run_cph_regression_exhaustive(alphas, resolutions, meta_folder, matching_field, folds_pickle, event_ind_field, event_data_field, h5_complete_path, h5_additional_path, diversity_key,
                              type_composition, min_tiles, max_months, additional_as_fold, force_fold, l1_ratios, use_conn=use_conn, use_ratio=use_ratio, top_variance_feat=top_variance_feat)

# Summarize AUC performance for top 4 penalties.
summary_resolution_cindex(resolutions, h5_complete_path, meta_folder, l1_ratios, min_tiles, force_fold)

