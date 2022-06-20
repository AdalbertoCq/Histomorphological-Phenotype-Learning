# Imports.
from data_manipulation.data import Data
import tensorflow as tf
import argparse
import os

# Folder permissions for cluster.
os.umask(0o002)
# H5 File bug over network file system.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

parser = argparse.ArgumentParser(description='Self-Supervised model training.')
parser.add_argument('--epochs',        dest='epochs',        type=int,            default=90,                     help='Number epochs to run: default is 45 epochs.')
parser.add_argument('--batch_size',    dest='batch_size',    type=int,            default=64,                     help='Batch size, default size is 64.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='vgh_nki',              help='Dataset to use.')
parser.add_argument('--marker',        dest='marker',        type=str,            default='he',                   help='Marker of dataset to use, default is H&E.')
parser.add_argument('--img_size',      dest='img_size',      type=int,            default=224,                    help='Image size for the model.')
parser.add_argument('--img_ch',        dest='img_ch',        type=int,            default=3,                      help='Number of channels for the model.')
parser.add_argument('--z_dim',         dest='z_dim',         type=int,            default=128,                    help='Latent space size for constrastive loss.')
parser.add_argument('--model',         dest='model',         type=str,            default='ContrastivePathology', help='Model name, used to select the type of model (SimCLR, BYOL, SwAV).')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,                   help='Path for the output run.')
parser.add_argument('--dbs_path',      dest='dbs_path',      type=str,            default=None,                   help='Directory with DBs to use.')
parser.add_argument('--check_every',   dest='check_every',   type=int,            default=10,                     help='Save checkpoint and projects samples every X epcohs.')
parser.add_argument('--restore',       dest='restore',       action='store_true', default=False,                  help='Restore previous run and continue.')
parser.add_argument('--report',        dest='report',        action='store_true', default=False,                  help='Report Latent Space progress.')
args           = parser.parse_args()
epochs         = args.epochs
batch_size     = args.batch_size
dataset        = args.dataset
marker         = args.marker
image_width    = args.img_size
image_height   = args.img_size
image_channels = args.img_ch
z_dim          = args.z_dim
model          = args.model
main_path      = args.main_path
dbs_path       = args.dbs_path
check_every    = args.check_every
restore        = args.restore
report         = args.report


# Main paths for data output and databases.
if main_path is None:
	main_path = os.path.dirname(os.path.realpath(__file__))
if dbs_path is None:
	dbs_path = os.path.dirname(os.path.realpath(__file__))

# Directory handling.
name_run = 'h%s_w%s_n%s_zdim%s' % (image_height, image_width, image_channels, z_dim)
data_out_path = os.path.join(main_path, 'data_model_output')
data_out_path = os.path.join(data_out_path, model)
data_out_path = os.path.join(data_out_path, dataset)
data_out_path = os.path.join(data_out_path, name_run)

# Hyperparameters for training.
regularizer_scale = 1e-4
learning_rate_e   = 5e-4
beta_1            = 0.5

# Model Architecture param.
layers_map = {512:7, 448:6, 256:6, 224:5, 128:5, 112:4, 56:3, 28:2}
layers     = layers_map[image_height]
spectral   = True
attention  = 56
init       = 'xavier'
# init       = 'orthogonal'

# Testing with CRC and 56x56.
if image_height == 56: attention = 28

# Handling of different models.
if 'BYOL' in model:
	z_dim = 256
	from models.selfsupervised.BYOL import RepresentationsPathology
elif 'SimCLR' in model:
	from models.selfsupervised.SimCLR import RepresentationsPathology
elif 'SwAV' in model:
	learning_rate_e   = 1e-5
	from models.selfsupervised.SwAV import RepresentationsPathology
elif 'SimSiam' in model:
	from models.selfsupervised.SimSiam import RepresentationsPathology
elif 'Relational' in model:
	from models.selfsupervised.RealReas import RepresentationsPathology
elif 'BarlowTwins' in model:
	from models.selfsupervised.BarlowTwins import RepresentationsPathology

# Collect dataset.
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

# Run PathologyContrastive Encoder.
with tf.Graph().as_default():
	# Instantiate Model.
    contrast_pathology = RepresentationsPathology(data=data, z_dim=z_dim, layers=layers, beta_1=beta_1, init=init, regularizer_scale=regularizer_scale, spectral=spectral, attention=attention,
    							   			  	  learning_rate_e=learning_rate_e, model_name=model)
	# Train Model.
    losses = contrast_pathology.train(epochs, data_out_path, data, restore, print_epochs=10, n_images=25, checkpoint_every=check_every, report=report)
