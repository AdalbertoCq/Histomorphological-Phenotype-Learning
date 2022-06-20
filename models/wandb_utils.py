

def save_model_config(model, data):
	model_config = dict()

	### Input data variables.
	model_config['image_height']   = model.image_height
	model_config['image_width']    = model.image_width
	model_config['image_channels'] = model.image_channels
	model_config['batch_size']     = model.batch_size

	### Architecture parameters.
	model_config['attention'] = model.attention
	model_config['layers']    = model.layers
	model_config['spectral']  = model.spectral
	model_config['z_dim']     = model.z_dim
	model_config['init']      = model.init

	### Hyperparameters.
	model_config['power_iterations']  = model.power_iterations
	model_config['regularizer_scale'] = model.regularizer_scale
	model_config['learning_rate_e']   = model.learning_rate_e
	model_config['beta_1']            = model.beta_1
	model_config['temperature']       = model.temperature

	### Data augmentation conditions.
	# Spatial transformation.
	model_config['crop']          = model.crop
	model_config['rotation']      = model.rotation
	model_config['flip']          = model.flip
	# Color transofrmation.
	model_config['color_distort'] = model.color_distort
	# Gaussian Blur and Noise.
	model_config['g_blur']        = model.g_blur
	model_config['g_noise']       = model.g_noise
	# Sobel Filter.
	model_config['sobel_filter']  = model.sobel_filter

	model_config['lambda_']       = model.lambda_

	model_config['model_name']    = model.model_name

	model_config['conv_space_out']    = model.conv_space_out.shape.as_list()
	model_config['h_rep_out']         = model.h_rep_out.shape.as_list()
	model_config['z_rep_out']         = model.z_rep_out.shape.as_list()

	model_config['dataset']       = data.dataset

	return model_config


def save_model_config_att(model):
	model_config = dict()

	### Architecture parameters.
	model_config['z_dim']     = model.z_dim
	model_config['att_dim']   = model.att_dim
	model_config['init']      = model.init
	model_config['use_gated'] = model.use_gated

	### Hyperparameters.
	model_config['beta_1']            = model.beta_1
	model_config['beta_2']            = model.beta_2
	model_config['learning_rate']     = model.learning_rate
	model_config['regularizer_scale'] = model.regularizer_scale

	return model_config
