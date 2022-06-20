import tensorflow as tf

def optimizer(beta_1, loss_gen, loss_dis, loss_type, learning_rate_input_d, learning_rate_input_g=None, learning_rate_input_e=None, beta_2=None, clipping=None, display=True,
                gen_name='generator', dis_name='discriminator', mapping_name='mapping_', encoder_name='encoder', gpus=[0]):
    
    # Gather variables for each network system, mapping network is included in the generator
    trainable_variables = tf.trainable_variables()
    generator_variables = [variable for variable in trainable_variables if variable.name.startswith(gen_name)]
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith(dis_name)]
    mapping_variables = [variable for variable in trainable_variables if variable.name.startswith(mapping_name)]
    encoder_variables = [variable for variable in trainable_variables if variable.name.startswith(encoder_name)]
    if len(mapping_variables) != 0:
        generator_variables.extend(mapping_variables)

    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Wasserstein distance with gradient penalty and Hinge loss.
        if ('wasserstein distance' in loss_type and 'gradient penalty' in loss_type) or ('hinge' in loss_type):
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type

        # Wasserstein distance loss.
        elif 'wasserstein distance' in loss_type and 'gradient penalty' not in loss_type:
            # Weight Clipping on Discriminator, this is done to ensure the Lipschitz constrain.
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            dis_weight_clipping = [value.assign(tf.clip_by_value(value, -clipping, clipping)) for value in discriminator_variables]
            train_discriminator = tf.group(*[train_discriminator, dis_weight_clipping])
            train_generator = tf.train.AdamOptimizer(learning_rate_input_g, beta_1, beta_2).minimize(loss_gen, var_list=generator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        
        # Standard, Least square, and standard relativistic loss.
        elif 'standard' in loss_type or 'least square' in loss_type or 'relativistic' in loss_type:
            with tf.device('/gpu:%s' % gpus[0]):
                train_encoder = None
                train_discriminator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=discriminator_variables) 
            other_gpu = gpus[0]
            
            if len(gpus) > 1:
                other_gpu = gpus[1]
            with tf.device('/gpu:%s' % other_gpu):
                train_generator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_g, beta1=beta_1).minimize(loss_gen, var_list=generator_variables)
                if len(encoder_variables) != 0 and learning_rate_input_e is not None: train_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate_input_e, beta1=beta_1).minimize(loss_gen, var_list=encoder_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        
        else:
            print('Optimizer: Loss %s not defined' % loss_type)
            exit(1)

        if display:
            print('[Optimizer] Loss %s' % optimizer_print)
            print()
    
    if len(encoder_variables) != 0:
        return train_discriminator, train_generator, train_encoder
    else:
        return train_discriminator, train_generator


def contrastive_optimizer(learning_rate_input_d, beta_1, loss_contrastive, constrastive_dis_name='contrastive_discriminator'):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Quick dirty optimizer for Encoder.
        trainable_variables = tf.trainable_variables()
        discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith(constrastive_dis_name)]
        train_discriminator_contrastive = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_contrastive, var_list=discriminator_variables)
    return train_discriminator_contrastive


def encoder_optimizer(loss_enc, learning_rate_input_e, beta_1, encoder_name='encoder'):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Quick dirty optimizer for Encoder.
        trainable_variables = tf.trainable_variables()
        encoder_variables = [variable for variable in trainable_variables if variable.name.startswith(encoder_name)]
        train_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate_input_e, beta1=beta_1).minimize(loss_enc, var_list=encoder_variables)
    return train_encoder

def vae_gan_optimizer(beta_1, loss_prior, loss_dist_likel, loss_gen, loss_dis, loss_type, learning_rate_input_g, learning_rate_input_d, beta_2=None, clipping=None, 
                      display=True, gamma=1):
    trainable_variables = tf.trainable_variables()
    encoder_variables = [variable for variable in trainable_variables if variable.name.startswith('encoder')]
    generator_decoder_variables = [variable for variable in trainable_variables if variable.name.startswith('generator_decoder')]
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith('discriminator')]

    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if 'wasserstein distance' in loss_type and 'gradient penalty' in loss_type:
            train_encoder = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_prior+loss_dist_likel, var_list=encoder_variables)
            train_gen_decod = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dist_likel+loss_gen, var_list=generator_decoder_variables)
            train_discriminator = tf.train.AdamOptimizer(learning_rate_input_d, beta_1, beta_2).minimize(loss_dis, var_list=discriminator_variables)
            optimizer_print += 'Wasserstein Distance Gradient penalty - AdamOptimizer'
        elif 'relativistic' in loss_type:
            train_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_prior+loss_dist_likel, var_list=encoder_variables)
            train_gen_decod = tf.train.AdamOptimizer(learning_rate=learning_rate_input_g, beta1=beta_1).minimize((gamma*loss_dist_likel)+loss_gen, var_list=generator_decoder_variables)
            train_discriminator = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=discriminator_variables)
            optimizer_print += '%s - AdamOptimizer' % loss_type
        else:
            print('Loss %s not defined' % loss_type)
            exit(1)

        if display:
            print('Optimizer: %s' % optimizer_print)
            print()
            
    return train_encoder, train_gen_decod, train_discriminator


def optimizer_alae(loss_dis, loss_gen, loss_enc, loss_type, learning_rate_input_d, learning_rate_input_g, learning_rate_input_e, beta_1, beta_2=None, display=True,
                gen_name='generator', dis_name='discriminator', mapping_name='mapping_', encoder_name='encoder', gpus=[0]):
    
    # Gather variables for each network system, mapping network is included in the generator
    trainable_variables = tf.trainable_variables()

    mapping_variables = [variable for variable in trainable_variables if variable.name.startswith(mapping_name)]
    generator_variables = [variable for variable in trainable_variables if variable.name.startswith(gen_name)]
    
    encoder_variables = [variable for variable in trainable_variables if variable.name.startswith(encoder_name)]
    discriminator_variables = [variable for variable in trainable_variables if variable.name.startswith(dis_name)]
    
    # Optimizer variable to track with optimizer is actually used.
    optimizer_print = ''

    # Handling Batch Normalization.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Wasserstein distance with gradient penalty and Hinge loss.
        if 'standard' in loss_type or 'least square' in loss_type or 'relativistic' in loss_type:
            e_d_variables = list()
            e_d_variables.extend(encoder_variables)
            e_d_variables.extend(discriminator_variables)
            train_e_d = tf.train.AdamOptimizer(learning_rate=learning_rate_input_d, beta1=beta_1).minimize(loss_dis, var_list=e_d_variables) 
            
            m_g_variables = list()
            m_g_variables.extend(mapping_variables)
            m_g_variables.extend(generator_variables)
            train_m_g = tf.train.AdamOptimizer(learning_rate=learning_rate_input_g, beta1=beta_1).minimize(loss_gen, var_list=m_g_variables)            

            e_g_variables = list()
            e_g_variables.extend(generator_variables)
            e_g_variables.extend(encoder_variables)
            train_g_e = tf.train.AdamOptimizer(learning_rate=learning_rate_input_e, beta1=beta_1).minimize(loss_enc, var_list=e_g_variables)

            optimizer_print += '%s - AdamOptimizer' % loss_type
    
        else:
            print('Optimizer: Loss %s not defined' % loss_type)
            exit(1)

    if display:
        print('[Optimizer] Loss %s' % optimizer_print)
        print()
    
    return train_e_d, train_m_g, train_g_e


def optimizer_contrastive_accumulated_gradients(loss, trainable_variables, learning_rate):
    # Gather trainable variables, create variables for accumulated gradients, assign zero value to accumulated gradients. 
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_variables]                

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Optimizer Initialization
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)                                                                                                   

        # Operation to apply zeros value to accumulated gradients.
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        # Compute gradients w.r.t loss. 
        gvs = opt.compute_gradients(loss, trainable_variables)
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

        # Applied accumulated gradients. 
        train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])  

    return zero_ops, accum_ops, train_step

