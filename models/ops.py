from models.activations import *
import tensorflow as tf
import numpy as np

# StyleGAN: Apply a weighted input noise to a layer.
def noise_input(inputs, scope):
    with tf.variable_scope('noise_input_%s' % scope):

        # Scale per channel as mentioned in the paper.
        if len(inputs.shape) == 2:
            noise_shape = [tf.shape(inputs)[0], inputs.shape[1]]
        else:
            noise_shape = [tf.shape(inputs)[0], 1, 1, inputs.shape[3]]
        noise = tf.random_normal(noise_shape)
        weights = tf.get_variable('noise_weights', shape=inputs.shape[-1], initializer=tf.contrib.layers.xavier_initializer())

        outputs = inputs + tf.multiply(weights, noise)

    return outputs


def embedding(shape, init='xavier', power_iterations=1, display=True, trainable=True):
    if init=='normal':
        weight_init = tf.initializers.random_normal(stddev=0.02)
    elif init=='orthogonal':
        weight_init = tf.initializers.orthogonal()
    elif init=='glorot_uniform':
        weight_init = tf.initializers.glorot_uniform()
    else:
        weight_init = tf.contrib.layers.xavier_initializer_conv2d()
    embedding = tf.get_variable('Embedding', shape=shape, initializer=weight_init, trainable=trainable)
    # embedding = spectral_normalization(embedding, power_iterations=power_iterations)
    if display:
        print('Emb. Layer:     Output Shape: %s' % (embedding.shape))
    return embedding
    

def style_extract(inputs, latent_dim, spectral, init, regularizer, scope, power_iterations=1):
    with tf.variable_scope('style_extract_%s' % scope) :
        # mean 
        means = tf.reduce_mean(inputs, axis=[1,2], keep_dims=True)
        # std
        stds = tf.sqrt(tf.reduce_mean((inputs-means)**2, axis=[1,2], keep_dims=True))
        means = tf.reduce_mean(means, axis=[1,2])
        stds = tf.reduce_mean(stds, axis=[1,2])
        c = tf.concat([means, stds], axis=1)
        latent = dense(inputs=c, out_dim=latent_dim, use_bias=True, spectral=spectral, power_iterations=power_iterations, init=init, regularizer=regularizer, display=False, scope=1)
        net = ReLU(net)
    return latent
    

def style_extract_2(inputs, latent_dim, spectral, init, regularizer, scope, power_iterations=1):
    with tf.variable_scope('style_extract_%s' % scope) :
        means = tf.reduce_mean(inputs, axis=[1,2], keep_dims=True)
        stds  = tf.sqrt(tf.reduce_mean((inputs-means)**2, axis=[1,2], keep_dims=True))
        means = tf.reduce_mean(means, axis=[1,2])
        stds  = tf.reduce_mean(stds, axis=[1,2])
        comb  = tf.concat([means, stds], axis=1)
        
        net    = dense(inputs=comb, out_dim=latent_dim, use_bias=True, spectral=spectral, power_iterations=power_iterations, init=init, regularizer=regularizer, display=False, scope=1)
        net    = ReLU(net)
        net    = dense(inputs=net, out_dim=int(latent_dim+latent_dim/2.), use_bias=True, spectral=spectral, power_iterations=power_iterations, init=init, regularizer=regularizer, display=False, scope=2)
        net    = ReLU(net)
        latent = dense(inputs=net, out_dim=latent_dim, use_bias=True, spectral=spectral, power_iterations=power_iterations, init=init, regularizer=regularizer, display=False, scope=3)
    return latent


def attention_block(x, scope, spectral=True, init='xavier', regularizer=None, power_iterations=1, display=True):

    batch_size, height, width, channels = x.get_shape().as_list()
    with tf.variable_scope('attention_block_%s' % scope):

        # Global value for all pixels, measures how important is the context for each of them.
        gamma = tf.get_variable('gamma', shape=(1), initializer=tf.constant_initializer(0.0))
        f_g_channels = channels//8

        f = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, 
                          power_iterations=power_iterations, scope=1, display=False)
        g = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, 
                          power_iterations=power_iterations, scope=2, display=False)
        h = convolutional(inputs=x, output_channels=channels    , filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, 
                          power_iterations=power_iterations, scope=3, display=False)

        # Flatten f, g, and h per channel.
        f_flat = tf.reshape(f, shape=tf.stack([tf.shape(x)[0], height*width, channels//8]))
        g_flat = tf.reshape(g, shape=tf.stack([tf.shape(x)[0], height*width, channels//8]))
        h_flat = tf.reshape(h, shape=tf.stack([tf.shape(x)[0], height*width, channels]))

        s = tf.matmul(g_flat, f_flat, transpose_b=True)

        beta = tf.nn.softmax(s)

        o = tf.matmul(beta, h_flat)
        o = tf.reshape(o, shape=tf.stack([tf.shape(x)[0], height, width, channels]))
        y = gamma*o + x

    if display:
        print('Att. Layer:     Scope=%15s Channels %5s Output Shape: %s' % 
            (str(scope)[:14], channels, y.shape))

    return y


def attention_block_2(x, scope, spectral=True, init='xavier', regularizer=None, power_iterations=1, display=True):

    batch_size, height, width, channels = x.get_shape().as_list()
    with tf.variable_scope('attention_block_2_%s' % scope):

        # Global value for all pixels, measures how important is the context for each of them.
        gamma = tf.get_variable('gamma', shape=(1), initializer=tf.constant_initializer(0.0))
        f_g_channels = channels//8
        h_channels = channels//2

        location_n = height*width
        downsampled_n = location_n//4

        f = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, power_iterations=power_iterations, scope=1, display=False)
        f = tf.layers.max_pooling2d(inputs=f, pool_size=[2, 2], strides=2)

        g = convolutional(inputs=x, output_channels=f_g_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, power_iterations=power_iterations, scope=2, display=False)
        
        h = convolutional(inputs=x, output_channels=h_channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, power_iterations=power_iterations, scope=3, display=False)
        h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2)

        # Flatten f, g, and h per channel.
        f_flat = tf.reshape(f, shape=tf.stack([tf.shape(x)[0], downsampled_n, f_g_channels]))
        g_flat = tf.reshape(g, shape=tf.stack([tf.shape(x)[0], location_n, f_g_channels]))
        h_flat = tf.reshape(h, shape=tf.stack([tf.shape(x)[0], downsampled_n, h_channels]))

        attn = tf.matmul(g_flat, f_flat, transpose_b=True)
        attn = tf.nn.softmax(attn)

        o = tf.matmul(attn, h_flat)
        o = tf.reshape(o, shape=tf.stack([tf.shape(x)[0], height, width, channels//2]))
        o = convolutional(inputs=o, output_channels=channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=True, init=init, regularizer=regularizer, power_iterations=power_iterations, scope=4, display=False)
        y = gamma*o + x

    if display:
        print('Atv2 Layer:     Scope=%15s Channels %5s Output Shape: %s' % 
            (str(scope)[:14], channels, y.shape))

    return y


def spectral_normalization(filter, power_iterations):
    # Vector is preserved after each SGD iteration, good performance with power_iter=1 and presenving. 
    # Need to make v trainable, and stop gradient descent to going through this path/variables.
    # Isotropic gaussian. 

    filter_shape = filter.get_shape()
    filter_reshape = tf.reshape(filter, [-1, filter_shape[-1]])
    
    u_shape = (1, filter_shape[-1])
    # If I put trainable = False, I don't need to use tf.stop_gradient()
    u = tf.get_variable('u', shape=u_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(), trainable=False)

    # u_norm, singular_w = power_iteration_method(filter_reshape, u, power_iterations)

    u_norm = u
    v_norm = None

    for i in range(power_iterations):
        v_iter = tf.matmul(u_norm, tf.transpose(filter_reshape))
        v_norm = tf.math.l2_normalize(x=v_iter, epsilon=1e-12)
        u_iter = tf.matmul(v_norm, filter_reshape)
        u_norm = tf.math.l2_normalize(x=u_iter, epsilon=1e-12)

    singular_w = tf.matmul(tf.matmul(v_norm, filter_reshape), tf.transpose(u_norm))[0,0]

    '''
    tf.assign(ref,  value):
        This operation outputs a Tensor that holds the new value of 'ref' after the value has been assigned. 
        This makes it easier to chain operations that need to use the reset value.
        Do the previous iteration and assign u.

    with g.control_dependencies([a, b, c]):
            `d` and `e` will only run after `a`, `b`, and `c` have executed.

    To keep value of u_nom in u?
    If I put this here, the filter won't be use in here until the normalization is done and the value of u_norm kept in u.
    The kernel of the conv it's a variable it self, with its dependencies.
    '''
    with tf.control_dependencies([u.assign(u_norm)]):
        filter_normalized = filter / singular_w
        filter_normalized = tf.reshape(filter_normalized, filter.shape)

    # We can control the normalization before the executing the optimizer by runing the update of all the assign operations 
    # in the variable collection.
    # filter_normalized = filter / singular_w
    # filter_normalized = tf.reshape(filter_normalized, filter.shape)
    # tf.add_to_collection('SPECTRAL_NORM_UPDATE_OPS', u.assign(u_norm))

    '''
    CODE TRACK SINGULAR VALUE OF WEIGHTS.
    filter_normalized_reshape = filter_reshape / singular_w
    s, _, _ = tf.svd(filter_normalized_reshape)
    tf.summary.scalar(filter.name, s[0])
    '''
    
    return filter_normalized


def convolutional(inputs, output_channels, filter_size, stride, padding, conv_type, scope, init='xavier', init_std=None, regularizer=None, data_format='NHWC', output_shape=None, spectral=False, 
                  power_iterations=1, use_bias=True, display=True):
    with tf.variable_scope('conv_layer_%s' % scope):
        # Weight Initlializer.
        if init=='normal':
            if init_std is None:
                weight_init = tf.initializers.random_normal(stddev=0.02)
            else:
                weight_init = tf.initializers.random_normal(stddev=0.02)
        elif init=='orthogonal':
            weight_init = tf.initializers.orthogonal()
        elif init=='glorot_uniform':
            weight_init = tf.initializers.glorot_uniform()
        else:
            weight_init = tf.contrib.layers.xavier_initializer_conv2d()

        # Shapes.
        current_shape = inputs.get_shape()
        input_channels = current_shape[3]
        if 'transpose'in conv_type or 'upscale' in conv_type: filter_shape = (filter_size, filter_size, output_channels, input_channels)   
        else: filter_shape = (filter_size, filter_size, input_channels, output_channels)    

        # Weight and Bias Initialization.
        bias = tf.get_variable(name='bias', shape=[output_channels], initializer=tf.constant_initializer(0.0), trainable=True, dtype=tf.float32) 
        filter = tf.get_variable(name='filter_conv', shape=filter_shape, initializer=weight_init, trainable=True, dtype=tf.float32, regularizer=regularizer)    
        
       # Type of convolutional operation.
        if conv_type == 'upscale':
            output_shape = [tf.shape(inputs)[0], current_shape[1]*2, current_shape[2]*2, output_channels]
            # Weight filter initializer.
            filter = tf.pad(filter, ([1,1], [1,1], [0,0], [0,0]), mode='CONSTANT')
            filter = tf.add_n([filter[1:,1:], filter[:-1,1:], filter[1:,:-1], filter[:-1,:-1]])
            if spectral: filter = spectral_normalization(filter, power_iterations)
            strides = [1, 2, 2, 1]
            output = tf.nn.conv2d_transpose(value=inputs, filter=filter, output_shape=tf.stack(output_shape), strides=strides, padding=padding, data_format=data_format)
            
        elif conv_type == 'downscale':
            # Weight filter initializer.
            filter = tf.pad(filter, ([1,1], [1,1], [0,0], [0,0]), mode='CONSTANT')
            filter = tf.add_n([filter[1:,1:], filter[:-1,1:], filter[1:,:-1], filter[:-1,:-1]])
            if spectral: filter = spectral_normalization(filter, power_iterations)
            strides = [1, 2, 2, 1]
            output = tf.nn.conv2d(input=inputs, filter=filter, strides=strides, padding=padding, data_format=data_format)
            
        elif conv_type == 'transpose':
            output_shape = [tf.shape(inputs)[0], current_shape[1]*stride, current_shape[2]*stride, output_channels]
            strides = [1, stride, stride, 1]
            if spectral: filter = spectral_normalization(filter, power_iterations)
            output = tf.nn.conv2d_transpose(value=inputs, filter=filter, output_shape=tf.stack(output_shape), strides=strides, padding=padding, data_format=data_format)
        
        elif conv_type == 'convolutional':
            strides = [1, stride, stride, 1]
            if spectral: filter = spectral_normalization(filter, power_iterations)
            output = tf.nn.conv2d(input=inputs, filter=filter, strides=strides, padding=padding, data_format=data_format)
        
        if use_bias:
            output = tf.nn.bias_add(output, bias, data_format=data_format)

    if display:
        print('Conv Layer:     Scope=%15s Channels %5s Filter_size=%2s  Stride=%2s Padding=%6s Conv_type=%15s Output Shape: %s' % 
            (str(scope)[:14], output_channels, filter_size, stride, padding, conv_type, output.shape))
    return output


def dense(inputs, out_dim, scope, use_bias=True, spectral=False, power_iterations=1, init='xavier', regularizer=None, display=True):
    if init=='normal':
        weight_init = tf.initializers.random_normal(stddev=0.02)
    elif init=='orthogonal':
        weight_init = tf.initializers.orthogonal()
    elif init=='glorot_uniform':
        weight_init = tf.initializers.glorot_uniform()
    else:
        weight_init = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('dense_layer_%s' % scope):
        in_dim = inputs.get_shape()[-1]
        weights = tf.get_variable('filter_dense', shape=[in_dim, out_dim], dtype=tf.float32, trainable=True, initializer=weight_init, regularizer=regularizer)
        
        if spectral:
            output = tf.matmul(inputs, spectral_normalization(weights, power_iterations))
        else:
            output = tf.matmul(inputs, weights)
        
        if use_bias : 
            bias = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0.0), trainable=True, dtype=tf.float32)
            output = tf.add(output, bias)

    if display:
        print('Dens Layer:     Scope=%15s Channels %5s Output Shape: %s' % 
            (str(scope)[:14], out_dim, output.shape))

    return output


def residual_block(inputs, filter_size, stride, padding, scope, cond_label=None, is_training=True, normalization=None, noise_input_f=False, use_bias=True, spectral=False, activation=None,
                   style_extract_f=False, latent_dim=None, init='xavier', regularizer=None, power_iterations=1, display=True):
    channels = inputs.shape.as_list()[-1]
    with tf.variable_scope('resblock_%s' % scope):
        with tf.variable_scope('part_1'):
            # Convolutional
            net = convolutional(inputs, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, power_iterations=power_iterations, display=False)
            if style_extract_f:
                style_1 = style_extract_2(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)
            if noise_input_f:
               net = noise_input(inputs=net, scope=1)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=1)
            # Activation
            if activation is not None: net = activation(net)
        
        with tf.variable_scope('part_2'):
            # Convolutional
            net = convolutional(net, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, power_iterations=power_iterations, display=False)
            if style_extract_f:
                style_2 = style_extract_2(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
            if noise_input_f:
               net = noise_input(inputs=net, scope=2)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=2)
            # Activation
            if activation is not None: net = activation(net)

        output = inputs + net

        if display:
            print('ResN Layer:     Scope=%15s Channels %5s Filter_size=%2s  Stride=%2s Padding=%6s Conv_type=%15s Output Shape: %s' % 
            (str(scope)[:14], channels, filter_size, stride, padding, 'convolutional', output.shape))
        
        if style_extract_f:
            style = style_1 + style_2
            return output, style

        return output


# Definition of Residual Blocks for dense layers.
def residual_block_dense(inputs, scope, cond_label=None, is_training=True, normalization=None, use_bias=True, spectral=False, activation=None, init='xavier', regularizer=None, power_iterations=1, display=True):
    channels = inputs.shape.as_list()[-1]
    with tf.variable_scope('resblock_dense_%s' % scope):
        with tf.variable_scope('part_1'):
            # Dense
            net = dense(inputs, channels, scope=1, use_bias=use_bias, spectral=spectral, power_iterations=1, init=init, regularizer=regularizer, display=False)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=1)
            # Activation
            if activation is not None: net = activation(net)
            
        with tf.variable_scope('part_2'):
            # Dense
            net = dense(inputs, channels, scope=1, use_bias=use_bias, spectral=spectral, power_iterations=1, init=init, regularizer=regularizer, display=False)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=1)
            # Activation
            if activation is not None: net = activation(net)

        output = inputs + net

        if display:
            print('ResN Layer:     Scope=%15s Channels %5s Output Shape: %s' % (str(scope)[:14], channels, output.shape))
        return output


def residual_block_mod(inputs, filter_size, stride, padding, scope, cond_label=None, is_training=True, normalization=None, noise_input_f=False, use_bias=True, spectral=False, activation=None, init='xavier', regularizer=None, power_iterations=1, display=True):
    channels = inputs.shape.as_list()[-1]
    with tf.variable_scope('resblock_%s' % scope):
        with tf.variable_scope('part_1'):
            # Convolutional
            net = conv_mod(inputs, cond_label, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, power_iterations=power_iterations, display=False)
            if noise_input_f:
               net = noise_input(inputs=net, scope=1)
            # Activation
            if activation is not None: net = activation(net)
            
        with tf.variable_scope('part_2'):
            # Convolutional
            net = conv_mod(net, cond_label, channels, filter_size, stride, padding, 'convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, power_iterations=power_iterations, display=False)
            if noise_input_f:
               net = noise_input(inputs=net, scope=2)
            # Activation
            if activation is not None: net = activation(net)

        output = inputs + net

        if display:
            print('ResN Layer:     Scope=%15s Channels %5s Filter_size=%2s  Stride=%2s Padding=%6s Conv_type=%15s Output Shape: %s' % 
            (str(scope)[:14], channels, filter_size, stride, padding, 'convolutional', output.shape))
        return output


def conv_mod(inputs, label, output_channels, filter_size, stride, padding, conv_type, scope, init='xavier', regularizer=None, data_format='NHWC', output_shape=None, spectral=False, power_iterations=1, display=True):
    with tf.variable_scope('conv_layer_%s' % scope):

        # Weight Initlializer.
        if init=='normal':
            weight_init = tf.initializers.random_normal(stddev=0.02)
        elif init=='orthogonal':
            weight_init = tf.initializers.orthogonal()
        elif init=='glorot_uniform':
            weight_init = tf.initializers.glorot_uniform()
        else:
            weight_init = tf.contrib.layers.xavier_initializer_conv2d()

        # Style NN.
        batch, height, width, input_channels = inputs.shape.as_list()

        inter_dim = int((input_channels+label.shape.as_list()[-1])/2)
        net = dense(inputs=label, out_dim=inter_dim, scope=1, spectral=spectral, display=False)
        net = ReLU(net)
        net = dense(inputs=net, out_dim=int(inter_dim/2), scope='gamma', spectral=spectral, display=False)
        net = ReLU(net)
        style = dense(inputs=net, out_dim=input_channels, scope='beta', spectral=spectral, display=False)

        
        # Filter Shapes.
        if 'convolutional' in conv_type: 

            filter_shape = (filter_size, filter_size, input_channels, output_channels)    
            # Weight and Bias Initialization.
            bias   = tf.get_variable(name='bias',        shape=[output_channels], initializer=tf.constant_initializer(0.0), trainable=True, dtype=tf.float32) 
            filter = tf.get_variable(name='filter_conv', shape=filter_shape,      initializer=weight_init,                  trainable=True, dtype=tf.float32, regularizer=regularizer)    
            
            # print('Input shape:', inputs.shape)
            if spectral: filter = spectral_normalization(filter, power_iterations)

            strides = [1, stride, stride, 1]

            # Add another dimension at the beginnig for batch.
            filter_f = filter[np.newaxis] 

            # Filter Modulation.
            # Add dimensions to latent vector scale input feature map of filters: W'_i_j_k = S_j * w_i_j_k
            filter_f = filter_f * tf.cast(style[:, np.newaxis, np.newaxis, :, np.newaxis], filter.dtype)

            # Demodulate filter:
            # W''_i_j_k = W'_i_j_k / Sqrt(sum_i_k W'_i_j_k +epsilon)
            norm = tf.sqrt(tf.reduce_sum(tf.square(filter_f), axis=[1,2,3]) + 1e-8)
            filter_f = filter_f/tf.cast(norm[:, np.newaxis, np.newaxis, np.newaxis, :], inputs.dtype)

            # Group convolutions
            inputs = tf.reshape(inputs, [1, -1, height, width])
            filter = tf.reshape(tf.transpose(filter_f, [1, 2, 3, 0, 4]), [filter_f.shape[1], filter_f.shape[2], filter_f.shape[3], -1])

            output = tf.nn.conv2d(input=inputs, filter=tf.cast(filter, inputs.dtype), strides=strides, padding=padding, data_format='NCHW')
            output = tf.reshape(output, [-1, output_channels, height, width])
            output = tf.transpose(output, [0, 2, 3, 1])
        
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    if display:
        print('Conv Layer:     Scope=%15s Channels %5s Filter_size=%2s  Stride=%2s Padding=%6s Conv_type=%15s Output Shape: %s' % 
            (str(scope)[:14], output_channels, filter_size, stride, padding, conv_type, output.shape))
    return output 


def lambda_network(x, scope, heads=4, dim_k=16, dim_u=1, m=23, spectral=True, init='xavier', regularizer=None, power_iterations=1, display=True):
    import einops
    def calc_rel_pos(n):
        pos = tf.stack(tf.meshgrid(tf.range(n), tf.range(n), indexing = 'ij'))
        pos = einops.rearrange(pos, 'n i j -> (i j) n')      # [n*n, 2] pos[n] = (i, j)
        rel_pos = pos[None, :] - pos[:, None]                # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
        rel_pos += n - 1                                     # shift value range from [-n+1, n-1] to [0, 2n-2]
        return rel_pos

    batch_size, height, width, channels = x.get_shape().as_list()

    n = height*width
    dim_out = channels

    local_context = False
    if m>0:
        local_context = True
        'Use local contest'

    assert (dim_out%heads)==0, 'must be a clean division between output channels and heads'
    dim_v = dim_out//heads

    with tf.variable_scope('lambda_network_%s' % scope):

        # Queries
        q = convolutional(inputs=x, output_channels=heads*dim_k, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init='normal', init_std=(dim_k*dim_v)**(-0.5),
                          regularizer=None, power_iterations=power_iterations, use_bias=False, scope=1, display=False)
        # Keys
        k = convolutional(inputs=x, output_channels=dim_k*dim_u, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init='normal', init_std=(dim_v)**(-0.5), 
                          regularizer=None, power_iterations=power_iterations, use_bias=False, scope=2, display=False)
        # Values
        v = convolutional(inputs=x, output_channels=dim_v*dim_u, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', spectral=spectral, init='normal', init_std=(dim_v)**(-0.5),
                          regularizer=None, power_iterations=power_iterations, use_bias=False, scope=3, display=False)

        # Batch Normalize q and v.
        q = tf.layers.batch_normalization(q)
        v = tf.layers.batch_normalization(v)

        # Rearrange queries, keys, and values: https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb
        # Instead of tf.transpose, convinient package.
        # batch height width (heads*k)
        q = einops.rearrange(q, 'b hh ww (h k) -> b h k (hh ww)', h=heads)
        k = einops.rearrange(k, 'b hh ww (u k) -> b u k (hh ww)', u=dim_u)
        v = einops.rearrange(v, 'b hh ww (u v) -> b u v (hh ww)', u=dim_u)

        # Normalize keys
        k = tf.nn.softmax(k, axis=-1)

        # Lambda Content.
        lambdaC = tf.einsum('b u k m, b u v m -> b k v', k, v)
        # Compute Content output. 
        yC      = tf.einsum('b h k n, b k v -> b n h v', q, lambdaC)

        # Positional Embeddings
        if local_context:
            v = einops.rearrange(v, 'b u v (hh ww) -> b v hh ww u', hh=height, ww=width)
            #  Conv3D(dim_k, (1, r, r), padding='same')
            lambdaP = tf.layers.conv3d(inputs=v, filters=dim_k, kernel_size=(1,m,m), strides=(1,1,1), padding='same')
            lambdaP = einops.rearrange(lambdaP, 'b v h w k -> b v k (h w)')
            yP      = tf.einsum('b h k n, b v k n -> b n h v', q, lambdaP)

        else:
            # pe_shape = [ n,  n, dim_k, dim_u]
            # positionalEmbedding = tf.get_variable('PositionalEmbedding', shape=pe_shape, initializer=tf.initializers.random_normal(), trainable=True)
            # positionalEmbedding = tf.get_variable('PositionalEmbedding', shape=pe_shape, trainable=True)

            n = height
            rel_length = 2 * n - 1
            rel_pos = calc_rel_pos(n)
            pe_shape = [ n,  n, dim_k, dim_u]
            positionalEmbedding = tf.get_variable('PositionalEmbedding', shape=pe_shape, initializer=tf.initializers.random_normal(), trainable=True)

            # Lambda Positional Encoding.
            rel_pos_emb = tf.gather_nd(positionalEmbedding, rel_pos)
            lambdaP = tf.einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
            # lambdaP = tf.einsum('n m k u, b u v m -> b n k v', positionalEmbedding, v)
            # Compute Positional output. 
            yP      = tf.einsum('b h k n, b n k v -> b n h v', q,           lambdaP)


        y = yC + yP
        y = einops.rearrange(y,'b (hh ww) h v -> b hh ww (h v)', hh=height, ww=width)

    if display:
        print('Lambda Layer:   Scope=%15s Channels %5s Output Shape: %s' %  (str(scope)[:14], channels, y.shape))
    return y


def lambda_residual_block(inputs, filter_size, stride, padding, scope, cond_label=None, is_training=True, normalization=None, noise_input_f=False, spectral=False, activation=None,
                   style_extract_f=False, latent_dim=None, init='xavier', regularizer=None, power_iterations=1, display=True):
    channels = inputs.shape.as_list()[-1]
    with tf.variable_scope('resblock_%s' % scope):

        # Conv 1x1
        with tf.variable_scope('part_1'):
            # Convolutional
            net = convolutional(inputs, channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, 
                                power_iterations=power_iterations, display=False)
            if style_extract_f:
                style_1 = style_extract(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)
            if noise_input_f:
               net = noise_input(inputs=net, scope=1)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=1)
            # Activation
            if activation is not None: net = activation(net)
            
        # Lambda Network
        with tf.variable_scope('part_2'):
            # Convolutional
            net = lambda_network(net, heads=4, dim_k=16, dim_u=1, m=23, spectral=spectral, init=init, regularizer=regularizer, power_iterations=power_iterations, display=False, scope=1)
            if style_extract_f:
                style_2 = style_extract(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
            if noise_input_f:
               net = noise_input(inputs=net, scope=2)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=2)
            # Activation
            if activation is not None: net = activation(net)

        # Conv 1x1
        with tf.variable_scope('part_3'):
            net = convolutional(net, channels, filter_size=1, stride=1, padding='SAME', conv_type='convolutional', scope=1, spectral=spectral, init=init, regularizer=regularizer, 
                                power_iterations=power_iterations, display=False)
            if style_extract_f:
                style_2 = style_extract(inputs=net, latent_dim=latent_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
            if noise_input_f:
               net = noise_input(inputs=net, scope=2)
            # Normalization
            if normalization is not None: 
                net = normalization(inputs=net, training=is_training, c=cond_label, spectral=spectral, scope=2)
            

        output = inputs + net

        # Activation
        if activation is not None: output = activation(output)

        if display:
            print('LambResN Layer: Scope=%15s Channels %5s Filter_size=%2s  Stride=%2s Padding=%6s Conv_type=%15s Output Shape: %s' % 
            (str(scope)[:14], channels, filter_size, stride, padding, 'convolutional', output.shape))
        
        if style_extract_f:
            style = style_1 + style_2
            return output, style

        return output


