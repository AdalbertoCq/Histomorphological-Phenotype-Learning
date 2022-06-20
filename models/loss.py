from models.regularizers import *
from models.activations import *
import tensorflow as tf
import numpy as np
import sys

def realness_loss(features_fake, features_real, anchor_0, anchor_1, v_max=1., v_min=-1., relativistic=True, discriminator=None, real_images=None, fake_images=None, init=None, gp_coeff=None, dis_name='discriminator'):

    # Probabilities over realness features.
    prob_real = tf.nn.softmax(features_real, axis=-1)
    prob_fake = tf.nn.softmax(features_fake, axis=-1)

    # Discriminator loss.
    anchor_real = anchor_1
    anchor_fake = anchor_0

    # Real data term - Positive skew for anchor.
    skewed_anchor = anchor_real
    loss_dis_real = tf.reduce_mean(tf.reduce_sum(-(skewed_anchor*tf.log(prob_real+1e-16)), axis=-1))
    # Fake data term - Negative skew for anchor.
    skewed_anchor = anchor_fake
    loss_dis_fake = tf.reduce_mean(tf.reduce_sum(-(skewed_anchor*tf.log(prob_fake+1e-16)), axis=-1))
    loss_dis = loss_dis_real + loss_dis_fake

    # Generator loss.
    # Fake data term - Negative skew for anchor.
    skewed_anchor = anchor_fake
    loss_gen_fake = -tf.reduce_mean(tf.reduce_sum(-(skewed_anchor*tf.log(prob_fake+1e-16)), axis=-1))

    # Relative term, comparison between fake probability realness and reference of real, either prob. realness or anchor.
    if relativistic:
        # Relativistic term, use real features as anchor.
        skewed_anchor = prob_real
        loss_gen_real = tf.reduce_mean(tf.reduce_sum(-(skewed_anchor*tf.log(prob_fake+1e-16)), axis=-1))
    else:
        # Positive skew for anchor.
        skewed_anchor = anchor_real
        loss_gen_real = tf.reduce_mean(tf.reduce_sum(-(skewed_anchor*tf.log(prob_fake+1e-16)), axis=-1))

    loss_gen = loss_gen_fake + loss_gen_real

    epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
    x_gp = real_images*(1-epsilon) + fake_images*epsilon

    out = discriminator(x_gp, reuse=True, init=init, name=dis_name)
    logits_gp = out[1]

    # Calculating Gradient Penalty.
    grad_gp = tf.gradients(logits_gp, x_gp)
    l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
    grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))
    loss_dis +=  (gp_coeff*grad_penalty)

    return loss_dis, loss_gen


def losses(loss_type, output_fake, output_real, logits_fake, logits_real, label=None, real_images=None, fake_images=None, encoder=None, discriminator=None, init=None, gp_coeff=None, hard=None,
           top_k_samples=None, display=True, enc_name='discriminator', dis_name='discriminator'):
    
    # Variable to track which loss function is actually used.
    loss_print = ''
    if 'relativistic' in loss_type:
        logits_diff_real_fake = logits_real - tf.reduce_mean(logits_fake, axis=0, keepdims=True)
        logits_diff_fake_real = logits_fake - tf.reduce_mean(logits_real, axis=0, keepdims=True)
        loss_print += 'relativistic '

        if 'standard' in loss_type:

            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.ones_like(logits_fake)))
            loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.zeros_like(logits_fake)))
            loss_dis = loss_dis_real + loss_dis_fake

            # Generator loss.
            if top_k_samples is not None:
                logits_diff_fake_real_top =  tf.math.top_k(input= tf.reshape( logits_diff_fake_real, (1, -1)), k=top_k_samples, sorted=False, name='top_performance_samples_max')[0]
                logits_diff_real_fake_min = -tf.math.top_k(input= tf.reshape(-logits_diff_real_fake, (1, -1)), k=top_k_samples, sorted=False, name='top_performance_samples_min')[0]
                logits_diff_fake_real_top = tf.reshape(logits_diff_fake_real_top, (-1,1))
                logits_diff_real_fake_min = tf.reshape(logits_diff_real_fake_min, (-1,1))
                loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real_top, labels=tf.ones_like(logits_diff_fake_real_top)))
                loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake_min, labels=tf.zeros_like(logits_diff_real_fake_min)))
            else:
                loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.ones_like(logits_fake)))
                loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.zeros_like(logits_fake)))
            loss_gen = loss_gen_real + loss_gen_fake

            loss_print += 'standard '

        elif 'least square' in loss_type:
            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.square(logits_diff_real_fake-1.0))
            loss_dis_fake = tf.reduce_mean(tf.square(logits_diff_fake_real+1.0))
            loss_dis = loss_dis_real + loss_dis_fake

            # Generator loss.
            loss_gen_real = tf.reduce_mean(tf.square(logits_diff_fake_real-1.0))
            loss_gen_fake = tf.reduce_mean(tf.square(logits_diff_real_fake+1.0))
            loss_gen = loss_gen_real + loss_gen_fake
            loss_print += 'least square '

        elif 'gradient penalty' in loss_type:
            # Calculating X hat.
            epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
            x_gp = real_images*(1-epsilon) + fake_images*epsilon

            if encoder is None:
                if label is not None: 
                    if hard is not None:
                        out = discriminator(x_gp, hard=hard, label_input=label, reuse=True, init=init, name=dis_name)
                    else:
                        out = discriminator(x_gp, label_input=label, reuse=True, init=init, name=dis_name)
                else:
                    if hard is not None:
                        out = discriminator(x_gp, hard=hard, reuse=True, init=init, name=dis_name)
                    else:
                        out = discriminator(x_gp, reuse=True, init=init, name=dis_name)
            else:
                out_1 = encoder(x_gp, True, is_train=True, init=init, name=enc_name)
                out = discriminator(out_1, reuse=True, init=init, name=dis_name)
            
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))

            # Discriminator loss.
            loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.ones_like(logits_fake)))
            loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.zeros_like(logits_fake)))
            loss_dis = loss_dis_real + loss_dis_fake + (gp_coeff*grad_penalty)

            # Generator loss. 
            if top_k_samples is not None:

                # For fake images, we want gradients from fake samples where the critic says that they are more realistic than reals. 
                # Fake - Avg(real)
                ind_logits_diff_fake_real = tf.math.top_k(input=tf.reshape(-tf.abs(logits_diff_fake_real), (1, -1)), k=top_k_samples, sorted=False, name='top_performance_samples_fake_real').indices
                # For real images,
                # Real - Avg(fake) 
                ind_logits_diff_real_fake = tf.math.top_k(input=tf.reshape(-tf.abs(logits_diff_real_fake), (1, -1)), k=top_k_samples, sorted=False, name='top_performance_samples_real_fake').indices
                
                n = 2*top_k_samples
                mask_fake_real  = tf.reshape(tf.reduce_sum(tf.one_hot(ind_logits_diff_fake_real, n), axis=1), (-1,1))
                mask_real_fake  = tf.reshape(tf.reduce_sum(tf.one_hot(ind_logits_diff_real_fake, n), axis=1), (-1,1))

                top_k_logits_fake_real = logits_diff_fake_real * mask_fake_real
                top_k_logits_real_fake = logits_diff_real_fake * mask_real_fake

                loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=top_k_logits_fake_real, labels=tf.ones_like(top_k_logits_fake_real)))
                loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=top_k_logits_real_fake, labels=tf.zeros_like(top_k_logits_real_fake)))
            else:
                loss_gen_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_fake_real, labels=tf.ones_like(logits_fake)))
                loss_gen_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_diff_real_fake, labels=tf.zeros_like(logits_fake)))
            loss_gen = loss_gen_real + loss_gen_fake
            loss_print += 'gradient penalty '   

    elif 'standard' in loss_type:

        loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(output_fake)))
        loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(output_fake)*0.9))
        
        if 'gradient penalty' in loss_type:
            # Calculating X hat.
            epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
            x_gp = real_images*(1-epsilon) + fake_images*epsilon

                
            if encoder is None:
                if label is not None: 
                    out = discriminator(x_gp, label_input=label, reuse=True, init=init, name=dis_name)
                else:
                    out = discriminator(x_gp, reuse=True, init=init, name=dis_name)
            else:
                out_1 = encoder(x_gp, True, is_train=True, init=init, name=enc_name)
                out = discriminator(out_1, reuse=True, init=init, name=dis_name)
            
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))

            # Discriminator loss. Uses hinge loss on discriminator.
            loss_dis = loss_dis_fake + loss_dis_real + (gp_coeff*grad_penalty)
        
        else:
            # Discriminator loss. Uses hinge loss on discriminator.
            loss_dis = loss_dis_fake + loss_dis_real 

        # Generator loss.
        # This is where we implement -log[D(G(z))] instead log[1-D(G(z))].
        # Recall the implementation of cross-entropy, sign already in. 
        loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(output_fake)))
        loss_print += 'standard '

    elif 'least square' in loss_type:       
        # Discriminator loss.
        loss_dis_fake = tf.reduce_mean(tf.square(output_fake))
        loss_dis_real = tf.reduce_mean(tf.square(output_real-1.0))
        loss_dis = 0.5*(loss_dis_fake + loss_dis_real)

        # Generator loss.
        loss_gen = 0.5*tf.reduce_mean(tf.square(output_fake-1.0))
        loss_print += 'least square '

    elif 'wasserstein distance' in loss_type:
        # Discriminator loss.
        loss_dis_real = tf.reduce_mean(logits_real)
        loss_dis_fake = tf.reduce_mean(logits_fake)
        loss_dis = -loss_dis_real + loss_dis_fake
        loss_print += 'wasserstein distance '

        # Generator loss.
        loss_gen = -loss_dis_fake
        if 'gradient penalty' in loss_type:
            # Calculating X hat.
            epsilon = tf.random.uniform(shape=tf.stack([tf.shape(real_images)[0], 1, 1, 1]), minval=0.0, maxval=1.0, dtype=tf.float32, name='epsilon')
            x_gp = real_images*(1-epsilon) + fake_images*epsilon
            out = discriminator(x_gp, True, init=init)
            logits_gp = out[1]

            # Calculating Gradient Penalty.
            grad_gp = tf.gradients(logits_gp, x_gp)
            l2_grad_gp = tf.sqrt(tf.reduce_sum(tf.square(grad_gp), axis=[1, 2, 3]))
            grad_penalty= tf.reduce_sum(tf.square(l2_grad_gp-1.0))
            loss_dis += (gp_coeff*grad_penalty)
            loss_print += 'gradient penalty '
        
    elif 'hinge' in loss_type:
        loss_dis_real = tf.reduce_mean(tf.maximum(tf.zeros_like(logits_real), tf.ones_like(logits_real) - logits_real))
        loss_dis_fake = tf.reduce_mean(tf.maximum(tf.zeros_like(logits_real), tf.ones_like(logits_real) + logits_fake))
        loss_dis = loss_dis_fake + loss_dis_real

        '''
        tf.reduce_mean(- tf.minimum(0., -1.0 + real_logits))
        tf.reduce_mean(  tf.maximum(0.,  1.0 - logits_real))
        
        tf.reduce_mean(- tf.minimum(0., -1.0 - fake_logits))
        tf.reduce_mean(  tf.maximum(0.,  1.0 + logits_fake))
        '''

        
        loss_gen = -tf.reduce_mean(logits_fake)
        loss_print += 'hinge '

    else:
        print('Loss: Loss %s not defined' % loss_type)
        sys.exit(1)

    if display:
        print('[Loss] Loss %s' % loss_print)
        
    return loss_dis, loss_gen
    

def reconstruction_loss(z_dim, w_latent_ref, w_latent):
    # MSE on Reference W latent and reconstruction, normalized by the dimensionality of the z vector.
    latent_recon_error = tf.reduce_mean(tf.square(w_latent_ref-w_latent), axis=[-1])
    latent_recon_error = tf.reduce_sum(latent_recon_error, axis=[-1])
    loss_enc = tf.reduce_mean(latent_recon_error)/float(z_dim)
    return loss_enc


def cross_entropy_class(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,  logits=logits, axis=-1))


###### Self-Representation Learning ###### 

def cosine_similarity(a, b):
    num = tf.matmul(a, b, transpose_b=True)
    a_mod = tf.sqrt(tf.reduce_sum(tf.matmul(a, a, transpose_b=True), keepdims=False))
    b_mod = tf.sqrt(tf.reduce_sum(tf.matmul(b, b, transpose_b=True), keepdims=False))
    den = a_mod * b_mod
    return num/den


def consitency_loss(feature_aug, features):
    l2_loss = tf.square(feature_aug-features)
    l2_loss = tf.reduce_sum(l2_loss, axis=[-1])/2
    l2_loss = tf.reduce_mean(l2_loss)
    return l2_loss


def contrastive_loss(a, b, batch_size, temperature=1.0, weights=1.0):

    # Masks for same sample in batch.
    masks  = tf.one_hot(tf.range(batch_size), batch_size)
    labels = tf.one_hot(tf.range(batch_size), batch_size*2)
    
    # Logits:
    logits_aa = tf.matmul(a, a, transpose_b=True)/temperature
    logits_aa = logits_aa - masks * 1e9
    
    logits_bb = tf.matmul(b, b, transpose_b=True)/temperature
    logits_bb = logits_bb - masks * 1e9
    
    logits_ab = tf.matmul(a, b, transpose_b=True)/temperature
    logits_ba = tf.matmul(b, a, transpose_b=True)/temperature
    
    loss_a = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], axis=1), weights=weights)
    loss_b = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], axis=1), weights=weights)
    
    loss = loss_a + loss_b

    return loss, logits_ab, labels


def byol_loss(prediction, z_rep):
    p = tf.math.l2_normalize(prediction, axis=1)
    z = tf.math.l2_normalize(z_rep, axis=1)
    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    loss = 2 - 2 * tf.reduce_mean(similarities)
    return loss

def cross_correlation_loss(z_a, z_b, lambda_):
    def normalize_repr(z):
        z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
        return z_norm

    def off_diagonal(x):
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
        return tf.reshape(off_diagonals, [-1])

    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Batch normalize representations
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    inv_term    = tf.linalg.diag_part(c)
    on_diag     = tf.reduce_sum(tf.pow(1-inv_term, 2))

    redred_term = (tf.ones_like(c)-tf.eye(repr_dim))*c
    off_diag    = tf.reduce_sum(tf.pow(redred_term, 2))

    loss = on_diag + (lambda_ * off_diag)
    return loss


