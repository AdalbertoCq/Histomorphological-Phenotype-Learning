from skimage import color, io
import tensorflow as tf
import numpy as np
import functools
import random

# Random sampling to recide if the transformation is applied.
def random_apply(func, p, x):
    return tf.cond(tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


############### COLOR ###############
# Two version of random change to brightness: Addition/Multiplicative.
def random_brightness(image, max_delta, impl='simclrv2'):
    if impl == 'simclrv2':
        factor = tf.random_uniform([], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
        image = image * factor
    elif impl == 'simclrv1':    
        image = tf.image.random_brightness(image, max_delta=max_delta)
    else:
        raise ValueError('Unknown impl {} for random brightness.'.format(impl))
    return image

# Random color transformations: Bright, Contrast, Saturation, and Hue.
def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0, impl='simclrv2'):
    with tf.name_scope('distort_color'):
        def apply_transform(i, x):
            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness, impl=impl)
            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)
            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)
            x = tf.cond(tf.less(i, 2),
                        lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                        lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
            return x

        perm = tf.random_shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0., 1.)
        return image

# No random color transformations: 1st Bright, 2nd Contrast, 3rd Saturation, and 4th Hue.
def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0, impl='simclrv2'):
    with tf.name_scope('distort_color'):
        def apply_transform(i, x, brightness, contrast, saturation, hue):
            if brightness != 0 and i == 0:
                x = random_brightness(x, max_delta=brightness, impl=impl)
            elif contrast != 0 and i == 1:
                x = tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
            elif saturation != 0 and i == 2:
                x = tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)
            elif hue != 0:
                x = tf.image.random_hue(x, max_delta=hue)
            return x

        for i in range(4):
            image = apply_transform(i, image, brightness, contrast, saturation, hue)
            image = tf.clip_by_value(image, 0., 1.)
        return image
    
# Color transformation on image: Random or not random order.
def color_jitter(image, strength, random_order=True, impl='simclrv2'):
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue, impl=impl)
    else:
        return color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl=impl)
    
# Image RGB to Grayscale.
def to_grayscale(image, keep_channels=True):
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image

# Color transformation on image.
def random_color_jitter(image, p=1.0, impl='simclrv2'):
    def transformation(image):
        color_jitter_t = functools.partial(color_jitter, strength=0.5, impl=impl)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(transformation, p=p, x=image)

# Color transformation on image.
def random_color_jitter_1p0(image, p=1.0, impl='simclrv2'):
    def transformation(image):
        color_jitter_t = functools.partial(color_jitter, strength=1.0, impl=impl)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(transformation, p=p, x=image)


############### SPATIAL: Cropping and Resizing ############### 

# Crop.
def distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), max_attempts=100, scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(shape, bounding_boxes=bbox, min_object_covered=min_object_covered, aspect_ratio_range=aspect_ratio_range,
                                                                           area_range=area_range, max_attempts=max_attempts, use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)

        return image

# Crop and resize image.
def crop_and_resize(image, height, width, area_range):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(image, bbox, min_object_covered=0.1, aspect_ratio_range=(3./4*aspect_ratio, 4./3.*aspect_ratio), area_range=area_range, max_attempts=100, scope=None)
    return tf.image.resize_bicubic([image], [height, width])[0]

# Random crop and resize.
def random_crop_and_resize(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.08, 1.0))
        return images

    return random_apply(func=transformation, p=prob, x=image)


def random_crop_and_resize_p075(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.75, 1.0))
        return images

    return random_apply(func=transformation, p=prob, x=image)


# Random crop and resize SwAV Global view.
def random_crop_and_resize_global(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.14, 1.0))
        return images
    return random_apply(func=transformation, p=prob, x=image)

# Random crop and resize SwAV Local view.
def random_crop_and_resize_local(image, prob=1.0):
    height, width, channels = image.shape.as_list()
    def transformation(image):
        images = crop_and_resize(image=image, height=height, width=width, area_range=(0.05, 0.14))
        return images
    return random_apply(func=transformation, p=prob, x=image)


############### SPATIAL: Cutout ##############################


############### SPATIAL: Rotation and Flipping ############### 

def random_rotate(image, p=0.5):
    return random_apply(tf.image.rot90, p, image)

def random_flip(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


############### BLUR ############### 

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    radius = tf.to_int32(kernel_size / 2)
    kernel_size = radius * 2 + 1
    x = tf.to_float(tf.range(-radius, radius + 1))
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.to_float(sigma), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(image,   blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

def random_blur(image, p=0.5):
    height, width, channels = image.shape.as_list()
    del width
    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(image, kernel_size=height//10, sigma=sigma, padding='SAME')
    return random_apply(_transform, p=p, x=image)


############### GAUSSIAN NOISE ############### 

# Adds gaussian noise to an image.
def add_gaussian_noise(image):
    # image must be scaled in [0, 1]
    with tf.name_scope('Add_gaussian_noise'):
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=(50)/(255), dtype=tf.float32)
        noise_img = image + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img

# Adds gaussian noise randomly to image.
def random_gaussian_noise(image, p=0.5):
    return random_apply(add_gaussian_noise, p, image)


############### SOBEL FILTER ############### 

def random_apply_sobel(func, p, x):
    return tf.cond(tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)), lambda: tf.reduce_mean(func(x), axis=-1), lambda: x)

def random_sobel_filter(image, p=0.5):
    height, width, channels = image.shape.as_list()
    image = tf.reshape(image, (1, height, width, channels))
    applied = random_apply_sobel(tf.image.sobel_edges, p, image)
    applied = tf.reduce_mean(applied, axis=0)
    return applied


############### DATA AUG WRAPPER ############### 
def data_augmentation(images, crop, rotation, flip, g_blur, g_noise, color_distort, sobel_filter, img_size, num_channels):
    images_trans = images
    # Spatial transformations.
    if crop:
        images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    if rotation:
        images_trans = tf.map_fn(random_rotate, images_trans)
    if flip:
        images_trans = random_flip(images_trans)
    # Gaussian blur and noise transformations.    
    if g_blur:
        images_trans = tf.map_fn(random_blur, images_trans)
    if g_noise:
        images_trans = tf.map_fn(random_gaussian_noise, images_trans)
    # Color distorsions. 
    if color_distort:
        images_trans = tf.map_fn(random_color_jitter, images_trans)         
    # Sobel filter.
    if sobel_filter:
        images_trans = tf.map_fn(random_sobel_filter, images_trans)
    # Make sure the image batch is in the right format.
    images_trans = tf.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = tf.clip_by_value(images_trans, 0., 1.)
    
    return images_trans


############### DATA AUG WRAPPER INVARIABILITY STAIN COLOR ############### 
def get_mean_std_patches(imgs):
    means_ch_0 = list()
    means_ch_1 = list()
    means_ch_2 = list()
    
    stds_ch_0 = list()
    stds_ch_1 = list()
    stds_ch_2 = list()
    
    for i in range(imgs.shape[0]):
        if np.max(imgs[i]) <= 1:
            arr = np.array(imgs[i]* 255, dtype=np.uint8)
        else:
            arr = np.array(imgs[i])
        lab = lab = color.rgb2lab(arr)
        means_ch_0.append(np.mean(lab[:,:,0]))
        means_ch_1.append(np.mean(lab[:,:,1]))
        means_ch_2.append(np.mean(lab[:,:,2]))

        stds_ch_0.append(np.std(lab[:,:,0]))
        stds_ch_1.append(np.std(lab[:,:,1]))
        stds_ch_2.append(np.std(lab[:,:,2]))
        
    return [means_ch_0, means_ch_1, means_ch_2], [stds_ch_0, stds_ch_1, stds_ch_2]

def random_renorm(imgs, means, stds):
    
    batch_size, height, width, channels = imgs.shape
    processed_img = np.zeros((batch_size, height, width, channels), dtype=np.uint8)
    
    random_indeces = list(range(batch_size))
    random.shuffle(random_indeces)
    
    for j in range(batch_size):
        if np.max(imgs[j]) <= 1:
            arr = np.array(imgs[j]* 255, dtype=np.uint8)
        else:
            arr = np.array(imgs[j])
        lab = color.rgb2lab(arr)
        p = random_indeces[j]
        
        # Each channel 
        for i in range(3):
            
            new_mean = means[i][p]
            new_std  = stds[i][p]
            
            t_mean = np.mean(lab[:,:,i])
            t_std  = np.std(lab[:,:,i])
            tmp = ( (lab[:,:,i] - t_mean) * (new_std / t_std) ) + new_mean
            if i == 0:
                tmp[tmp<0] = 0
                tmp[tmp>100] = 100
                lab[:,:,i] = tmp
            else:
                tmp[tmp<-128] = 128
                tmp[tmp>127] = 127
                lab[:,:,i] = tmp
                
        processed_img[j] = (color.lab2rgb(lab) * 255).astype(np.uint8)
        
    return processed_img/255.

def random_batch_renormalization(batch_images):
    means, stds  = get_mean_std_patches(imgs=batch_images)
    proc_images  = random_renorm(imgs=batch_images, means=means, stds=stds)
    return proc_images

def tf_wrapper_rb_stain(batch_images):
    out_trans = tf.py_function(random_batch_renormalization, [batch_images], tf.float32)
    return out_trans

def data_augmentation_stain_variability(images, img_size, num_channels):
    images_trans = images
    # images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    # images_trans = tf.map_fn(random_rotate, images_trans)
    # images_trans = random_flip(images_trans)
    images_trans = tf_wrapper_rb_stain(images_trans)

    # Make sure the image batch is in the right format.
    images_trans = tf.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = tf.clip_by_value(images_trans, 0., 1.)
    return images_trans

def data_augmentation_color(images, img_size, num_channels):
    images_trans = images
    images_trans = tf.map_fn(random_crop_and_resize, images_trans)
    images_trans = tf.map_fn(random_rotate, images_trans)
    images_trans = random_flip(images_trans)
    images_trans = tf.map_fn(random_color_jitter, images_trans)         

    # Make sure the image batch is in the right format.
    images_trans = tf.reshape(images_trans, [-1, img_size, img_size, num_channels])
    images_trans = tf.clip_by_value(images_trans, 0., 1.)
    return images_trans