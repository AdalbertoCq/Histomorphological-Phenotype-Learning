from PIL import Image
import numpy as np
import os

from models.utils import *
from models.visualization.utils import *

# Reset coordinates.
def return_coor(i):
    if i==0:
        x_y = np.array([0,0])
    elif i==1:
        x_y = np.array([1,0])
    elif i==2:
        x_y = np.array([0,1])
    elif i==3:
        x_y = np.array([1,1])

    elif i==4:
        x_y = np.array([2,0])
    elif i==5:
        x_y = np.array([3,0])
    elif i==6:
        x_y = np.array([2,1])
    elif i==7:
        x_y = np.array([3,1])

    elif i==8:
        x_y = np.array([0,2])
    elif i==9:
        x_y = np.array([1,2])
    elif i==10:
        x_y = np.array([0,3])
    elif i==11:
        x_y = np.array([1,3])

    elif i==12:
        x_y = np.array([2,2])
    elif i==13:
        x_y = np.array([3,2])
    elif i==14:
        x_y = np.array([2,3])
    elif i==15:
        x_y = np.array([3,3])
    return x_y

# Get tile information.
def get_x_y(tile_info):
    if '.' in str(tile_info):
        string = tile_info.split('.')[0]
    else:
        string = str(tile_info)
    x, y   = string.split('_')
    return int(x),int(y)


def multimag_wsi_process_5x(label, probability, slide, tiles, slide_indices, oriind_5x, orig_set, img_5x, weights_5x, slides_img_mil_path, img_size):
    # Get size of the WSI.
    y,x = get_x_y(tiles[slide_indices[0], 0])
    x_min = x
    x_max = x
    y_min = y
    y_max = y
    for i in tiles[slide_indices, 0]:
        y_i, x_i  = get_x_y(i)
        x_min = min(x_min, x_i)
        y_min = min(y_min, y_i)
        x_max = max(x_max, x_i)
        y_max = max(y_max, y_i)
    x_max += 1
    y_max += 1

    # try:
    # Original 5x.
    wsi = np.zeros((x_max*img_size, y_max*img_size, 3), dtype=np.uint8)

    for index in slide_indices:
        if oriind_5x[index] == 0:
            continue
        tile_i   = tiles[index, 0]
        y_i, x_i  = get_x_y(tile_i)
        x_i *= img_size
        y_i *= img_size
        img_index = oriind_5x[index, 0].astype(np.int32)
        img_set   = orig_set[index,0]
        if 'train' in img_set:
            tile_img = img_5x[0][img_index]
        elif 'valid' in img_set:
            tile_img = img_5x[1][img_index]
        elif 'test' in img_set:
            tile_img = img_5x[2][img_index]
        wsi[x_i:x_i+img_size, y_i:y_i+img_size, :] = tile_img.astype(np.uint8)
    pil_img = Image.fromarray(wsi)
    pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_5X_original.jpg' % (slide, label, str(np.round(probability,3)))))

    del wsi
    del pil_img

    slide_weights_5x       = weights_5x[slide_indices,0]
    slide_weights_5x_norm  = (slide_weights_5x - np.min(slide_weights_5x))/(np.max(slide_weights_5x)-np.min(slide_weights_5x))

    # Weighted 5x.
    wsi = np.zeros((x_max*img_size, y_max*img_size, 3), dtype=np.uint8)
    for i, index in enumerate(slide_indices):
        if oriind_5x[index] == 0:
            continue
        tile_i   = tiles[index, 0]
        y_i, x_i  = get_x_y(tile_i)
        x_i *= img_size
        y_i *= img_size
        img_index = oriind_5x[index, 0].astype(np.int32)
        img_set   = orig_set[index,0]
        if 'train' in img_set:
            tile_img = img_5x[0][img_index]
        elif 'valid' in img_set:
            tile_img = img_5x[1][img_index]
        elif 'test' in img_set:
            tile_img = img_5x[2][img_index]
        weight   = slide_weights_5x_norm[i]
        image_pil = tile_img*(weight)
        wsi[x_i:x_i+img_size, y_i:y_i+img_size, :] = image_pil.astype(np.uint8)
    pil_img = Image.fromarray(wsi)
    pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_5X_weighted.jpg' % (slide, label, str(np.round(probability,3)))))

    del wsi
    del pil_img

    # except:
    #     print('Error with slide:', slide)


def multimag_wsi_process_10x(label, probability, slide, tiles, slide_indices, oriind_10x, orig_set, img_10x, weights_5x, weights_10x, slides_img_mil_path, img_size):
    # Get size of the WSI.
    y,x = get_x_y(tiles[slide_indices[0],0])
    x_min = x
    x_max = x
    y_min = y
    y_max = y
    for i in tiles[slide_indices,0]:
        y_i, x_i  = get_x_y(i)
        x_min = min(x_min, x_i)
        y_min = min(y_min, y_i)
        x_max = max(x_max, x_i)
        y_max = max(y_max, y_i)
    x_max += 1
    y_max += 1

    # try:

    slide_weights_10x      = weights_10x[slide_indices, :, 0]*np.reshape(weights_5x[slide_indices,0], (-1, 1))
    slide_weights_10x      = np.reshape(slide_weights_10x, (-1,1))
    slide_weights_10x_norm = (slide_weights_10x - np.min(slide_weights_10x))/(np.max(slide_weights_10x)-np.min(slide_weights_10x))
    slide_weights_10x_norm = np.reshape(slide_weights_10x_norm, (weights_10x[slide_indices, :, 0].shape[0], weights_10x[slide_indices, :, 0].shape[1],1))

    # Original 10x.
    wsi_10x = np.zeros((x_max*img_size*2, y_max*img_size*2, 3), dtype=np.uint8)

    for index in slide_indices:
        tile_i   = tiles[index,0]
        img_set  = orig_set[index,0]
        y_i, x_i = get_x_y(tile_i)
        y_i *= 2
        x_i *= 2
        for i_i, index_10x in enumerate(oriind_10x[index, :, :]):
            if index_10x == 0.0:
                continue
            if i_i == 0:
                x_10x = x_i + 0
                y_10x = y_i + 0
            elif i_i == 1:
                x_10x = x_i + 1
                y_10x = y_i + 0
            elif i_i == 2:
                x_10x = x_i + 0
                y_10x = y_i + 1
            else:
                x_10x = x_i + 1
                y_10x = y_i + 1

            if 'train' in img_set:
                tile_img = img_10x[0][index_10x[0]]
            elif 'valid' in img_set:
                tile_img = img_10x[1][index_10x[0]]
            elif 'test' in img_set:
                tile_img = img_10x[2][index_10x[0]]

            wsi_10x[x_10x*img_size:(x_10x+1)*img_size, y_10x*img_size:(y_10x+1)*img_size, :] = tile_img.astype(np.uint8)

    pil_img = Image.fromarray(wsi_10x)
    pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_10X_original.jpg' % (slide, label, str(np.round(probability,3)))))

    del wsi_10x
    del pil_img

    # Weighted 10x.
    wsi_10x = np.zeros((x_max*img_size*2, y_max*img_size*2, 3), dtype=np.uint8)

    for i, index in enumerate(slide_indices):
        tile_i    = tiles[index,0]
        img_set   = orig_set[index,0]
        y_i, x_i  = get_x_y(tile_i)
        y_i *= 2
        x_i *= 2
        for i_i, index_10x in enumerate(oriind_10x[index, :, :]):
            if index_10x == 0.0:
                continue
            if i_i == 0:
                x_10x = x_i + 0
                y_10x = y_i + 0
            elif i_i == 1:
                x_10x = x_i + 1
                y_10x = y_i + 0
            elif i_i == 2:
                x_10x = x_i + 0
                y_10x = y_i + 1
            else:
                x_10x = x_i + 1
                y_10x = y_i + 1

            if 'train' in img_set:
                tile_img = img_10x[0][index_10x[0]]
            elif 'valid' in img_set:
                tile_img = img_10x[1][index_10x[0]]
            elif 'test' in img_set:
                tile_img = img_10x[2][index_10x[0]]

            weight   = slide_weights_10x_norm[i, i_i, 0]
            image_pil = tile_img*(weight)

            wsi_10x[x_10x*img_size:(x_10x+1)*img_size, y_10x*img_size:(y_10x+1)*img_size, :] = image_pil.astype(np.uint8)

    pil_img = Image.fromarray(wsi_10x)
    pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_10X_weighted.jpg' % (slide, label, str(np.round(probability,3)))))

    del wsi_10x
    del pil_img

    # except:
    #     print('Error with slide:', slide)


def multimag_wsi_process_20x(label, probability, slide, tiles, slide_indices, oriind_20x, orig_set, img_20x, weights_5x, weights_20x, slides_img_mil_path, img_size):
    # Get size of the WSI.
    y,x = get_x_y(tiles[slide_indices[0],0])
    x_min = x
    x_max = x
    y_min = y
    y_max = y
    for i in tiles[slide_indices,0]:
        y_i, x_i  = get_x_y(i)
        x_min = min(x_min, x_i)
        y_min = min(y_min, y_i)
        x_max = max(x_max, x_i)
        y_max = max(y_max, y_i)
    x_max += 1
    y_max += 1

    try:
        slide_weights_20x      = weights_20x[slide_indices, :, 0]*np.reshape(weights_5x[slide_indices,0], (-1, 1))
        slide_weights_20x      = np.reshape(slide_weights_20x, (-1,1))
        slide_weights_20x_norm = (slide_weights_20x - np.min(slide_weights_20x))/(np.max(slide_weights_20x)-np.min(slide_weights_20x))
        slide_weights_20x_norm = np.reshape(slide_weights_20x_norm, (weights_20x[slide_indices, :, 0].shape[0], weights_20x[slide_indices, :, 0].shape[1],1))

        # Original 20x.
        wsi_20x = np.zeros((x_max*img_size*4, y_max*img_size*4, 3), dtype=np.uint8)
        oriind_20x = np.reshape(oriind_20x, (oriind_20x.shape[0], 16, oriind_20x.shape[-1]))
        for index in slide_indices:
            tile_i   = tiles[index,0]
            img_set   = orig_set[index,0]
            y_i, x_i  = get_x_y(tile_i)
            x_i_20x = x_i*4
            y_i_20x = y_i*4

            for i_i, i in enumerate(oriind_20x[index,:, :]):
                if i == 0:
                    continue
                i = int(i)
                x_y = return_coor(i_i)
                x_i_20x_p = x_i_20x + x_y[0]
                y_i_20x_p = y_i_20x + x_y[1]

                if 'train' in img_set:
                    tile_img = img_20x[0][i]
                elif 'valid' in img_set:
                    tile_img = img_20x[1][i]
                elif 'test' in img_set:
                    tile_img = img_20x[2][i]

                wsi_20x[x_i_20x_p*img_size:(x_i_20x_p+1)*img_size, y_i_20x_p*img_size:(y_i_20x_p+1)*img_size, :] = tile_img.astype(np.uint8)

        pil_img = Image.fromarray(wsi_20x)
        pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_20X_original.jpg' % (slide, label, str(np.round(probability,3)))))

    except:
        print('Not able to file original slide:', slide)
    finally:
        del wsi_20x
        del pil_img

    try:
        # Weighted 20x.
        wsi_20x = np.zeros((x_max*img_size*4, y_max*img_size*4, 3), dtype=np.uint8)
        oriind_20x = np.reshape(oriind_20x, (oriind_20x.shape[0], 16, oriind_20x.shape[-1]))
        for j, index in enumerate(slide_indices):
            tile_i   = tiles[index,0]
            img_set   = orig_set[index,0]
            y_i, x_i  = get_x_y(tile_i)
            x_i_20x = x_i*4
            y_i_20x = y_i*4

            for i_i, i in enumerate(oriind_20x[index,:, :]):
                if i == 0:
                    continue
                i = int(i)
                x_y = return_coor(i_i)
                x_i_20x_p = x_i_20x + x_y[0]
                y_i_20x_p = y_i_20x + x_y[1]
                if 'train' in img_set:
                    tile_img = img_20x[0][i]
                elif 'valid' in img_set:
                    tile_img = img_20x[1][i]
                elif 'test' in img_set:
                    tile_img = img_20x[2][i]
                weight   = slide_weights_20x_norm[j, i_i, 0]
                image_pil = tile_img*(weight)
                wsi_20x[x_i_20x_p*img_size:(x_i_20x_p+1)*img_size, y_i_20x_p*img_size:(y_i_20x_p+1)*img_size, :] = image_pil.astype(np.uint8)

        pil_img = Image.fromarray(wsi_20x)
        pil_img.save(os.path.join(slides_img_mil_path, '%s_%s_%s_20X_weighted.jpg' % (slide, label, str(np.round(probability,3)))))

    except:
        print('Not able to file weighted slide:', slide)
    finally:
        del wsi_20x
        del pil_img


def get_attentions_5x(uniq_slides, slides, patterns, tiles, oriind_5x, orig_set, img_5x, weights_5x, slides_img_mil_path, label='LUAD', img_size=224):

    for slide, probability, class_slide in uniq_slides:
        slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
        if len(slide_indices)==0:
            print(slide)
            continue

        if label in patterns[slide_indices[0], 0]:
            label_slide = 'LUAD'
        else:
            label_slide = 'LUSC'

        multimag_wsi_process_5x(label_slide, probability, slide, tiles, slide_indices, oriind_5x, orig_set, img_5x, weights_5x, slides_img_mil_path, img_size)


def get_attentions_10x(uniq_slides, slides, patterns, tiles, oriind_10x, orig_set, img_10x, weights_5x, weights_10x, slides_img_mil_path, label='LUAD', img_size=224):

    for slide, probability, class_slide in uniq_slides:
        slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
        if len(slide_indices)==0:
            print(slide)
            continue

        if label in patterns[slide_indices[0], 0]:
            label_slide = 'LUAD'
        else:
            label_slide = 'LUSC'

        multimag_wsi_process_10x(label_slide, probability, slide, tiles, slide_indices, oriind_10x, orig_set, img_10x, weights_5x, weights_10x, slides_img_mil_path, img_size)


def get_attentions_20x(uniq_slides, slides, patterns, tiles, oriind_20x, orig_set, img_20x, weights_5x, weights_20x, slides_img_mil_path, label='LUAD', img_size=224):

    for slide, probability, class_slide in uniq_slides:
        slide_indices = list(np.argwhere(slides[:]==slide)[:,0])
        if len(slide_indices)==0:
            print(slide)
            continue

        if label in patterns[slide_indices[0], 0]:
            label_slide = 'LUAD'
        else:
            label_slide = 'LUSC'

        multimag_wsi_process_20x(label_slide, probability, slide, tiles, slide_indices, oriind_20x, orig_set, img_20x, weights_5x, weights_20x, slides_img_mil_path, img_size)


def get_attentions(slides, patterns, tiles, oriind_5x, oriind_10x, oriind_20x, orig_set, img_5x, img_10x, img_20x, label, img_size, directories, fold_path, num_folds, h5_file_name):
    histograms_img_mil_path, latent_paths, wsi_paths, miss_wsi_paths = directories
    for fold_number in range(num_folds):
        print('\tFold', fold_number)

        hdf5_path_weights_comb  = '%s/fold_%s/results/%s' % (fold_path, fold_number, h5_file_name)

        ### Attention runs
        weights_20x, weights_10x, weights_5x, probs, slides_metrics, train_slides, valid_slides, test_slides = gather_attention_results_multimag(hdf5_path_weights_comb)

        top_slides, wrt_slides = pull_top_missclassified(test_slides, slides, slides_metrics, probs, patterns, label, top_percent=0.05)

        top_slides = top_slides[:3]
        wrt_slides = wrt_slides[:3]

        # # cluster 23
        # wrt_slides = ['TCGA-78-7220-01Z-00-DX1', 'TCGA-55-7727-01Z-00-DX1', 'TCGA-96-A4JK-01Z-00-DX1', 'TCGA-77-8009-01Z-00-DX1', 'TCGA-56-A4BY-01Z-00-DX1', 'TCGA-55-5899-01Z-00-DX1',
        # 'TCGA-NC-A5HO-01Z-00-DX1', 'TCGA-22-4591-01Z-00-DX1', 'TCGA-55-1594-01Z-00-DX1', 'TCGA-22-0940-01Z-00-DX1', 'TCGA-53-7624-01Z-00-DX1', 'TCGA-49-6743-01Z-00-DX1']
        #
        # # cluster 28
        # wrt_slides = ['TCGA-97-8175-01Z-00-DX1', 'TCGA-95-7562-01Z-00-DX1', 'TCGA-77-A5GB-01Z-00-DX1', 'TCGA-75-5125-01Z-00-DX1', 'TCGA-69-A59K-01Z-00-DX1', 'TCGA-52-7811-01Z-00-DX1',
        # 'TCGA-MP-A4TF-01Z-00-DX1', 'TCGA-56-8503-01Z-00-DX1', 'TCGA-43-5668-01Z-00-DX1', 'TCGA-56-7822-01Z-00-DX1', 'TCGA-NK-A5CT-01Z-00-DX1', 'TCGA-MP-A4T4-01Z-00-DX1', ]

        ### Attention Maps WSI
        print('\t\tSlides at 5x')
        get_attentions_5x(top_slides, slides, patterns, tiles[:].astype(str), oriind_5x, orig_set, img_5x, weights_5x, wsi_paths[0],      label, img_size)
        get_attentions_5x(wrt_slides, slides, patterns, tiles[:].astype(str), oriind_5x, orig_set, img_5x, weights_5x, miss_wsi_paths[0], label, img_size)

        # print('\t\tSlides at 10x')
        get_attentions_10x(top_slides, slides, patterns, tiles[:].astype(str), oriind_10x, orig_set, img_10x, weights_5x, weights_10x, wsi_paths[1],      label, img_size)
        get_attentions_10x(wrt_slides, slides, patterns, tiles[:].astype(str), oriind_10x, orig_set, img_10x, weights_5x, weights_10x, miss_wsi_paths[1], label, img_size)

        # print('\t\tSlides at 20x')
        get_attentions_20x(top_slides, slides, patterns, tiles[:].astype(str), oriind_20x, orig_set, img_20x, weights_5x, weights_20x, wsi_paths[2],      label, img_size)
        get_attentions_20x(wrt_slides, slides, patterns, tiles[:].astype(str), oriind_20x, orig_set, img_20x, weights_5x, weights_20x, miss_wsi_paths[2], label, img_size)
