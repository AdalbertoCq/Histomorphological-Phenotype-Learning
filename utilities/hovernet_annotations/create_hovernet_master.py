import pandas as pd
import numpy as np
import argparse
import math
import glob
import sys
import os

# Add project path
main_path = os.path.dirname(os.path.realpath(__file__))
main_path = '/'.join(main_path.split('/')[:-2])
sys.path.append(main_path)

# Own library.
from models.visualization.attention_maps import get_x_y

def create_main_file(cell_types, cell_names, csvs_path, output_csv):
    all_csvs   = glob.glob(os.path.join(csvs_path, '*_data_details_per_tile_per_celltype.txt'))
    for i_csv, csv_path in enumerate(sorted(all_csvs)):
        all_data = list()
        slide_id = csv_path.split('/')[-1].split('_')[0]
        print('(%s/%s) Iterating through slide %s...' % (i_csv+1, len(all_csvs), slide_id))
        slide_df = pd.read_csv(csv_path)
        slide_df = slide_df[['TileID', 'type', 'Nb_partices']]
        slide_df = slide_df[~slide_df['type'].isna()]

        if slide_df.shape[0] == 0:
            print('\t[Warning] Empty slide annotation.')
            continue

        # For each tile
        slide_df = slide_df.sort_values(by='TileID')
        tile_id_prev = ''
        cell_counts_tile = [0]*len(cell_types)
        for i, values in enumerate(zip(slide_df.TileID.values.tolist(), slide_df.type.values.astype(int).tolist(), slide_df.Nb_partices.values.astype(int).tolist())):
            tile_id, type_cell, counts = values
            if tile_id!=tile_id_prev:
                if i!=0:
                    all_data.append([slide_id, tile_id_prev, x, y] + cell_counts_tile)
                x, y = get_x_y(tile_id)
                cell_counts_tile = [0]*len(cell_types)
                tile_id_prev = tile_id
            cell_counts_tile[type_cell] = counts

        all_data = np.stack(all_data)
        all_df   = pd.DataFrame(all_data, columns=['slides', 'tile_id', 'x', 'y'] + cell_names)
        all_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

def include_5x_xy_annotations(all_df, output_csv):
    x_values    = all_df.x.values.astype(int).tolist()
    y_values    = all_df.y.values.astype(int).tolist()
    x_5x_values = list()
    y_5x_values = list()
    for x, y in zip(x_values, y_values):
        x_5x_values.append(math.floor(x/4))
        y_5x_values.append(math.floor(y/4))

    all_df['x_5x'] = x_5x_values
    all_df['y_5x'] = y_5x_values

    all_df.to_csv(output_csv, index=False)
    return all_df

def create_5x_from_20x(all_df, cell_names, output_5x_csv):
    # Get 5x annotations from 20x.
    if 'slide_tile_5x' not in all_df.columns:
        all_df['slide_tile_5x'] = all_df.apply(lambda x: '%s_%s_%s' % (x['slides'], x['x_5x'], x['y_5x']), axis=1)
    all_df['annotated_20x_tile_count'] = 1
    all_5x_df = all_df[['slide_tile_5x', 'annotated_20x_tile_count']+cell_names].groupby('slide_tile_5x').sum()
    all_5x_df = all_5x_df.reset_index()
    if 'slide' not in all_5x_df.columns:
        all_5x_df['slides'] = all_5x_df.apply(lambda x: x['slide_tile_5x'].split('_')[0], axis=1)
        all_5x_df['tiles']  = all_5x_df.apply(lambda x: '%s_%s' % (x['slide_tile_5x'].split('_')[1], x['slide_tile_5x'].split('_')[2]), axis=1)
        del all_5x_df['slide_tile_5x']
    all_5x_df.to_csv(output_5x_csv, index=False)


##### Main #######
parser = argparse.ArgumentParser(description='Script to all HoverNet annotations into one file.\nDirectory structure assumption: workspace/datasets/HoverNet/\'dataset\'/\'magnification\'.')
parser.add_argument('--dataset',       dest='dataset',       type=str,            default='NYU_LUADall_5x', help='Dataset to use.')
parser.add_argument('--magnification', dest='magnification', type=str,            default='20x',            help='Magnification.')
parser.add_argument('--main_path',     dest='main_path',     type=str,            default=None,             help='Path for the output run.')
args           = parser.parse_args()
dataset        = args.dataset
magnification  = args.magnification
main_path      = args.main_path


if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))
    main_path = '/'.join(main_path.split('/')[:-2])

# Working directories.
csvs_path     = '%s/datasets/HoverNet/%s/%s' % (main_path, dataset, magnification)
output_csv    = os.path.join(csvs_path, '%s_hovernet_annotations_20x.csv' % dataset)

# Reference for cell types in the tiles.
cell_types = [0,            1,                 2,                   3,                 4,           5                               ]
cell_names = ['cell other', 'cell neoplastic', 'cell inflammatory', 'cell connective', 'cell dead', 'cell non-neoplastic epithelial']

# Create main file.
if not os.path.isfile(output_csv):
    create_main_file(cell_types, cell_names, csvs_path, output_csv)
all_df = pd.read_csv(output_csv)

# Include 5x locations.
if magnification == '20x' and (('x_5x' not in all_df.columns) or ('y_5x' not in all_df.columns)):
    all_df = include_5x_xy_annotations(all_df, output_csv)

# Create a reference file for 5x tiles from 20x.
output_csv = os.path.join(csvs_path, '%s_hovernet_annotations_5x.csv' % dataset)
if not os.path.isfile(output_csv):
    create_5x_from_20x(all_df, cell_names, output_csv)