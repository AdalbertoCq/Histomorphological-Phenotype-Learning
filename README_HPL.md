# HPL Instructions
The flow consists in the following steps:
1. Self-supervised Barlow Twins training.
2. Tile vector representations.
3. Combination of all sets into one H5.
4. Fold cross validation files.
5. Include metadata in H5 file.
6. Leiden clustering.
7. Removing background tiles.
8. Logistic regression for lung type WSI classification.
9. Cox proportional hazards for survival regression.
10. Correlation between annotations and clusters.
11. Get tiles and WSI samples for HPCs.

## 1. Self supervised model training
This step trains the self-supervised model on a given dataset.

**Step Inputs:**
- Dataset H5 train file. E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`

**Step Outputs:**
- Model weights. At the end of training, there should be a folder with the Self-supervised CNN details. Weights are located at the 'checkpoints' folder. E.g.: `data_model_output/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128`

In our work, we train the model only on 250K tiles. You can use this [script](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/h5_handling/subsample_h5.py) to subsample tiles from you training set. Afterward, you will need to setup a dataset structure for the 250K training set. Make sure you follow the details from [**Workspace Setup**](#Workspace-setup). E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc_250K/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_250K_he_train.h5`

Usage:
```
Self-Supervised model training.
optional arguments:
  -h, --help            show this help message and exit
  --epochs              Number epochs to run: default is 45 epochs.
  --batch_size          Batch size, default size is 64.
  --dataset             Dataset to use.
  --marker              Marker of dataset to use, default is H&E.
  --img_size            Image size for the model.
  --img_ch              Number of channels for the model.
  --z_dim               Latent space size for constrastive loss.
  --model               Model name.
  --main_path           Path for the output run.
  --dbs_path            Directory with DBs to use.
  --check_every         Save checkpoint and show UMAP samples every X epcohs.
  --restore             Restore previous run and continue.
  --report              Report latent pace progress.
```
Command example:
```
python3 run_representationspathology.py \
--img_size 224 \
--batch_size 64 \
--epochs 60 \
--z_dim 128 \
--model BarlowTwins_3 \
--dataset TCGAFFPE_LUADLUSC_5x_250K \
--check_every 10 \
--report 
```

## 2. Tile vector representations
This step uses the self-supervised trained CNN to find vector representations for each tile image.

There are two options when to run this step. You can do it by file (e.g. an external cohort) or by using an entire dataset (e.g. TCGA training, validation, and test sets).

**Step Inputs:**
- Dataset H5 files. E.g.:
    - Training set: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
    - Validation set: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`
    - Test set: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5`
- Individual file. E.g.: `datasets/NYUFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined.h5`
- Self-supervised trained CNN. E.g.: `data_model_output/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/checkpoints/BarlowTwins_3.ckt`

**Step Outputs:**
- Tile vector representations. E.g.:
    - Dataset:
        - Training set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
        - Validation set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`
        - Test set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5`
    - Individual file: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined.h5`

**Scripts:**
- Individual file: Please refer to `run_representationspathology_projection.py`
- Entire dataset: Please refer to `run_representationspathology_projection_dataset.py`

Usage:
```
Project images onto Self-Supervised model latent space.
optional arguments:
  -h, --help            show this help message and exit
  --checkpoint          Path to pre-trained weights (.ckt) of Self-Supervised model.
  --real_hdf5           Path for real image to encode.
  --img_size            Image size for the model.
  --img_ch              Number of channels for the model.
  --z_dim               Latent space size, default is 128.
  --dataset             Dataset to use.
  --marker              Marker of dataset to use.
  --batch_size          Batch size, default size is 64.
  --model               Model name.
  --main_path           Path for the output run.
  --dbs_path            Directory with DBs to use.
  --save_img            Save reconstructed images in the H5 file.
```
Command example:
```
python3 ./run_representationspathology_projection.py \
--checkpoint ./data_model_output/BarlowTwins_3/TCGAFFPE_5x_perP_250k/h224_w224_n3_zdim128/checkpoints/BarlowTwins_3.ckt \
--real_hdf5 ./datasets/TCGAFFPE_5x_perP/he/patches_h224_w224/hdf5_TCGAFFPE_5x_perP_he_test.h5 \
--model BarlowTwins_3 
```


Usage:
```
Project images onto Self-Supervised model latent space.
optional arguments:
  -h, --help            show this help message and exit
  --checkpoint          Path to pre-trained weights (.ckt) of Self-Supervised model.
  --img_size            Image size for the model.
  --img_ch              Number of channels for the model.
  --z_dim               Latent space size, default is 128.
  --dataset             Dataset to use.
  --marker              Marker of dataset to use.
  --batch_size          Batch size, default size is 64.
  --model MODEL         Model name.
  --main_path           Path for the output run.
  --dbs_path            Directory with DBs to use.
  --save_img            Save reconstructed images in the H5 file.
```
Command example:
```
python3 ./run_representationspathology_projection.py \
--checkpoint ./data_model_output/BarlowTwins_3/TCGAFFPE_5x_perP_250k/h224_w224_n3_zdim128/checkpoints/BarlowTwins_3.ckt \
--dataset TCGAFFPE_5x_perP \
--model BarlowTwins_3 
```

## 3. Combine all representation sets into one H5 file
This step takes all set H5 files with tile vector representations and merges then into a single H5 file.

**Step Inputs:**
- Dataset H5 files with tile vector representations. E.g.:
    - Training set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
    - Validation set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`
    - Test set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5`

**Step Output:**
- Combined H5 file. E.g.:
    - Complete set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete.h5`

[**Important**] This code works on the assumptions specified in the [**Workspace setup**](#Workspace-setup)

You can find the TCGA tile vector representation used in the paper in the section [**TCGA tile vector representations**](#TCGA-tile-vector-representations)

Usage:
```
Script to combine all H5 representation file into a 'complete' one.
optional arguments:
  -h, --help            show this help message and exit
  --img_size            Image size for the model.
  --z_dim               Dimensionality of vector representations. Default is the z latent of Self-Supervised.
  --dataset             Dataset to use.
  --model               Model name.
  --main_path           Path for the output run.
  --override            Override 'complete' H5 file if it already exists.
```
Command example:
```
python3 ./utilities/h5_handling/combine_complete_h5.py \
--img_size 224 \
--z_dim 128 \
--dataset TCGAFFPE_5x_perP \
--model ContrastivePathology_BarlowTwins_2
```

## 4. Fold cross validation files for classification and survival analysis
This step defines the 5-fold cross-validation to run the classification and survival analysis. The files created here will be used in the Leiden clustering, logistic regression, and Cox proportional hazards.

**Step Inputs:**
You can create the CSV and pickle files with these notebooks:
1. Class classification: [notebook](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/fold_creation/class_folds.ipynb)
2. Survival: [notebook](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/fold_creation/survival_folds.ipynb)

**Step Outputs:**
1. Pickle file: It contains samples (patients or slides) for each fold in the 5-fold cross-validation. E.g.:
    - LUAD vs LUSC: [utilities/files/LUADLUSC/lungsubtype_Institutions.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUADLUSC/lungsubtype_Institutions.pkl)
    - LUAD Overall Survival: [utilities/files/LUAD/overall_survival_TCGA_folds.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)
2. CSV file with data used for the task: It contains labels for each sample. This file is used in Step 5 (Include metadata in H5 file). Please verify that the values in the column with patients or slides (matching_field) follows the same format as the 'dataset' in the H5 file that contains the same type of information. This field is to cross-check each sample and include the metadata into the H5 file. E.g.:
    - LUAD vs LUSC: [utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv)
    - LUAD Overall Survival: [utilities/files/LUAD/overall_survival_TCGA_folds.csv](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUAD/overall_survival_TCGA_folds.csv)

## 5. Include metadata in H5 file
This step includes metadata into the H5 file. It used the data in the CSV files from the previous steps. The metadata is later used in the cancer type classification (logistic regression) or survival analysis (Cox proportional hazards).

**Step Inputs:**
- H5 file with tile vector representations. E.g.:
    - Complete set (Set 3): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete.h5`
- CSV file with metadata. E.g.:
    - Lung type and survival data: [utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv)

**Step Output:**
- H5 file with tile vector representations and metadata. E.g.:
    - Complete set: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5`

[**Important**] Please verify that the values in the column with patients or slides (matching_field) follows the same format as the 'dataset' in the H5 file that contains the same type of information. This field is to cross-check each sample and include the metadata into the H5 file.

You can find the TCGA tile vector representation used in the paper in the section [**TCGA tile vector representations**](#TCGA-tile-vector-representations)

```
Script to create a subset H5 representation file based on meta data file.
optional arguments:
  -h, --help            show this help message and exit
  --meta_file            Path to CSV file with meta data.
  --meta_name           Name to use to rename H5 file.
  --list_meta_field      Field name that contains the information to include in the H5 file.
  --matching_field       Reference filed to use, cross check between original H5 and meta file.
  --h5_file              Original H5 file to parse.
  --override            Override 'complete' H5 file if it already exists.
```
Command example:
```
 python3 ./utilities/h5_handling/create_metadata_h5.py \
 --meta_file ./utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv \
 --matching_field slides \
 --list_meta_field luad os_event_ind os_event_data \
 --h5_file ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete.h5
 --meta_name lungsubtype_survival
```

## 6. Leiden clustering based on fold cross validation
This step performs clustering by only using representations in the training set. Samples in the training set are taken from the fold pickle (Step 4). Keep in mind that if there are 5 folds, the script will perform 5 different clustering configurations. One per training set.

At this step you can select if you want to run several resolution parameters or just one. The resolution parameter indirectly controls the number of clusters, where a higher value results in higher number of clusters.
By default it will run the following resolutions `[0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`. If you are just running one resolution value, you can provide it through the argument `--resolution`. If you want to run a range of them, you can modify this at line 45 of `run_representationsleiden.py`.

[**Note**] If you are running this step for filtering out background and artifact tiles (Step 7). I suggest to use `--resolution 5.0`.

[**Important**] You can find further information on this step in the sections **Online Methods - Evaluation** and **Supplementary Figure 8** from the [paper](https://arxiv.org/abs/2205.01931).

**Step Inputs:**
- H5 file with tile vector representations and metadata. E.g.:
    - Complete set (Step 5): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5`
- [Optional] H5 file with external cohort. It should include the same kind of metadata. E.g.:
    - Additional file: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined_lungsubtype_survival.h5`
- Pickle file with 5-fold cross-validation. E.g.:
    - Lung type classification (Step 4): [utilities/files/LUADLUSC/lungsubtype_Institutions.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUADLUSC/lungsubtype_Institutions.pkl)
    - LUAD Overall Survival (Step 4): [utilities/files/LUAD/overall_survival_TCGA_folds.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)

**Step Output:**
- Cluster configuration files will be under the directory `meta_field`/adatas. E.g.:
    - Lung type classification: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/lungsubtype_nn250/adatas`
        - Per resolution and fold. The output files are:
            - Train set H5AD: This file contains the cluster configuration and it is used by the scanpy package to map external tile vector representations to existing clusters. E.g.: `TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_1p0__fold1_subsample.h5ad`
            - Train set CSV: Tiles from training set with cluster assignations. E.g.: `TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_1p0__fold1.csv`
            - Validation set CSV: Tiles from validation set with cluster assignations. E.g.: `TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_1p0__fold1_valid.csv`
            - Test set CSV: Tiles from test set with cluster assignations. E.g.: `TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_1p0__fold1_test.csv`

Usage:
```
Run Leiden Comunity detection over Self-Supervised representations.
optional arguments:
  -h, --help            show this help message and exit
  --subsample           Number of samples used to run Leiden. Default is None, 200000 works well.
  --n_neighbors         Number of neighbors to use when creating the graph. Default is 250.
  --meta_field          Purpose of the clustering, name of output folder.
  --matching_field      Key used to match folds split and H5 representation file.
  --rep_key             Key pattern for representations to grab: z_latent, h_latent.
  --folds_pickle        Pickle file with folds information.
  --main_path           Workspace main path.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.

```
Command examples:
```
python3 ./run_representationsleiden.py \
--meta_field lung_subtypes_nn250 \
--matching_field slides \
--folds_pickle ./utilities/files/LUAD/folds_LUAD_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5 \
--subsample 200000

```
```
python3 ./run_representationsleiden.py \
--meta_field luad_overall_survival_nn250 \
--matching_field slides \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5 \
--subsample 200000

```

## 7. Remove background tiles
**Optional step**.
This is step removes tile vector representations that correspond to background or artifact tile images. It's composed by 4 different steps.

1. Get tile samples per HPC:
    - Select a cluster fold and resolution. At this step is good to select a high resolution parameter (e.g. 5) so the clusters are more compact that allows to a finer filtering of background and artifact clusters.
    - You can refer to the step ['11. Get tiles and WSI samples for HPCs'](#11.-Get-tiles-and-WSI-samples-for-HPCs).
2. Identify background and artifact clusters and create pickle file with tiles to remove:
    - From the previous step you can identify which HPCs contain background and artifacts.
    - Use this [notebook](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/tile_cleaning/review_cluster_create_pickles.ipynb) to create the pickle files in order to remove tile vector representations from the H5 file.
3. Remove tile instances from H5 file:
    - Remove tile vector representations contained into the pickle file from the H5 file (created in Step 5).

**Step Inputs:**
- Pickle file created in sub-step 2. E.g.: [utilities/files/indexes_to_remove/TCGAFFPE_LUADLUSC_5x_60pc/complete.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/indexes_to_remove/TCGAFFPE_LUADLUSC_5x_60pc/complete.pkl)
- Original H5 file with tile vector representations used for clustering (Step 5). E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5`

**Step Output:**
- H5 file without background and artifact tile vector representations: E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`

Usage:
```
Script to remove indexes from H5 file.

optional arguments:
  -h, --help            show this help message and exit
  --h5_file H5_FILE      Original H5 file to parse.
  --pickle_file          Pickle file with indexes to remove.
  --override            Override 'complete' H5 file if it already exists.
```
Command example:
```
python3 ./utilities/tile_cleaning/remove_indexes_h5.py \
--pickle_file ./utilities/files/indexes_to_remove/TCGAFFPE_LUADLUSC_5x_60pc/complete.pkl \ 
--h5_file ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5 \
```

4. Re-cluster representations:
    - Create a new directory for a new clustering step and copy over the filtered H5 file from the previous sub-step. E.g.: `./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered`
    - Repeat Step 6 ['Leiden clustering'](#6.-Leiden-clustering-based-on-fold-cross-validation).


## 8. Logistic regression for classification
This is step runs a binary classification over logistic regression.

It is important to mention that you can run this step by using the different cluster configuration per fold or you can select to use a common cluster configuration across the classification folds. 
The latter allows to compare the significance of HPCs across folds for the classification task. 

[**Important**] You can find further information on this step in the sections **Online Methods - Evaluation** and **Supplementary Figure 8** from the [paper](https://arxiv.org/abs/2205.01931).

In our paper, we first run the classification task with different cluster configurations per fold. The purpose of this step is to ensure that defining HPCs with different WSI will yield similar results. 
After this, we locked down a cluster fold by providing the argument `--force_fold`.

[**Note**] The script runs the alphas `[0.1, 0.5, 1.0, 5.0, 10.0, 25.0]` and resolutions `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]` by default. If you want to run different alphas or resolutions you can edit these at lines 45-46 of `report_representationsleiden_lr.py`. 

**Step Inputs:**
- Cluster configuration files (Step 6): Files that contain HPC assignations for each tile. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/lungsubtype_nn250/adatas`
- Folds pickle file (Step 4): This file contains the training, validation, and test set for the classification task. E.g.: [utilities/files/LUADLUSC/lungsubtype_Institutions.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUADLUSC/lungsubtype_Institutions.pkl)
- H5 file with tile vector representations (Step 5/7). E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`

**Step Outputs:**
At the provided `meta_folder` directory, you will find the following files:
- `alphas_summary_auc_mintiles_%s_label1.jpg`: Summary figure with logistic regression performance for all different alpha penalties. In addition, it provides a table with statistically significant HPCs per resolution and fold for prediciton.
- `clusters_stats_mintiles_%s_label1.csv`: CSV file with raw results of the previous figure.
- `alpha_%s_mintiles_%s`: Directory with results for a given alpha penalty:
  - `luad_auc_results_mintiles_%s.jpg`: Figure with logistic regression performance for the given penalty.
  - `luad_auc_results_mintiles_%s.csv`: Raw data for previous figure.
  - `forest_plots`: Directory with Forest plots for all resolutions and folds. If `force_fold` is provided, it will output an additional file `leiden_%s_stats_all_folds_label1.jpg` that summarizes the Forest plot for all folds. 

Usage:
```
Report classification and cluster performance based on Logistic Regression.
optional arguments:
  -h, --help            show this help message and exit
  --meta_folder         Purpose of the clustering, name of output folder.
  --meta_field           Meta field to use for the Logistic Regression.
  --matching_field       Key used to match folds split and H5 representation file.
  --diversity_key       Key use to check diversity within cluster: Slide, Institution, Sample.
  --type_composition    Space trasnformation type: percent, clr, ilr, alr.
  --min_tiles           Minimum number of tiles per matching_field.
  --folds_pickle        Pickle file with folds information.
  --force_fold          Force fold of clustering.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters [Optional].
  --report_clusters     Flag to report cluster circular plots.
```
Command example:
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lung_subtypes_nn250 \
--meta_field luad \
--matching_field slides \
--min_tiles 100 \
--folds_pickle ./utilities/files/LUAD/folds_LUAD_Institutions.pkl \
--h5_complete_path ./results/ContrastivePathology_BarlowTwins_3/TCGAFFPE_5x_perP/h224_w224_n3_zdim128/hdf5_TCGAFFPE_5x_perP_he_complete_lungsubtype_survival.h5 \
--h5_additional_path ./results/ContrastivePathology_BarlowTwins_3/NYU_BiFrFF_5x/h224_w224_n3_zdim128/hdf5_NYU_BiFrFF_5x_he_test_luad.h5
```
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lung_subtypes_nn250 \
--meta_field luad \
--matching_field slides \
--min_tiles 100 \
--force_fold 4 \
--folds_pickle ./utilities/files/LUAD/folds_LUAD_Institutions.pkl \
--h5_complete_path ./results/ContrastivePathology_BarlowTwins_3/TCGAFFPE_5x_perP/h224_w224_n3_zdim128/hdf5_TCGAFFPE_5x_perP_he_complete_lungsubtype_survival.h5 \
--h5_additional_path ./results/ContrastivePathology_BarlowTwins_3/NYU_BiFrFF_5x/h224_w224_n3_zdim128/hdf5_NYU_BiFrFF_5x_he_test_luad.h5
```

## 9.A Cox proportional hazards for survival regression
This is step runs a survival analysis with a Cox proportional hazards.

It is important to mention that you can run this step by using the different cluster configuration per fold or you can select to use a common cluster configuration across the survival folds.
The latter allows to compare the significance of HPCs across folds for the survival task.

[**Important**] You can find further information on this step in the sections **Online Methods - Evaluation** and **Supplementary Figure 8** from the [paper](https://arxiv.org/abs/2205.01931).

In our paper, we first run the survival task with different cluster configurations per fold. The purpose of this step is to ensure that defining HPCs with different WSI will yield similar results.
After this, we locked down a cluster fold by providing the argument `--force_fold`.

[**Note**] The script runs a range of alpha penalties `10. ** np.linspace(-4, 4, 50)`, l1 ratios `[0.0]`, and resolutions `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]` by default. If you want to run different alphas or resolutions you can edit these at lines 48-52 of `report_representationsleiden_cox.py`.

**Step Inputs:**
- Cluster configuration files (Step 6): Files that contain HPC assignations for each tile. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/luad_overall_survival_nn250/adatas`
- Folds pickle file (Step 4): This file contains the training and test set for the survival task. E.g.: [utilities/files/LUAD/overall_survival_TCGA_folds.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)
- H5 file with tile vector representations (Step 5/7). E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`

**Step Outputs:**
At the provided `meta_folder` directory, you will find the following files:
- `c_index_%s_l1_ratio_%s_mintiles_%s.jpg`: Summary figure with c-index performance for all resolutions
- `c_index_%s_l1_ratio_%s_mintiles_%s.csv`: CSV file with raw results of the previous figure.

Usage:
```
Report survival and cluster performance based on Cox proportional hazards.

optional arguments:
  -h, --help            show this help message and exit
  --meta_folder         Purpose of the clustering, name of folder.
  --matching_field       Key used to match folds split and H5 representation file.
  --event_ind_field      Key used to match event indicator field.
  --event_data_field     Key used to match event data field.
  --diversity_key       Key use to check diversity within cluster: Slide, Institution, Sample.
  --type_composition    Space transformation type: percent, clr, ilr, alr.
  --min_tiles           Minimum number of tiles per matching_field.
  --folds_pickle        Pickle file with folds information.
  --force_fold          Force fold of clustering.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.
  --additional_as_fold  Flag to specify if additional H5 file will be used for cross-validation.
  --report_clusters     Flag to report cluster circular plots.
```
Command example:
```
python3 ./report_representationsleiden_cox.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field os_event_ind \
--event_data_field os_event_data \
--min_tiles 100 \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--h5_additional_path ./results/ContrastivePathology_BarlowTwins_3/NYU_LUADall_5x/h224_w224_n3_zdim128/hdf5_NYU_LUADall_5x_he_combined_os_pfs_survival.h5  
```
```
python3 ./report_representationsleiden_cox.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field os_event_ind \
--event_data_field os_event_data \
--min_tiles 100 \
--force_fold 0 \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined_filtered.h5  
```

## 9.B Cox proportional hazards for survival regression - Individual resolution and penalty
This is step runs a survival analysis with a Cox proportional hazards for a particular resolution and penalty.

This script will provide forest plots and Kaplan-Meier plots for the specific resolution and alpha. 
The alpha penalty is optional, if not provided it will run a range of penalties and select the one based on best performance for the TCGA test set, using the additional cohort to verify performance and generalization.  

**Step Inputs:**
- Cluster configuration files (Step 6): Files that contain HPC assignations for each tile. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/luad_overall_survival_nn250/adatas`
- Folds pickle file (Step 4): This file contains the training and test set for the survival task. E.g.: [utilities/files/LUAD/overall_survival_TCGA_folds.pkl](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)
- H5 file with tile vector representations (Step 5/7). E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`

**Step Outputs:**
At the provided `meta_folder` directory, you will find the following files:
- `KM_leiden_%s.jpg`: Kaplan-Meier plot for a given set. 
- `hazard_ratios_summary.jpg`: Summary figure with log hazard ratios and performance variations for the different alpha penalties.
- `leiden_%s_fold_%s_clusters.jpg`: Forest plot for HPC and fold.
- `leiden_%s_stat_all_clusters.jpg`: If the `force_fold` parameter is provided, it will output this file that summarizes the Forest plot for HPC across all folds.

Usage:
```
Report survival and cluster performance based on Cox proportional hazards.

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         Cox regression penalty value.
  --resolution          Leiden resolution.
  --meta_folder         Purpose of the clustering, name of folder.
  --matching_field       Key used to match folds split and H5 representation file.
  --event_ind_field      Key used to match event indicator field.
  --event_data_field     Key used to match event data field.
  --diversity_key       Key use to check diversity within cluster: Slide, Institution, Sample.
  --type_composition    Space transformation type: percent, clr, ilr, alr.
  --l1_ratio            L1 Penalty for Cox regression.
  --min_tiles           Minimum number of tiles per matching_field.
  --force_fold          Force fold of clustering.
  --folds_pickle        Pickle file with folds information.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.
  --additional_as_fold  Flag to specify if additional H5 file will be used for cross-validation.
```
Command example:
```
python3 ./report_representationsleiden_cox_individual.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field os_event_ind \
--event_data_field os_event_data \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \ 
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined_filtered.h5 \ 
--resolution 2.0 \
--force_fold 0 \
--l1_ratio 0.0 \
--alpha 1.0 
```

## 10. Correlation between annotations and clusters
You can find the notebook to run correlations and figures [here](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/cluster_correlations_figures.ipynb).

## 11. Get tiles and WSI samples for HPCs
This step provides tile images per each HPC and WSI with cluster overlays. In order to provide WSIs, you will need to edit the dictionary `value_cluster_ids` in line 52 of `report_representationsleiden_samples.py`. Clusters provided at key `1` will show in the output csv files as related to outcome classification (`1`) or survival(`dead event`). If the cluster if provided at key `0`, it will show as related to outcome classification (`0`) or survival (`survival event`).

**Step Inputs:**
- H5 file with tile vector representations and metadata. E.g.:
    - Complete set (Step 5): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`
- Dataset with original tile images that macth with previous H5 file. E.g.:
    - TCGAFFPE_LUADLUSC_5x_60pc ([Workspace setup](#Workspace-setup)): `datasets/TCGAFFPE_LUADLUSC_5x_60pc_250K`

**Step Output:**
- Folder under `meta_folder/leiden_%resolution_fold%fold`. E.g.: `lung_subtypes_nn250/leiden_2p0_fold4`
    - wsi_clusters: folder with original WSI and WSI with HPC overlay.
    - wsi_annotation.csv: CSV with WSI names dominant HPC and percentage over the total area. In addition, it provides a link to the GDC website where it can be further visualized.
    - images: folder with tiles per HPC
    - cluster_annotation.csv: CSV with HPCs.
    - backtrack: folder with information to back track tiles provided for each HPC.

Usage:
```
Report cluster images from a given Leiden cluster configuration.

optional arguments:
  -h, --help            show this help message and exit
  --meta_folder         Purpose of the clustering, name of folder.
  --meta_field           Meta field to use for the Logistic Regression or Cox event indicator.
  --matching_field       Key used to match folds split and H5 representation file.
  --resolution          Minimum number of tiles per matching_field.
  --dpi DPI             Highest quality: 1000.
  --fold FOLD           Cluster fold configuration to use.
  --dataset DATASET     Dataset with thei tile images that match tile vector representations in h5_complete_path.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.
  --min_tiles MIN_TILES Minimum number of tiles per matching_field.
  --dbs_path DBS_PATH   Path for the output run.
  --img_size IMG_SIZE   Image size for the model.
  --img_ch IMG_CH       Number of channels for the model.
  --marker MARKER       Marker of dataset to use.
  --tile_img            Flag to dump cluster tile images.
  --extensive           Flag to dump test set cluster images in addition to train.
  --additional_as_fold  Flag to specify if additional H5 file will be used for cross-validation.

```

Command example:
```
python3 ./report_representationsleiden_samples.py \
--meta_folder lung_subtypes_nn250 \
--meta_field luad \
--matching_field slides \
--resolution 2.0 \
--fold 0 \
--h5_complete_path results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--dpi 1000 \
--dataset TCGAFFPE_LUADLUSC_5x_60pc
  
```

