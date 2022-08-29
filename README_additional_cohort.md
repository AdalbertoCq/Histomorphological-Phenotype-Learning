# HPL Instructions - Mapping an external cohort to existing clusters

## 1. WSI tiling process
Use the [WSI tiling process](./README.md#WSI-tiling-process) to obtain tile images from the original WSIs.

## 2. Download TCGA tile images
Download and setup both datasets TCGA LUAD & LUSC WSI tile images [here](./README.md#TCGA-HPL-files)

## 3. Tile vector representations
Use [step 2 of HPL methodology](./README_HPL.md) to find tile vector representations for each tile image.

## 4. Include metadata in your H5 file
Include metadata in the H5 file of your cohort by running [step 5 of HPL methodology](./README_HPL.md).

## 5. Background and artifact removal
1. Download the cluster configurations for removal of background and artifact tiles:
   - [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. Copy over the H5AD file:
   - `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/removal/adatas/TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_leiden_5p0__fold4_subsample.h5ad`
3. Assign tiles to existing cluster configuration:

```
python3 ./run_representationsleiden_assignment.py \
--resolution 5.0
--meta_field removal \
--folds_pickle ./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5 \
--h5_additional_path hdf5_this_is_your_external_cohort.h5
```
- [**Note**] You will see warnings for folds 0-3, that's fine: [Warning] H5AD file not found at '.h5ad' or '_subsample.h5ad'.
- At the end of this step you should see a csv file with your cohort tile cluster assignations. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/removal/adatas/NYUFFPE_LUADLUSC_5x_60pc_he_combined_leiden_5p0__fold4.csv`

4. Execute [steps 7.2 and 7.3 of the HPL instructions](./README_HPL.md):
   - The cluster ids used to remove this tiles can be found in the [notebook](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/tile_cleaning/review_cluster_create_pickles.ipynb)
   - `clusters_to_remove = [104, 102, 99, 90, 86, 83, 73, 58, 97, 89, 21]`
   - At the end of these steps you should have a file without tile representations that belong to the background/artifact clusters. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_this_is_your_external_cohort_filtered.h5`

## 6. Setup directory with filtered representations
1. Create the directory `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered`
2. Copy over the resulting H5 from the previous step `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_this_is_your_external_cohort_filtered.h5`
3. Download and copy over the [TCGA LUAD & LUSC tile vector representations](./README.md#TCGA-HPL-files). E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`
4. Create directories for lung type classification task and LUAD overall survival:
   1. LUAD vs LUSC type classification: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/lung_subtypes_nn250`
   2. LUAD Survival: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/luad_overall_survival_nn250`
5. Download cluster configurations and copy them over to the `adatas` directory for each of them:
   1. [LUAD vs LUSC type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
   2. [LUAD survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)

## 7. Assign clusters for lung type classification and LUAD survival

1. Assign clusters for lung subtype
```
python3 ./run_representationsleiden_assignment.py \
--resolution 2.0
--meta_field lung_subtypes_nn250 \
--folds_pickle ./utilities/files/LUAD/folds_LUAD_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_this_is_your_external_cohort_filtered.h5
```
- [**Note**] You will see warnings for folds 0-3, that's fine: [Warning] H5AD file not found at '.h5ad' or '_subsample.h5ad'.
- At the end of this step you should see a csv file with your cohort tile cluster assignations. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/lung_subtypes_nn250/adatas/NYUFFPE_LUADLUSC_5x_60pc_he_combined_filtered_leiden_2p0__fold4.csv`

2. Assign clusters for LUAD survival
```
python3 ./run_representationsleiden_assignment.py \
--resolution 2.0
--meta_field luad_overall_survival_nn250 \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_this_is_your_external_cohort_filtered.h5
```
- [**Note**] You will see warnings for folds 1-4, that's fine: [Warning] H5AD file not found at '.h5ad' or '_subsample.h5ad'.
- At the end of this step you should see a csv file with your cohort tile cluster assignations. E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/lung_subtypes_nn250/adatas/NYUFFPE_LUADLUSC_5x_60pc_he_combined_filtered_leiden_2p0__fold4.csv`

Now you should be able to run lung classification, survival analysis, or cluster correlations for your external cohort.

## 8. Logistic regression for classification
You can find full details on this step at the [HPL instruction readme](./README_HPL.md).

This is an example of the command to the logistic regression for your cohort. Make sure you modify line 46 in `report_representationsleiden_lr.py` to only run `resolutions=[2.0]`
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lung_subtypes_nn250 \
--meta_field luad \
--matching_field slides \
--min_tiles 100 \
--force_fold 4 \
--folds_pickle ./utilities/files/LUAD/folds_LUAD_Institutions.pkl \
--h5_complete_path ./results/ContrastivePathology_BarlowTwins_3/TCGAFFPE_5x_perP/h224_w224_n3_zdim128/hdf5_TCGAFFPE_5x_perP_he_complete_lungsubtype_survival.h5 \
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_this_is_your_external_cohort_filtered.h5
```

## 9. Cox proportional hazards for survival regression - Individual resolution and penalty
You can find full details on this step at the [HPL instruction readme](./README_HPL.md).

This is an example of the command to the Cox proportional hazards for survival regression.
```
python3 ./report_representationsleiden_cox_individual.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field pfs_event_ind \
--event_data_field pfs_event_data \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \ 
--h5_additional_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_NYUFFPE_LUADLUSC_5x_60pc_he_combined_filtered.h5 \ 
--resolution 2.0 \
--force_fold 0 \
--l1_ratio 0.0 \
--alpha 1.0 
```

## 10. Correlation between annotations and clusters
You can find the notebooks for cluster correlations and UMAP/PAGA figures at:
1. [Cluster (HPC) correlations figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/cluster_correlations.ipynb).
4. [UMAP and PAGA figures](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/utilities/visualizations/visualizations_UMAP_PAGA.ipynb).


## 11. Get tiles and WSI samples for HPCs
This step provides tile images per each HPC and WSI with cluster overlays.

You will need to use the flag `--tile_img` to obtain tile examples for all clusters.

To obtain WSIs, you will need to set the variables `only_id` and `value_cluster_ids` in line 52 of `report_representationsleiden_samples.py`.
The dictionary `value_cluster_ids` defines which WSI will be selected based on the percentage of the provided clusters. Clusters provided at key `1` will show in the output csv files as related to outcome classification (`1`) or survival(`dead event`). If the cluster is provided at key `0`, it will show as related to outcome classification (`0`) or survival (`survival event`).
The flag `only_id=False` will provide randomly selected WSI with cluster overlays. You can find examples of usage for these variables at lines 57 and 70 of `report_representationsleiden_samples.py`.

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
--dataset TCGAFFPE_LUADLUSC_5x_60pc \
--tile_img
```