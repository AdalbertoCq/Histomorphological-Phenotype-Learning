# HPL Paper Results - Files and replication.
These instructions will provide the results files for the TCGA data. In the following section we do not include the external cohort from the New York University, for any questions regarding these cohorts please address reasonable requests to the corresponding authors.

## 1. Download TCGA tile images
Download and setup both datasets TCGA LUAD & LUSC WSI tile images [here](./README.md#TCGA-HPL-files)

After doing this step you should have a directory containing the TCGA LUAD & LUSC tile image dataset `datasets/TCGAFFPE_LUADLUSC_5x_60pc`

## 2. Download folder with TCGA tile vector representations and cluster configurations. 
You can directly download the whole `results` folder [here](). The folder contains the following:
1. TCGA tile vector representations (filtered background and artifacts): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`
2. lungtype_nn250: cluster configurations and results lung type classification. 
3. lungtype_nn250_clusterfold4: cluster configurations and results lung type classification with consistent clusters across classification folds.
4. luad_overall_survival_nn250: cluster configurations and results LUAD survival regression.
5. luad_overall_survival_nn250_clusterfold0: cluster configurations and results LUAD survival regression with consistent clusters across survival folds.

In our paper, we first run the classification and survival task with different cluster configurations per fold. The purpose of this step is to ensure that defining clusters (HPCs) with different WSI will yield similar results. After this, we locked down a cluster fold by providing the argument `--force_fold`.
This is the difference between `lungtype_nn250` and `lungtype_nn250_clusterfold4`, and also `luad_overall_survival_nn250` and `luad_overall_survival_nn250_clusterfold0`. 

**[Important]** You can find further information on this step in the sections Online Methods - Evaluation and Supplementary Figure 8 from the paper.


## 3. Running the lung classification and LUAD survival regressions.
The previous folders already contain the results from the paper. Nevertheless, if you wanted to rerun the steps 8 (Logisitic regression for lung classification) and 9 (Cox proportional hazards for survival analysis) from [HPL](./README_HPL.md); these are the commands:

Logistic regression for lung classification (different cluster configurations per fold):
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lungtype_nn250_clusterfold4 \
--meta_field luad \
--matching_field slides \
--folds_pickle ./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 
```

Logistic regression for lung classification (consistent cluster configuration per fold):
```
python3 ./report_representationsleiden_lr.py \
--meta_folder lungtype_nn250_clusterfold4 \
--meta_field luad \
--matching_field slides \
--folds_pickle ./utilities/files/LUADLUSC/lungsubtype_Institutions.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--force_fold 4
```

Cox proportional hazards for LUAD overall survival analysis (different cluster configurations per fold):
```
python3 ./report_representationsleiden_cox.py \
 --meta_folder luad_overall_survival_nn250 \
 --matching_field samples \
 --event_ind_field os_event_ind \
 --event_data_field os_event_data \
 --folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
 --h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5```
```

Cox proportional hazards for LUAD overall survival analysis (consistent cluster configuration per fold):
```
python3 ./report_representationsleiden_cox.py \
 --meta_folder luad_overall_survival_nn250_clusterfold0 \
 --matching_field samples \
 --event_ind_field os_event_ind \
 --event_data_field os_event_data \
 --folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
 --h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5
 --force_fold 0
```

Cox proportional hazards for LUAD overall survival analysis - Individual resolution and Forest plots:
```
python3 ./report_representationsleiden_cox_individual.py \
--meta_folder luad_overall_survival_nn250 \
--matching_field samples \
--event_ind_field os_event_ind \
--event_data_field os_event_data \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5 \
--force_fold 0 \ 
--resolution 2.0 \
--l1_ratio 0.0 \
--alpha 1.0 
```

## 4. Paper figures 