# HPL Paper Results - Files and replication.
These instructions will provide the results files for the TCGA data. For any questions regarding the  New York University cohorts, please address reasonable requests to the corresponding authors.

## 1. Download TCGA tile images
Download and setup both datasets TCGA LUAD & LUSC WSI tile images [here](./README.md#TCGA-HPL-files)

After doing this step you should have a directory containing the TCGA LUAD & LUSC tile image dataset `datasets/TCGAFFPE_LUADLUSC_5x_60pc`

## 2. Download folder with TCGA tile vector representations and cluster configurations. 
You can directly download the whole `results` folder [here](). The folder contains the following:
1. TCGA tile vector representations (filtered background and artifacts): `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128_filtered/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival_filtered.h5`
2. lungtype_nn250: cluster configurations and results lung type classification. 
3. lungtyoe_nn250_clusterfold4:
4. luad_overall_survival_nn250:
5. luad_overall_survival_nn250_clusterfold0:

The folders already contain results for lung type classification and overall survival analysis. In case you would like to rerun these steps, you can find the commands below:
- Lung type classification with 

## 3. Figures