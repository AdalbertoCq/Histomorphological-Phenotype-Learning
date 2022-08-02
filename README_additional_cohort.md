# HPL Instructions - Mapping an external cohort to existing clusters

## 1. Setup directories
You will need to setup directories for the lung type classification task and LUAD overall survival

E.g.:
- LUAD vs LUSC: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/lung_subtypes_nn250`
- LUAD Overall Survival: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/luad_overall_survival_nn250`

## 2. Download  and copy TCGA HPL files
### TCGA tile vector representations
1. Download tile projections for TCGA LUAD and LUSC cohorts [here](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view?usp=sharing). These are the projections used in the publication results.

2. Copy them over to the directory `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc_250K/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival`

### TCGA clusters
You can find cluster configurations used in the publication results at:
1. [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. [LUAD vs LUSC type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
3. [LUAD survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)

## 3. WSI tiling process
Use the [WSI tiling process](./README.md#WSI-tiling-process) to obtain tile images from the original WSIs.

## 3. Tile vector representations
Use [step 2 of HPL methodology](./README_HPL.md) to find tile vector representations for each tile image.

## 4. Tile vector representations
Include metadata in the H5 file of your cohort by runing [step 5 of HPL methodology](./README_HPL.md).

## 5. Assign tile vector representations to existing HPCs.
This steps assigns an HPC to each tile vector representation of your external cohort.


