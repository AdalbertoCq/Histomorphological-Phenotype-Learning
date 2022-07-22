# Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes
* **[Quiros A.C.<sup>+</sup>, Coudray N.<sup>+</sup>, Yeaton A., Yang X., Chiriboga L., Karimkhan A., Narula N., Pass H., Moreira A.L., Le Quesne J.<sup>\*</sup>, Tsirigos A.<sup>\*</sup>, and Yuan K.<sup>\*</sup> Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes. 2022](https://arxiv.org/abs/2205.01931)**

**Abstract:**

*Histopathological images provide the definitive source of cancer diagnosis, containing information used by pathologists to identify and subclassify malignant disease, and to guide therapeutic choices. These images contain vast amounts of information, much of which is currently unavailable to human interpretation. Supervised deep learning approaches have been powerful for classification tasks, but they are inherently limited by the cost and quality of annotations. Therefore, we developed Histomorphological Phenotype Learning, an unsupervised methodology, which requires no annotations and operates via the self-discovery of discriminatory image features in small image tiles. Tiles are grouped into morphologically similar clusters which appear to represent recurrent modes of tumor growth emerging under natural selection. These clusters have distinct features which can be identified using orthogonal methods. Applied to lung cancer tissues, we show that they align closely with patient outcomes, with histopathologically recognised tumor types and growth patterns, and with transcriptomic measures of immunophenotype.*

## Citation
```
@misc{QuirosCoudray2022,
      title={Self-supervised learning in non-small cell lung cancer discovers novel morphological clusters linked to patient outcome and molecular phenotypes},
      author={Adalberto Claudio Quiros and Nicolas Coudray and Anna Yeaton and Xinyu Yang and Luis Chiriboga and Afreen Karimkhan and Navneet Narula and Harvey Pass and Andre L. Moreira and John Le Quesne and Aristotelis Tsirigos and Ke Yuan},
      year={2022},
      eprint={2205.01931},
      archivePrefix={arXiv},
      primaryClass={cs.CV}        
}
```

## Demo Materials

Slides summarizing methodology and results: 
- [Light-weight version.](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/demos/slides/HPL%20Summary.pdf)
- [High-resolution version.](https://drive.google.com/file/d/1F5ffZqXoNLpT5dgzVLhhCnnspyUe4FPQ/view?usp=sharing)
<p align="center">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/12589de42685f38630e5b2378c0e6f27e16b3ea3/demos/framework_methodology.jpg" width="500">
</p>

## Repository overview

In this repository you will find the following sections: 
1. [WSI tiling process](#WSI-Tiling-process): Instructions on how to create H5 files from WSI tiles.
2. [Workspace setup](#Workspace-Setup): Details on H5 file content and directory structure.
3. [HPL instructions](#HPL-Instructions): Step-by-step instructions on how to run the complete methodology. 
4. [TCGA HPL files](#TCGA-HPL-files): HPL output files of paper results.  
5. [Dockers](#Dockers): Docker environments to run the different instruction steps.
6. [Python Environment](#Python-Environment): Python version and packages necessary.
7. [Frequently Asked Questions](#Frequently-Asked-Questions).

## WSI Tiling process
This step divides whole slide images (WSIs) in SVS format into 224x224 tiles and store them into H5 files.

We used the framework provided in [Coudray et al. 'Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning' Nature Medicine, 2018.](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code)

The steps to run it that framework are _0.1_, _0.2.a_, and _4_ (end of readme). In our work we used Reinhardt normalization, which can be applied at the same time as the tiling is done through the _'-N'_ option in step _0.1_.

[ToDo] Include here sentences of requirements for train, validation, and test.

## Workspace Setup 
This section specifies requirements on H5 file content and directory structure to run the flow.

In the instructions below we use the following variables and names:
- **dataset_name**: TCGAFFPE_LUADLUSC_5x_60pc
- **marker_name**: he
- **tile_size**: 224

### H5 file content specification.

[H5 file python package documentation](https://docs.h5py.org/en/stable/quick.html)

[ToDo] Naming convention for the datasets.

All H5 set files should have the same datasets. E.g. images, slides, cancer_type.

Example:
- File: **hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5**
    - Dataset names:
        - **train_img, train_tiles, train_slides, train_samples**
- File: **hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5**
    - Dataset names:
        - **valid_img, valid_tiles, valid_slides, valid_samples**
- File: **hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5**
    - Dataset names:
        - **test_img, test_tiles, test_slides, test_samples**

### Directory Structure
The code will make the following assumptions with respect to where the datasets, model training outputs, and image representations are stored:
- Datasets: 
    - Dataset folder.
    - Follows the following structure: 
        - datasets/**dataset_name**/**marker_name**/patches_h**tile_size**_w**tile_size**
        - E.g.: _datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224_
    - Train, validation, and test sets:
        - Each dataset will assume that at least there is a training set. 
        - Naming convention: 
            - hdf5_**dataset_name**\_**marker_name**\_**set_name**.h5 
            - E.g.: _datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5_
- Data_model_output: 
    - Output folder for self-supervised trained models.
    - Follows the following structure:
        - data_model_output/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: _data_model_output/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_
- Results: 
    - Output folder for self-supervised representations results.
    - This folder will contain the representation, clustering data, and logistic/cox regression results.
    - Follows the following structure:
        - results/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: _results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128_
    
## HPL Instructions
The flow consists in the following steps:
1. **Self-supervised Barlow Twins training.**
2. **Tile image projection on self-supervised trained encoder.**
3. **Combination of all sets into a complete one.**
4. **Fold cross-validation files.**
5. **Include metadata into complete H5 file.**
6. **Leiden clustering.**
7. **Removing background tiles [Optional] .** 
8. **Logistic regression for lung type WSI classification.**
9. **Cox proportional hazards for survival prediction.**
10. **Correlation between annotations and clusters.**

### 1. Self-Supervised model training

[**Note**] It is important to setup the directories and h5 files according to Section [Workspace Setup and Directory Structure](#Workspace-Setup-and-Directory-Structure).

Requirements on h5 file used for self-supervised model training:
1. H5 file naming and directory structure. E.g.: _datasets/LUADLUSC_5x/he/patches_h224_w224/hdf5_LUADLUSC_5x_he_train.h5_
2. The H5 dataset key containing the tile images must contain the patterns 'images' or 'img'.

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
  --model               Model name, used to select the type of model (SimCLR, RelationalReasoning, BarlowTwins).
  --main_path           Path for the output run.
  --dbs_path            Directory with DBs to use.
  --check_every         Save checkpoint and projects samples every X epcohs.
  --restore             Restore previous run and continue.
  --report              Report Latent Space progress.
```
Command example:
```
python3 /nfs/PhD_Workspace/run_representationspathology.py \
--img_size 224 \
--batch_size 64 \
--epochs 60 \
--z_dim 128 \
--model BarlowTwins_3 \
--dataset TCGAFFPE_LUADLUSC_5x_250K \
--check_every 10 \
--report 
```

### 2. Tile Projections

This step encodes each tile image into a vector representation using a pretrained self-supervised model. 
You can choose how to project tiles into vector representations if you want to, either with each h5 file individually or by projection an entire train/validation/test dataset. 

- File projection: Projects a given file tile images into the self-supervised trained encoder. Please refer to `run_representationspathology_projection.py`
- Dataset projection: Projects train/validation/test set tile images into the self-supervised trained encoder. Please refer to `run_representationspathology_projection_dataset.py`

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
  --model               Model name, used to select the type of model (SimCLR, RelationalReasoning, BarlowTwins).
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
  --model MODEL         Model name, used to select the type of model (SimCLR, RelationalReasoning, BarlowTwins).
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

### 3. Combine all representation sets into one representations file
This step takes all set H5 files in the projection folder and merges then into a single H5 file. E.g.:
- Original H5 files [**Required**]:
    - results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_he_train.h5
    - results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_he_validation.h5
    - results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_he_test.h5
- Combined H5 file [**Output**]:
    - results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_he_complete.h5

[**Important**] The code assumes that the datasets inside the H5 will have a common name across the train/validation/test sets. 
For further details please refer to Section [Specification on the content of the H5 files](#Specifications-on-the-content-of-the-H5-files).

Running this step allows to run the clustering step **5** based on an specified fold configuration, by collecting the complete H5 with all samples.

Usage:
```
Script to combine all H5 representation file into a 'complete' one.
optional arguments:
  -h, --help            show this help message and exit
  --img_size            Image size for the model.
  --z_dim               Dimensionality of projections, default is the Z latent of Self-Supervised.
  --dataset             Dataset to use.
  --model               Model name, used to select the type of model (SimCLR, RelationalReasoning, BarlowTwins).
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
### 4. Fold cross-validation files
In order to run clustering, logistic regression, and cox proportional hazard, you will need to create the following files:
1. Pickle file:
   1. It contains samples (patients or slides) for each fold in the 5-fold cross-validation.
   2. Examples used in paper:
       1. [LUAD vs LUSC](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUADLUSC/lungsubtype_Institutions.pkl)
       2. [LUAD Overall Survival](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)
2. CSV file with data used for the task (classification or survival):
    1. It contains labels (cancer type or survival data) for each sample. 
    2. This file is used in Step 5 (Include metadata into H5 file). Please verify that the values in the column with patients or slides (matching_field) follows the same format as the 'dataset' in the H5 file that contains the same type of information. This field is to cross-check each sample and include the metadata into the H5 file.  
    3. Examples used in paper:
        1. [LUAD vs LUSC](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv)
        2. [LUAD Overall Survival](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUAD/overall_survival_TCGA_folds.csv)

You can create the CSV and pickle files with these notebooks:
1. Class classification: [notebook](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/fold_creation/class_folds.ipynb)
2. Survival: [notebook](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/fold_creation/survival_folds.ipynb)

### 5. Include metadata into the H5 file.

This step includes metadata into the H5 file. The metadata is later used in the cancer type classification (logistic regression) or survival regression (Cox proportional hazards).

You can find examples of the CSV files used in this step here:
1. [LUAD vs LUSC](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv)
2. [LUAD Overall Survival](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUAD/overall_survival_TCGA_folds.csv)

Please verify that the values in the column with patients or slides (matching_field) follows the same format as the 'dataset' in the H5 file that contains the same type of information. This field is to cross-check each sample and include the metadata into the H5 file.

```
Script to create a subset H5 representation file based on meta data file.
optional arguments:
  -h, --help            show this help message and exit
  --meta_file           Path to CSV file with meta data.
  --meta_name           Name to use to rename H5 file.
  --list_meta_field     Field name that contains 
  he information to include in the H5 file.
  --matching_field      Reference filed to use, cross check between original H5 and meta file.
  --h5_file             Original H5 file to parse.
  --override            Override 'complete' H5 file if it already exists.
```
Command example that includes lung type and survival information into the H5 file:
```
 python3 ./utilities/h5_handling/create_metadata_h5.py \
 --meta_file ./utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv \
 --matching_field slides \
 --list_meta_field luad os_event_ind os_event_data \
 --h5_file ./results/ContrastivePathology_BarlowTwins_2/TCGAFFPE_5x_perP/h224_w224_n3_zdim128/hdf5_TCGAFFPE_5x_perP_he_complete.h5 \
 --meta_name lungsubtype_survival
```

### 6. [Optional] Removing background tiles
This is step allows to get rid of representation instances that are background or artifact tiles. It's composed by 4 different steps. 
1. Leiden clustering
2. Get cluster tile samples
3. Identify background and artifact clusters and create pickle file with tiles to remove
4. Remove tile instances from H5 file.

### 7. Leiden clustering based on fold cross validation
This step performs clustering by only using representations in the training set. Samples in the training set are taken from the specified fold pickle.

Keep in mind that if there are 5 folds, the script will perform 5 different clustering steps. One per training set. 

Usage:
```
Run Leiden Comunity detection over Self-Supervised representations.
optional arguments:
  -h, --help            show this help message and exit
  --subsample           Number of samples used to run Leiden. Default is None, 200000 works well.
  --n_neighbors         Number of neighbors to use when creating the graph. Default is 250.
  --meta_field           Purpose of the clustering, name of output folder.
  --matching_field       Key used to match folds split and H5 representation file.
  --rep_key REP_KEY     Key pattern for representations to grab: z_latent, h_latent.
  --folds_pickle        Pickle file with folds information.
  --main_path           Workspace main path.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.

```
Command example:
```
python3 ./run_representationsleiden.py \
--meta_field luad_overall_survival_nn250 \
--matching_field slides \
--folds_pickle ./utilities/files/LUAD/overall_survival_TCGA_folds.pkl \
--h5_complete_path ./results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_complete_lungsubtype_survival.h5 \
--subsample 200000

```

### 8. Logistic regression.

Usage:
```
Report classification and cluster performance based on Logistic Regression.
optional arguments:
  -h, --help            show this help message and exit
  --meta_folder         Purpose of the clustering, name of output folder.
  --meta_field          Meta field to use for the Logistic Regression.
  --matching_field      Key used to match folds split and H5 representation file.
  --diversity_key       Key use to check diversity within cluster: Slide, Institution, Sample.
  --type_composition    Space trasnformation type: percent, clr, ilr, alr.
  --min_tiles           Minimum number of tiles per matching_field.
  --folds_pickle        Pickle file with folds information.
  --force_fold          Force fold of clustering.
  --h5_complete_path    H5 file path to run the leiden clustering folds.
  --h5_additional_path  Additional H5 representation to assign leiden clusters.
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

### 9. Cox Proportional hazard regression

Usage:
```
Report classification and cluster performance based on Logistic Regression.

optional arguments:
  -h, --help            show this help message and exit
  --meta_folder         Purpose of the clustering, name of folder.
  --matching_field      Key used to match folds split and H5 representation file.
  --event_ind_field     Key used to match event indicator field.
  --event_data_field    Key used to match event data field.
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
--h5_complete_path ./results/ContrastivePathology_BarlowTwins_3/TCGAFFPE_5x_perP/h224_w224_n3_zdim128/hdf5_TCGAFFPE_5x_perP_he_complete_os_survival.h5 \
--h5_additional_path ./results/ContrastivePathology_BarlowTwins_3/NYU_LUADall_5x/h224_w224_n3_zdim128/hdf5_NYU_LUADall_5x_he_combined_os_pfs_survival.h5  
```

### 10. Correlation between annotations and clusters
You can find the notebook to run correlations and figures [here](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/visualizations/cluster_correlations_figures.ipynb). 

## TCGA HPL files.
This section contains the following TCGA files produced by HPL:
1. Self-supervised trained weights.
2. TCGA tile projections.
3. TCGA cluster configurations.
4. TCGA WSI & patient representations. 

### Pretrained Models
Self-supervised model weights:
1. [Lung adenocarcinoma (LUAD) and squamous cell carcinoma (LUSC) model](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020).
2. [PanCancer: BRCA, HNSC, KICH, KIRC, KIRP, LUSC, LUAD](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_PanCancer_5x/19949708).

### TCGA tile projections
You can find tile projections for TCGA LUAD and LUSC cohorts [here](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view?usp=sharing). These are the projections used in the publication results.

### TCGA clusters
You can find cluster configurations used in the publication results at:
1. [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. [LUAD vs LUSC type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
3. [LUAD survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)

At each of those locations you will find the AnnData H5 file with the cluster configuration. You can use this file along with the cluster assignment script 

### TCGA pa

## Dockers
These are the dockers with the environments to run different steps of the flow. Step 8 needs to be run with docker [**2**], all other steps can be run with docker [**1**]:
1. **Self-Supervised models training and projections:**
     - aclaudioquiros/tf_package:v16
2. **Leiden clustering:**
    - gcfntnu/scanpy:1.7.0 
   
## Python Environment
The code uses Python 3.7.12 and the following packages:
```
anndata==0.7.8
autograd==1.3
einops==0.3.0
h5py==3.4.0
lifelines==0.26.3
matplotlib==3.5.1
numba==0.52.0
numpy==1.21.2
opencv-python==4.1.0.25
pandas==1.3.3
Pillow==8.1.0
pycox==0.2.2
scanpy==1.8.1
scikit-bio==0.5.6
scikit-image==0.15.0
scikit-learn==0.24.0
scikit-network==0.24.0
scikit-survival==0.16.0
scipy==1.7.1
seaborn==0.11.2
setuptools-scm==6.3.2
simplejson==3.13.2
sklearn==0.0
sklearn-pandas==2.2.0
statsmodels==0.13.0
tensorboard==1.14.0
tensorflow-gpu==1.14.0
tqdm==4.32.2
umap-learn==0.5.0
wandb==0.12.7
```

## Frequently Asked Questions
#### I have my own cohort and I want to assign existing clusters to my own WSI tiles. Is it possible?
Yes, you can find the cluster configuration files for LUAD vs LUSC or LUAD survival at the TCGA cluster section.

#### When I run the Leiden clustering step. I get an \'TypeError: can't pickle weakref objects\' error in some folds.
Based on experience, this error occurs with non-compatible version on numba, umap-learn, and scanpy. The package versions in the python environment should work.

#### I want to reproduce the results from the paper.
These are the steps to reproduce the TCGA results.

Clone this repository and create a folder called '_results_'. Under that folder another one called '_TCGAFFPE_LUADLUSC_5x_60pc_'. Download the tile representation from '_TCGA tile projections_' section and place them under that folder.

**LUAD vs LUSC classification**

