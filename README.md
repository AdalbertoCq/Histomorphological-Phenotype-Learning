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
- [Light weight version.](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/demos/slides/PRL%20Summary.pdf)
- [High resolution version.](https://drive.google.com/file/d/1zy7oSqCvq_ZIUW1Ix7rEjYjYBnepQZLA/view?usp=sharing)
<p align="center">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/12589de42685f38630e5b2378c0e6f27e16b3ea3/demos/framework_methodology.jpg" width="500">
</p>

```
Pending:
    2. Datasets: Where and how.
    3. Workspace Setup and Directory Structure: More information on assumptions, name convention (datasets across train/validation/test must be named the same).
    4. Pre-trained models
    5. Instructions: comment on each step with clarifications.
    6. Fold creation.
    7. Background/Artifact cluster removal.
```

## Repository overview

In this repository you will find the following sections: 
1. [WSI Tiling Process](#WSI-Tiling-process): Instruction on how to create H5 files with WSI tiles.
2. [Workspace Setup and Directory Structure](#Workspace-Setup-and-Directory-Structure): Details on how to setup the directory and H5 file naming convention.
3. [Specification on the content of the H5 files](#Specifications-on-the-content-of-the-H5-files):
4. [HPL Instructions](#HPL-Instructions): Details on how to run the complete methodology. 
5. [Pretrained models](#Pretrained-models): Pretrained weights for the self-supervised trained CNN models. 
6. [Dockers](#Dockers): Docker environments to run the different instruction steps.
7. [Python Environment](#Python-Environment): Python version and packages necessary. 

## WSI Tiling process
This step converts the whole slide images (WSI) in SVS format into 224x224 tiles and store them into H5 files.

We used the framework provided in [Coudray et al. 'Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning' Nature Medicine, 2018.](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code)

The steps to run it that framework are _0.1_, _0.2.a_, and _4_ (end of readme). In our work we used Reinhardt normalization, which can be applied at the same time as the tiling is done through the _'-N'_ option in step _0.1_.

## Workspace Setup and Directory Structure
The code will make the following assumptions with respect to where the datasets, model training outputs, and image representations are stored. 
**This structure is necessary if you want to run step 2 and following**: 
- Datasets: 
    - Dataset folder.
    - Follows the following structure: 
        - datasets/**dataset_name**/**marker_name**/patches_h**tile_size**_w**tile_size**
        - E.g.: _datasets/LUAD_5x/he/patches_h224_w224_
    - Train, validation, and test sets:
        - Each dataset will assume that at least there is a training set. 
        - Naming convention: 
            - hdf5_**dataset_name**\_**marker_name**\_**set_name**.h5 
            - E.g.: _datasets/LUAD_5x/he/patches_h224_w224/hdf5_LUAD_5x_he_train.h5_
- Data_model_output: 
    - Output folder for self-supervised trained models.
    - Follows the following structure:
        - data_model_output/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: _data_model_output/BarlowTwins_3/TCGAFFPE_LUAD_5x/h224_w224_n3_zdim128_
- Results: 
    - Output folder for self-supervised representations results.
    - This folder will contain the representation, clustering data, and logistic/cox regression results.
    - Follows the following structure:
        - results/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: _results/BarlowTwins_3/TCGAFFPE_LUAD_5x/h224_w224_n3_zdim128_

## Specifications on the content of the H5 files
    
## HPL Instructions
The complete flow consists in the following steps:
1. **Self-supervised Barlow Twins training.**
2. **Tile image projection on self-supervised trained encoder.**
3. **Combination of all sets into a complete one**: This allows to later perform a clustering fold cross-validation.
4. **Fold cross-validation files.**
5. **Include meta data into complete H5 file**: Cancer subtype, event indicator, or event time. Fields that will later be used in the logistic or cox regression.
6. **Leiden clustering over fold cross validation**.
7. **[Optional] Removing background tiles.** 
8. **Logistic regression for WSI classification.**
9. **Cox proportional hazard regression for survival prediction.**
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
1. Pickle file containing samples (patients/slides) for a 5 fold cross validation:
   1. Class classification: [notebook](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/fold_creation/class_folds.ipynb)
   2. Survival: [notebook](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/fold_creation/survival_folds.ipynb)
   3. Examples used in paper:
      1. [LUAD vs LUSC](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUADLUSC/lungsubtype_Institutions.pkl)
      2. [LUAD Overall Survival](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUAD/overall_survival_TCGA_folds.pkl)
2. CSV file with data used for the task (classification or survival):
   1. Column names on the 
   2. Examples used in paper:
      1. [LUAD vs LUSC](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUADLUSC/LUADLUSC_lungsubtype_overall_survival.csv)
      2. [LUAD Overall Survival](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/files/LUAD/overall_survival_TCGA_folds.csv)

### 5. Add fields to samples. E.g.: Subtype flag, OS event indicator, or OS event time

```
Script to create a subset H5 representation file based on meta data file.
optional arguments:
  -h, --help            show this help message and exit
  --meta_file           Path to CSV file with meta data.
  --meta_name           Name to use to rename H5 file.
  --list_meta_field     Field name that contains the information to include in the H5 file.
  --matching_field      Reference filed to use, cross check between original H5 and meta file.
  --h5_file             Original H5 file to parse.
  --override            Override 'complete' H5 file if it already exists.
```
Command example:
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

Usage:
```
Run Leiden Comunity detection over Self-Supervised representations.
optional arguments:
  -h, --help            show this help message and exit
  --subsample           Number of sample to run Leiden on, default is None, 200000 works well.
  --n_neighbors         Number of neighbors to use when creating the graph, default is 250.
  --meta_field          Purpose of the clustering, name of output folder.
  --matching_field      Key used to match folds split and H5 representation file.
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
You can find the notebook to run the correaltion and figure [here](https://github.com/AdalbertoCq/Phenotype-Representation-Learning/blob/main/utilities/visualizations/cluster_correlations_figures.ipynb). 

## Pretrained Models
Self-supervised model weights:
1. [Lung adenocarcinoma (LUAD) and squamous cell carcinoma (LUSC) model](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020). 
2. [PanCancer: BRCA, HNSC, KICH, KIRC, KIRP, LUSC, LUAD](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_PanCancer_5x/19949708).
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
