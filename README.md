# Histomorphological Phenotype Learning
* **Quiros A.C.<sup>+</sup>, Coudray N.<sup>+</sup>, Yeaton A., Yang X., Liu B., Chiriboga L., Karimkhan A., Narula N., Moore D.A., Park C.Y., Pass H., Moreira A.L., Le Quesne J.<sup>\*</sup>, Tsirigos A.<sup>\*</sup>, and Yuan K.<sup>\*</sup> Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unlabeled, unannotated pathology slides. 2024**

[![DOI](https://zenodo.org/badge/505573958.svg)](https://zenodo.org/doi/10.5281/zenodo.10718821)

[Nature Communications](https://www.nature.com/articles/s41467-024-48666-7)

[ArXiv](https://arxiv.org/abs/2205.01931)

---

**Abstract:**

*Definitive cancer diagnosis and management depend upon the extraction of information from microscopy images by pathologists. These images contain complex information requiring time-consuming expert human interpretation that is prone to human bias. Supervised deep learning approaches have proven powerful for classification tasks, but they are inherently limited by the cost and quality of annotations used for training these models. To address this limitation of supervised methods, we developed Histomorphological Phenotype Learning (HPL), a fully unsupervised methodology that requires no expert labels or annotations and operates via the automatic discovery of discriminatory image features in small image tiles. Tiles are grouped into morphologically similar clusters which constitute a library of histomorphological phenotypes, revealing trajectories from benign to malignant tissue via inflammatory and reactive phenotypes. These clusters have distinct features which can be identified using orthogonal methods, linking histologic, molecular and clinical phenotypes. Applied to lung cancer tissues, we show that they align closely with patient survival, with histopathologically recognised tumor types and growth patterns, and with transcriptomic measures of immunophenotype. We then demonstrate that these properties are maintained in a multi-cancer study. These results show the clusters represent recurrent host responses and modes of tumor growth emerging under natural selection.*

---

## Citation
```
@article{QuirosCoudray2024,
	author = {Claudio Quiros, Adalberto and Coudray, Nicolas and Yeaton, Anna and Yang, Xinyu and Liu, Bojing and Le, Hortense and Chiriboga, Luis and Karimkhan, Afreen and Narula, Navneet and Moore, David A. and Park, Christopher Y. and Pass, Harvey and Moreira, Andre L. and Le Quesne, John and Tsirigos, Aristotelis and Yuan, Ke},
	journal = {Nature Communications},
	number = {1},
	pages = {4596},
	title = {Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides},
	volume = {15},
	year = {2024}}
}
```

<!-- ## Demo Materials -->

<!-- Slides summarizing methodology and results:  -->
<!-- - [Light-weight version.](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/demos/slides/HPL%20Summary.pdf)
- [High-resolution version.](https://drive.google.com/file/d/1F5ffZqXoNLpT5dgzVLhhCnnspyUe4FPQ/view?usp=sharing) -->
<p align="center">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/12589de42685f38630e5b2378c0e6f27e16b3ea3/demos/framework_methodology.jpg" width="400">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/demos/HPL_visualizer_mutlticancer.gif" alt="animated" />
</p>

---

## Repository overview

In this repository you will find the following sections: 
1. [WSI tiling process](#WSI-tiling-process): Instructions on how to create H5 files from WSI tiles.
2. [Workspace setup](#Workspace-setup): Details on H5 file content and directory structure.
3. [HPL instructions](./README_HPL.md): Step-by-step instructions on how to run the complete methodology.
   1. Self-supervised Barlow Twins training.
   2. Tile vector representations.
   3. Combination of all sets into one H5.
   4. Fold cross validation files.
   5. Include metadata in H5 file.
   6. Leiden clustering.
   7. Removing background tiles.
   8. HPC configuration selection.
   9. Logistic regression for lung type WSI classification.
   10. Cox proportional hazards for survival regression.
   11. Correlation between annotations and HPCs.
   12. Get tiles and WSI samples for HPCs.
4. [HPL Visualizer](#HPL-Visualizer): Interactive app to visualize UMAP representations, tiles, and HPC membership
5. [Frequently Asked Questions](#Frequently-Asked-Questions).
6. [TCGA HPL files](#TCGA-HPL-files): HPL output files from our paper results.
7. [Python Environment](#Python-Environment): Python version and packages.
8. [Dockers](#Dockers): Docker environments to run HPL steps.

---

## WSI tiling process
This step divides whole slide images (WSIs) into 224x224 tiles and store them into H5 files. At the end of this step, you should have three H5 files. One per training, validation, and test sets. The training set will be used to train the self-supervised CNN, in our work this corresponded to 60% of TCGA LUAD & LUSC WSIs.

We used the framework provided in [Coudray et al. 'Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning' Nature Medicine, 2018.](https://github.com/ncoudray/DeepPATH/tree/master/DeepPATH_code)
The steps to run the framework are _0.1_, _0.2.a_, and _4_ (end of readme). In our work we used Reinhardt normalization, which can be applied at the same time as the tiling is done through the _'-N'_ option in step _0.1_.

## Workspace setup 
This section specifies requirements on H5 file content and directory structure to run the flow.

In the instructions below we use the following variables and names:
- **dataset_name**: `TCGAFFPE_LUADLUSC_5x_60pc`
- **marker_name**: `he`
- **tile_size**: `224`

### H5 file content specification.
If you are not familiar with H5 files, you can find documentation on the python package [here](https://docs.h5py.org/en/stable/quick.html).

This framework makes the assumption that datasets inside each H5 set will follow the format 'set_labelname'. In addition, all H5 files are required to have the same number of datasets. 
Example:
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
    - Dataset names: `train_img`, `train_tiles`, `train_slides`, `train_samples`
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`
    - Dataset names: `valid_img`, `valid_tiles`, `valid_slides`, `valid_samples`
- File: `hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_test.h5`
    - Dataset names: `test_img`, `test_tiles`, `test_slides`, `test_samples`

### Directory Structure
The code will make the following assumptions with respect to where the datasets, model training outputs, and image representations are stored:
- Datasets: 
    - Dataset folder.
    - Follows the following structure: 
        - datasets/**dataset_name**/**marker_name**/patches_h**tile_size**_w**tile_size**
        - E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224`
    - Train, validation, and test sets:
        - Each dataset will assume that at least there is a training set. 
        - Naming convention: 
            - hdf5_**dataset_name**\_**marker_name**\_**set_name**.h5 
            - E.g.: `datasets/TCGAFFPE_LUADLUSC_5x_60pc/he/patches_h224_w224/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_train.h5`
- Data_model_output: 
    - Output folder for self-supervised trained models.
    - Follows the following structure:
        - data_model_output/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: `data_model_output/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128`
- Results: 
    - Output folder for self-supervised representations results.
    - This folder will contain the representation, clustering data, and logistic/cox regression results.
    - Follows the following structure:
        - results/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size**
        - E.g.: `results/BarlowTwins_3/TCGAFFPE_LUADLUSC_5x_60pc/h224_w224_n3_zdim128`
    
## HPL Instructions
The flow consists in the following steps:
1. Self-supervised Barlow Twins training.
2. Tile vector representations.
3. Combination of all sets into one H5.
4. Fold cross validation files.
5. Include metadata in H5 file.
6. Leiden clustering.
7. Removing background tiles.
8. HPC configuration selection.
9. Logistic regression for lung type WSI classification.
10. Cox proportional hazards for survival regression.
11. Correlation between annotations and HPCs.
12. Get tiles and WSI samples for HPCs.

*You can find the full details on HPL instructions in this [Readme_HPL file](README_HPL.md).*

---

## HPL Visualizer
You can find standalone apps in the following locations. These were built using [Marimo](https://github.com/marimo-team/marimo).
1. [HPL Multicancer Visualizer](https://drive.google.com/file/d/1SWOQnnB5n73-MMxqLZJ13OI6yc6bacYR/view?usp=share_link).
2. [HPL LUAD Visualizer](https://drive.google.com/file/d/1LMGTzb6vA3mL0NAbdwifJC1xrcvlMnWM/view?usp=share_link).

You can edit the code by running `marimo edit tile_visualizer_umap.py`. Run the app with `marimo run tile_visualizer_umap.py`. 

<p align="center">
  <img src="https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/demos/HPL_visualizer_mutlticancer.gif" alt="animated" />
</p>

---

## Frequently Asked Questions
#### I want to reproduce the paper results.
You can find TCGA files, results, and commands to reproduce them on this [Readme_replication file](./README_replication.md). For any questions regarding the  New York University cohorts, please address reasonable requests to the corresponding authors.

#### I have my own cohort and I want to assign existing HPCs to my own WSI.
You can follow steps on how to assign existing HPCs in this [Readme_additional_cohort file](README_additional_cohort.md). These instructions will guide you through assigning LUAD and Multi-cancer HPCs reported in the publication to your own cohort.

#### When I run the Leiden clustering step. I get an \'TypeError: can't pickle weakref objects\' error in some folds.
Based on experience, this error occurs with non-compatible version on numba, umap-learn, and scanpy. The package versions in the python environment should work.
But these alternative package combination works:
```
scanpy==1.7.1 
pynndescent==0.5.0 
numba==0.51.2
```

### If you are having any issue running these scripts, please leave a message on the Issues Github tab.

---

## TCGA HPL files
This section contains the following TCGA files produced by HPL:
1. TCGA WSI tile image datasets.
2. TCGA Self-supervised trained weights.
3. TCGA tile projections.
4. TCGA HPC configurations.
5. TCGA WSI & patient representations. 

For the New York University cohorts, please send reasonable requests to the corresponding authors.

### TCGA WSI tile image datasets
You can find the WSI tile images at:
1. [LUAD & LUSC](https://drive.google.com/drive/folders/18skVh8Vk6zoxG3Se5Vlb7a3EKP2xHXXd?usp=sharing)
2. [LUAD & LUSC 250K subsample](https://drive.google.com/drive/folders/1FuPkMnv6CiDe26doUXfEfQEWShgbmp9P?usp=sharing) for self-supervised model training.
3. [Multi-Cancer (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC)](https://drive.google.com/drive/folders/1CI99pwhWFQUgVlj3kFKYqcBECdl_xwnF?usp=share_link)
4. [Multi-Cancer (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC) 250K subsample](https://drive.google.com/drive/folders/1EfOZCXAwNheYCpIGS7nsee3SJkhNNEiy?usp=share_link) for self-supervised model training.

### TCGA Pretrained Models
Self-supervised model weights:
1. [LUAD & LUSC model](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020)
2. [Multi-Cancer model (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC)](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_PanCancer_5x/19949708)

### Other Pretrained Models
Self-supervised model weights:
1. UPenn Kidney model: [5X](https://figshare.com/articles/dataset/HPL_Kidney_5x/25715220), [10X]([https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020](https://figshare.com/articles/dataset/HPL_Kidney_10x/26270665), and [20X](https://figshare.com/articles/dataset/HPL_Kidney_20x/26270908)

### TCGA tile vector representations
You can find tile projections for TCGA LUAD and LUSC cohorts at the following locations. These are the projections used in the publication results.
1. [LUAD & LUSC tile vector representations (background and artifact tiles unfiltered)](https://drive.google.com/file/d/1_mXaTHAF6gb0Y4RgNhJCS2l9mgZoE7gR/view?usp=sharing)
2. [LUAD & LUSC tile vector representations](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view?usp=sharing)
3. [Multi-Cancer tile vector representations (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC)](https://drive.google.com/file/d/1u4FK45QrCjGS3FeWmOe2EspR6cfi7Y31/view?usp=share_link)

### TCGA HPC files
You can find HPC configurations used in the publication results at:
1. [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. [LUAD vs LUSC type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
3. [LUAD survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)
4. [Multi-cancer (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC)](https://drive.google.com/drive/folders/1RDxVR__cRiyJjRY2Cv8ofRcFz_cPK_aL?usp=share_link)

### TCGA WSI & patient vector representations
You can find WSI and patient vector representations used in the publication results at:
1. [LUAD vs LUSC type classification](https://drive.google.com/file/d/1K2Fteuv0UrTF856vnJMr4DSyrlqu_vop/view?usp=sharing)
2. [LUAD survival](https://drive.google.com/file/d/13P3bKcmD9C7fvEisArOVOTxf19ko6Xyv/view?usp=sharing)
3. [Multi-cancer (BLCA, BRCA, CESC, COAD, LUSC, LUAD, PRAD, SKCM, STAD, UCEC)](https://drive.google.com/file/d/1f9y4dtFfIMe9QsrKZnlk2BPfOt9zI73x/view?usp=share_link)

## Python Environment
The code uses Python 3.8 and the necessary packages can be found at [requirements.txt](./requirements.txt)

The flow uses TensorFlow 1.15 and according to [TensorFlows Specs](https://www.tensorflow.org/install/source#gpu) the closest CUDA and cuDNN version are `cudatoolkits==10.0` and `cudnn=7.6.0`. 
However, depending on your GPU card you might need to use `cudatoolkits==11.7` and `cudnn=8.0` instead. 
Newer cards with Ampere architecture (Nvidia 30s or A100s) would only work with CUDA 11.X, Nvidia maintains this [repo](https://github.com/NVIDIA/tensorflow), so you can use TensorFlow 1.15 with the new version of CUDA.

These commands should get the right environment to run HPL:
```
conda create -n HPL python=3.8 \ 
conda activate HPL \
python3 -m pip install --user nvidia-pyindex \
python3 -m pip install --user nvidia-tensorflow \
python3 -m pip install -r requirements.txt \
```

## Dockers
These are the dockers with the environments to run the steps of HPL. Step **'Leiden clustering'** needs to be run with docker [**2**], all other steps can be run with docker [**1**]:
1. **Self-Supervised models training and projections:**
    - [aclaudioquiros/tf_package:v16](https://hub.docker.com/r/aclaudioquiros/tf_package/tags)
2. **Leiden clustering:**
    - [gcfntnu/scanpy:1.7.0](https://hub.docker.com/r/gcfntnu/scanpy)

If you want to run the docker image in your local machine. These commands should get you up and running.
Please take into account that the image [aclaudioquiros/tf_package:v16](https://hub.docker.com/r/aclaudioquiros/tf_package/tags) uses CUDA 10.0, if your GPU card uses the Ampere architecture (Nvidia 30s or A100s) it won't work appropriately.   
In addition, if you want to run the [Step 6 - Leiden clustering in HPL](./README_HPL.md), you would need to change the image name:

```
docker run -it --mount src=`pwd`,target=/tmp/Workspace,type=bind aclaudioquiros/tf_package:v16
cd Workspace
# Command you want to run here.
```


