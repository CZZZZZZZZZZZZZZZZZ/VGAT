# VGAT: A Cancer Survival Analysis Framework Transitioning from Generative Visual Question Answering to Genomics Reconstruction
The PyTorch implementation of Vision Genomic Answering-Guided Transformer (VGAT) as described in the paper "VGAT: A Cancer Survival Analysis Framework Transitioning from Generative Visual Question Answering to Genomics Reconstruction."


## Installation Guide for Linux (using anaconda)
### Pre-requisities: 
- Linux (Tested on Ubuntu 22.04)
- NVIDIA GPU A6000*4 with CUDA 11.8 and cuDNN 8.6
- Python (3.10.14), numpy (1.25.0), pandas (2.2.2), torch (2.3.1), scikit-learn (1.5.0), scipy (1.13.1),
- （Used for obtaining BulkRNA embedding） jax(0.4.19), jaxlib(0.4.19+cudnn86), joblib(1.3.2), dm-haiku(0.0.10), pydantic(1.10.5)

### Downloading TCGA Data and Pathways Compositions 
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer  to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov)and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). 
## Processing Whole Slide Images 
To process Whole Slide Images (WSIs), first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, an SSL pretrained Swin Transformer [CTransPath](https://github.com/Xiyue-Wang/TransPath) is used to encode raw image patches into 768-dim feature vectors, which we then save as .pt files for each WSI. The extracted features then serve as input (in a .pt file) to the network. All pre-processing of WSIs is done using the [CLAM toolbox](https://github.com/mahmoodlab/CLAM).

## Transcriptomics and Pathway Compositions
We downloaded raw RNA-seq abundance data for the TCGA cohorts from the [Xena database](https://www.nature.com/articles/s41587-020-0546-8) and performed normalization in the dataset class. The raw data is included as CSV files [`datasets_csv`](https://github.com/ajv012/SurvPath/tree/main/datasets_csv/raw_rna_data/combine). Xena database was also used to access disease specific survival and associated censorhsip. Using the Reactome and MSigDB Hallmarks pathway compositions, we selected pathways that had more than 90% of transcriptomics data available. The compositions can be found at [`metadata`](https://github.com/ajv012/SurvPath/blob/main/datasets_csv/metadata/combine_signatures.csv).  



