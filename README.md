# GBM-MGMT-Detection

# A Multimodal Knowledge-based Deep Learning Approach for MGMT Promoter Methylation Identification
Official implementation of the A Multimodal Knowledge-based Deep Learning Approach for MGMT Promoter Methylation Identification


## Prerequisites
- PyTorch Stable Version (https://pytorch.org/get-started/locally/)
- Numpy Package (https://numpy.org/)
- Cuda Toolkit https://developer.nvidia.com/cuda-downloads


## Running the code

Before discussing what is inside this repository, it is important to acknowledge that some preliminar path fixing and dataset downloading occur in order to execute notebooks without errors. Typically, many files rely on a path called *"../../RSNA-BTC-Datasets"*, which was the former folder containing all datasets and results. Datasets can be downloaded by registering to the [RSNA MICCAI Brain Tumor Radiogenomic Classification Challenge](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification) and from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642#70225642c94d520b7b5f42e7925602d723412459).

### Knowledge-Based Filtering
Once downloaded all the datasets in any desired folder, it is required to preprocess .dcm files through notebooks inside the folder *KnowledgeBasedFiltering*. In particular, the first process to perform is to convert data into .mat format, with the notebook called *nii_and_mat_datasets_extraction.ipynb*, in order to save lighter and already isotropic data for future usage and for KB-Filtering.

Converted data can be then elaborated with the notebook called *knowledge_filter.ipynb*. The output of this process is a new dataset that can be used for both model training and model inference.

### Inferences with 2D and 3D 5-Fold Models
Folders called *2DFold* and *3DFold* contain inference notebooks and PyTorch model weights. Datasets will be splitted into 5 folds thanks to .txt files called *train_fold.txt* and *upenn_train_fold.py* included in this repository. Model weights included in this repository are made from these same folds, so that inferences always have the same results.

Results and metrics of inferences are saved in the folder called *pred_metrics*.

## Other Notebooks
The main flow is included in the folders described before. However, there are several other functions available to the user, for instance model training, XAI analysis, 3D volumes visualization, inference with other models and much more. All the notebooks containing these functionalities are included in the folder *OtherNotebooks*.

## Authors

* **Salvatore Capuozzo, Michela Gravina, Gianluca Gatta, Stefano Marrone and Carlo Sansone**

if you use this code, please cite: Capuozzo, S., Gravina, M., Gatta, G., Marrone, S., & Sansone, C. (2022, September). [A Multimodal Knowledge-based Deep Learning Approach for MGMT Promoter Methylation Identification](https://doi.org/link_al_paper_pubblicato). In Journal of Imaging (MDPI).
