import os
import sys 
import json
import glob
import random
import collections
import time
import re
import gc
import tracemalloc

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.spatial.distance import cdist

import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import logging
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from torch.autograd import Variable
from torch.optim import *
from scipy.io import loadmat
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, ConcatDataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import matplotlib.cm as cm
import matplotlib.animation as animation
#import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default = "notebook"
import math
import datetime
import timeit

from scipy import io
import os

sys.path.append("..")
from utils.dataset_utils import *
from utils.classifier_utils import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

def retrieve_patients_list_from_datasets(dataset_0_mat, dataset_1_mat, dataset_test_mat):
    list_0 = []
    list_1 = []
    list_test = []

    print("Train - Unmethylated Patients:")
    for i in range(len(dataset_0_mat)):
        #print(str(i)+" - "+dataset_0_flair_t2w.__getitem__(i)[0]+": "+str(dataset_0_flair_t2w.__getitem__(i)[4]))
        print(dataset_0_mat.__getid__(i))
        list_0.append(dataset_0_mat.__getid__(i))
        #print("---")
        #fig = plt.figure(dataset_flair_t2w.__getitem__(i)[0+idx]+": "+str(dataset_flair_t2w.__getitem__(i)[4+idx]))
        #plt.imshow(dataset_flair_t2w.__getitem__(i)[2+idx][128], cmap=cm.Greys_r, animated=True)

    #plt.show()
    print("Train - Methylated Patients:")
    for i in range(len(dataset_1_mat)):
        #print(str(i)+" - "+dataset_0_flair_t2w.__getitem__(i)[0]+": "+str(dataset_0_flair_t2w.__getitem__(i)[4]))
        print(dataset_1_mat.__getid__(i))
        list_1.append(dataset_1_mat.__getid__(i))
        #print("---")

    print("Test - Unknown Patients:")
    for i in range(len(dataset_test_mat)):
        #print(str(i)+" - "+dataset_0_flair_t2w.__getitem__(i)[0]+": "+str(dataset_0_flair_t2w.__getitem__(i)[4]))
        print(dataset_test_mat.__getid__(i))
        list_test.append(dataset_test_mat.__getid__(i))
        #print("---")

    patients_dict = {
        "train_0": list_0,
        "train_1": list_1,
        "test": list_test
    }

    return patients_dict

def filter_slice(X, is_T1w=False):
    X_th = X.copy()
    X_th_f = X_th.flatten()
    X_top = X.copy()
    X_top_f = X_top.flatten()

    patches = plt.hist(X_th_f[X_th_f>np.min(X_th_f)+0.1], bins=100)
    n0 = [p for p in patches[0]]
    n1 = [p for p in patches[1]]
    
    m_hist = n1[np.argmax(n0)]
    
    if is_T1w:
        X_th_f = np.where((X_th_f < 0.1) | (X_th_f > m_hist), 0, 1)
    else:
        X_th_f[X_th_f<m_hist] = 0
        X_th_f[X_th_f>=m_hist] = 1
    
    X_th = X_th_f.reshape(192,-1)
    
    #X_th[X_th<m_hist] = 0
    #X_th[X_th>=m_hist] = 1
    #X_th = X_th.reshape(192,-1)
    
    if is_T1w:
        top_idxs = np.argsort(X_top_f)[:(np.count_nonzero(X==0)+np.count_nonzero(X_th)//2)]
        m_top = X_top_f[top_idxs[-1]]
        #top_20 = np.max(X_top_f)*4/5
        X_top_f = np.where((X_top_f < 0.1) | (X_top_f > m_top), 0, 1)
        #X_top_f[X_top_f<m_top] = 1
        #X_top_f[X_th_f>=m_top] = 0
    else:
        top_idxs = np.argsort(X_top_f)[-(np.count_nonzero(X_th)//3):]
        m_top = X_top_f[top_idxs[0]]
        #top_20 = np.max(X_top_f)*4/5
        X_top_f[X_top_f<m_top] = 0
        X_top_f[X_top_f>=m_top] = 1
    
    #top_idxs = np.argsort(X_top_f)[-(np.count_nonzero(X_th_f)//4):]
    #m_top = X_top_f[top_idxs[0]]
    #X_top_f[X_top_f<m_top] = 0
    #X_top_f[X_top_f>=m_top] = 1
    X_top = X_top_f.reshape(192,-1).astype(np.int16)
    
    plt.clf()
    plt.close()
    #del patches
    #del n0
    #del n1
    #del X_th
    #del X_th_f
    #del X_top_f
    #del top_idxs
    patches = None
    n0 = None
    n1 = None
    X_th = None
    X_th_f = None
    X_top_f = None
    top_idxs = None
    X = None
    m_hist = None
    m_top = None

    return X_top

def get_filtered_volume(dataset, patient_id, sub_dirs):
    X_dict = {}
    X_top_dict = {}
    for i in range(len(sub_dirs)):
        #X_i = []
        X_top_i = []
        X, target = get_patient_data(dataset, sub_dirs[i], patient_id)
        X = X - np.min(X)
        for j in range(192):
            #X, target = get_single_slice(dataset, dataset.sub_dirs[i], patient_id, j)
            #X = X - np.min(X)
            #X_i.append(X)
            is_T1w = sub_dirs[i] == "T1w" or sub_dirs[i] == "T1wCE"
            X_th = filter_slice(X[j], is_T1w=is_T1w)
            X_top_i.append(X_th)
            
        #X_i = np.stack(X,axis=0).T
        X_top_i = np.stack(X_top_i,axis=0)
        X_dict[sub_dirs[i]] = X
        X_top_dict[sub_dirs[i]] = X_top_i
        
        #del X
        #del X_top_i
        X = None
        X_top_i = None

        #print(f"{sub_dirs[i]}: {target}")
        
    X_int = np.where((X_top_dict["FLAIR"] == X_top_dict["T1w"]), X_top_dict["FLAIR"], 0)
    X_fil = np.where((X_int == 1), X_dict["FLAIR"], 0)
    #del X_dict
    #del X_int
    #del X_top_dict
    X_dict = None
    X_int = None
    X_top_dict = None
    #gc.collect()
    
    return X_fil, target

#https://nih.figshare.com/articles/dataset/MRI_dataset_supporting_Comparison_of_T1-Post_and_FLAIR-Post_MRI_for_identification_of_traumatic_meningeal_enhancement_in_traumatic_brain_injury_patients_/12386102
# https://openneuro.org/datasets/ds000221/versions/1.0.0

def save_kb_volumes_as_mat(dataset, folder, target, patient_ids, sub_dirs, main_path):
    for patient_id in tqdm(patient_ids):
        if not os.path.exists(f"{main_path}/{folder}_mat/KLF/{target}/{patient_id}_KLF.mat"):
            img3d, t = get_filtered_volume(dataset, patient_id, sub_dirs)
            data = {'X': img3d, 'y': target}
            if not os.path.exists(f"{main_path}/{folder}_mat"):
                os.makedirs(f"{main_path}/{folder}_mat")
            if not os.path.exists(f"{main_path}/{folder}_mat/KLF"):
                os.makedirs(f"{main_path}/{folder}_mat/KLF")  
            if not os.path.exists(f"{main_path}/{folder}_mat/KLF/{target}"):
                os.makedirs(f"{main_path}/{folder}_mat/KLF/{target}")  
            if not os.path.exists(f"{main_path}/{folder}_mat/KLF/{target}/{patient_id}_KLF.mat"):
                io.savemat(f"{main_path}/{folder}_mat/KLF/{target}/{patient_id}_KLF.mat", data, do_compression=True)
            #del img3d
            #del data
            img3d = None
            data = None
            t = None
            
            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')
            #for stat in top_stats:
            #    print(stat)
            