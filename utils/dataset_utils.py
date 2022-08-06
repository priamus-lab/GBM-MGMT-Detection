import random
import os
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data
from torchvision import transforms, utils, datasets
from torch.nn import functional as F
from scipy.io import loadmat
from random import random, sample
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import math
from torch.utils.data import Dataset, ConcatDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

#set_seed(12)

def loader_fc(path):
    np.seterr(all='raise')
    x = loadmat(path)
    y_new = x['X'].astype(np.float32)
    if y_new.shape[0] > 192:
        y_new = y_new[32:224,32:224,32:224] #192x192x192
    # y_new = y_new[:,:,32:224,np.nexaxis]
    try:
        #print("path: " + path + " - y_new min: " + str(np.min(y_new)) + " - y_new max: " + str(np.max(y_new)))
        y_new = (y_new-np.min(y_new))/(np.max(y_new)-np.min(y_new))
        #print("path: " + path + " - y_new min: " + str(np.min(y_new)) + " - y_new max: " + str(np.max(y_new)))
        assert np.max(y_new) <= 1 #all(y_new <= 1) and all(y_new >= 0) 
    except:
        y_new = y_new
        #print("Division by zero, aborting...")
    return y_new

def loader_fc_3d_nifti(path):
    np.seterr(all='raise')
    x = nib.load(path)
    y_new = x.get_fdata().astype(np.float32)
    y_new = y_new[:,:,:,np.newaxis]
    scale_factors = x.header.get_zooms()
    return y_new, scale_factors

def loader_fc_3d(path):
    np.seterr(all='raise')
    #print(path)
    x = loadmat(path)
    y_new = x['X'].astype(np.float32)
    if y_new.shape[0] > 192:
        y_new = y_new[32:224,32:224,32:224] #192x192x192
    
    """
    try:
        #print("path: " + path + " - y_new min: " + str(np.min(y_new)) + " - y_new max: " + str(np.max(y_new)))
        y_new = (y_new-np.min(y_new))/(np.max(y_new)-np.min(y_new))
        #print("path: " + path + " - y_new min: " + str(np.min(y_new)) + " - y_new max: " + str(np.max(y_new)))
        assert np.max(y_new) <= 1 #all(y_new <= 1) and all(y_new >= 0) 
    except:
        print("Division by zero, aborting...")
    """
    y_new = y_new[:,:,:,np.newaxis]
    #y_new = y_new.unsqueeze(0)
    return y_new

def loader_fc_filtered(path, sel_slices):
    y_new = loader_fc(path)
    counter = np.zeros(y_new.shape[-1])
    for i in range(y_new.shape[-1]):
        counter[i] = np.count_nonzero(y_new[:,:,i])
    arg_sort_list = np.argsort(counter)
    top = arg_sort_list[-sel_slices:]
    top = np.sort(top)
    y_new = y_new[:,:,top]
    
    return y_new, top

def loader_fc_specific(path, slice_num):
    y_new = loader_fc(path)
    y_new = y_new[:,:,slice_num]
    
    return y_new

#choice = transforms.RandomChoice([Random_Rotation(p=0.5, n=1),
#    Random_Rotation(p=0.5, n=2),
#    Random_Rotation(p=0.5, n=3)
#])

class ToTensor3D(torch.nn.Module):  #-> ok
    def __init__(self):
        super().__init__()

  
    def forward(self, tensor):
        y_new = torch.from_numpy(tensor.transpose(3,2,0,1))
        #y_new = torch.from_numpy(tensor)
        return y_new

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Normalize3D(torch.nn.Module):
    def __init__(self, norm_type='min_max'):
        self.norm_type = norm_type                               
        super().__init__() 
        
    def forward(self, img):
        if self.norm_type == 'z_score':
            mean, std = torch.mean(img), torch.std(img)
            img -= mean
            img /= std+1e-8
        elif self.norm_type == 'min_max':
            try:
                nmin, nmax = torch.min(img), torch.max(img)
                inp = (img - nmin) / (nmax-nmin+1e-8)
                assert torch.max(inp) <= 1
            except:
                print("Division by zero, aborting...")
            img = inp

        return img
    
class AdjustContrast(torch.nn.Module):
    def __init__(self, mult=2.0):
        self.mult = mult
        super().__init__()
        
    def forward(self, img):
        #img = transforms.adjust_contrast(img, contrast_factor=mult)
        nmin, nmax = torch.min(img), torch.max(img)
        inp = (img - nmin) / (nmax-nmin+1e-8)
        img = inp**self.mult*(nmax-nmin)+nmin
        return img
    
class RandomRotation3D(torch.nn.Module):
    def __init__(self, p=0.5, n=1):
        self.p = p                        #probabilità di effettuare la rotazione
        self.n = n                        #numero di 90 gradi           
        super().__init__()         

    def forward(self, img):
        if random() < self.p:
            img = torch.rot90(img,self.n,dims=[2,3])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, n={})'.format(self.p, self.n)
    
class Rotation3D90Degrees(torch.nn.Module):
    def __init__(self, times=1, axis=1):              
        self.times = times                        #numero di 90 gradi   
        self.axis = axis
        super().__init__()         

    def forward(self, img):
        axes = [1,2,3]
        axes.remove(self.axis)
        img = torch.rot90(img,self.times,dims=axes)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(times={}, axis={})'.format(self.times, self.axis)

#------------------------------ 

class RandomZFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        self.p = p                        #probabilità di effettuare il flip  
        super().__init__()                 

    def forward(self, img):
        if random() < self.p:
            img = torch.flip(img, [1])
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class Resize3D(torch.nn.Module):  #-> ok
    def __init__(self, size=(32,32,32)):
        self.size = size          
        super().__init__()         

    def forward(self, tensor):
        #print(tensor.shape)
        #print(tensor.unsqueeze(0).shape)
        #align_corners = True, 
        img = F.interpolate( tensor.unsqueeze(0), self.size, mode='area').squeeze(0)

        #print(img.shape)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)
    
class ToFake3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        img = img.squeeze(0)
        return img
    
class Pad3D(torch.nn.Module):  #-> ok
    def __init__(self, size=(32,32,32)):
        self.size = size          
        super().__init__()         

    def forward(self, tensor):
        #print(tensor.shape)
        #print(tensor.unsqueeze(0).shape)
        pad = (int(math.ceil((self.size[2]-tensor.shape[3])/2)), int(math.floor((self.size[2]-tensor.shape[3])/2)),
               int(math.ceil((self.size[1]-tensor.shape[2])/2)), int(math.floor((self.size[1]-tensor.shape[2])/2)),
               int(math.ceil((self.size[0]-tensor.shape[1])/2)), int(math.floor((self.size[0]-tensor.shape[1])/2))
              )

        img = F.pad(tensor, pad, "constant", 0)

        #print(img.shape)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)
    
def get_transform(dims):
    if dims == 3:
        return get_transform_3d()
    elif dims == 2:
        return get_transform_2d()
    
    return get_transform_3d()
    
def get_transform_2d():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90)
    ])
    
    return transform

def get_transform_3d():
    choice = transforms.RandomChoice([RandomRotation3D(p=0.5, n=1),
                                    RandomRotation3D(p=0.5, n=2), 
                                    RandomRotation3D(p=0.5, n=3)])
    
    transform = transforms.Compose([#ToTensor3D(),
        #Resize3D(size=(128,128,128)),
        
        #transforms.CenterCrop((192,192))#,
        #transforms.Resize((128,128))
        #transforms.Normalize([0.5 for _ in range(0,256)], [0.5 for _ in range(0,256)]) # mean = 0.5, std = 0.5
        AdjustContrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomZFlip(),
        choice
        #transforms.RandomRotation(),
        #choice
    ])

    return transform
"""
def get_tio_transform():
    transform = tio.Compose([
        tio.transforms.Resize([128,128,128])
    ])
    return transform
"""
class Dataset(torch_data.Dataset):   
    def __init__(self, root="", sub_dirs=[], list_classes=[], label_smoothing=0.01, transform=None, target_transform=None, ext="mat", dims=3, sel_slices=None, filter_list=None):
        self.root = root 
        self.sub_dirs = sub_dirs
        self.transform = transform
        #self.is_valid_file = is_valid_file
        self.list_classes = list_classes
        #self.samples_1, self.samples_2 = self.__get_samples()
        self.samples_dict = self.__get_samples()
        self.label_smoothing = label_smoothing
        self.ext = ext
        self.dims = dims
        self.sel_slices = sel_slices
        
    def concat_datasets(self, dataset_1, dataset_2=None, import_transform=True, balance=True):
        self.root = dataset_1.root 
        self.sub_dirs = dataset_1.sub_dirs
        if import_transform:
            self.transform = dataset_1.transform
        #self.is_valid_file = is_valid_file
        if dataset_2 is None:
            self.list_classes = dataset_1.list_classes
            self.samples_dict = dataset_1.samples_dict
        else:
            self.list_classes = dataset_1.list_classes + dataset_2.list_classes
            if balance:
                self.samples_dict = self.balanced_datasets(dataset_1, dataset_2)#self.__get_samples()
            else:
                self.samples_dict = self.merged_datasets(dataset_1, dataset_2)
        print(f"Length of concatenated dataset: {len(self.samples_dict[self.sub_dirs[0]])}")
        self.label_smoothing = dataset_1.label_smoothing
        self.ext = dataset_1.ext
        self.dims = dataset_1.dims
        self.sel_slices = dataset_1.sel_slices
        return self
    
    def balanced_datasets(self, dataset_1, dataset_2):
        datasets = [dataset_1, dataset_2]
        l = max([len(dataset.samples_dict[self.sub_dirs[0]]) for dataset in datasets])
        for dataset in datasets:
            while len(dataset.samples_dict[self.sub_dirs[0]]) < l:
                for i in range(len(dataset.samples_dict)):
                    dataset.samples_dict[self.sub_dirs[i]] += sample(dataset.samples_dict[self.sub_dirs[i]], min(len(dataset.samples_dict[self.sub_dirs[i]]), l - len(dataset.samples_dict[self.sub_dirs[i]])))
        bal_dataset_dict = {}
        for i in range(len(dataset.samples_dict)):          
            #bal_dataset_list.append(datasets[0].samples_dict[self.sub_dirs[i]] + datasets[1].samples_dict[self.sub_dirs[i]])
            bal_dataset_dict[self.sub_dirs[i]] = datasets[0].samples_dict[self.sub_dirs[i]] + datasets[1].samples_dict[self.sub_dirs[i]]
        return bal_dataset_dict
    
    def merged_datasets(self, dataset_1, dataset_2):
        datasets = [dataset_1, dataset_2]
        dataset_dict = {}
        for i in range(len(dataset_1.samples_dict)):          
            #bal_dataset_list.append(datasets[0].samples_dict[self.sub_dirs[i]] + datasets[1].samples_dict[self.sub_dirs[i]])
            dataset_dict[self.sub_dirs[i]] = datasets[0].samples_dict[self.sub_dirs[i]] + datasets[1].samples_dict[self.sub_dirs[i]]
        return dataset_dict
    
    def tensor_transform(self, sample):
        if self.dims == 3:
            t_transform = transforms.Compose([
                                            ToTensor3D(),
                                            Normalize3D(norm_type='min_max')
            ])
        elif self.dims == 2:
            mean, std = np.mean(sample), np.std(sample)
            t_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            #transforms.Normalize(mean=mean, std=std)
            ])
        
        return t_transform(sample)
    
    def to_fake_3d_transform(self, sample):
        f_transform = transforms.Compose([ToFake3D()])
        return f_transform(sample)
    
    def isometric_transform(self, sample, scale_factors):
        #print(scale_factors)
        #print(sample.shape)
        # With Tensor3D Transformation, idx 0 becomes 2, idx 1 becomes 0 and idx 2 becomes 1
        s1 = int(sample.shape[1]*scale_factors[2])
        s2 = int(sample.shape[2]*scale_factors[0])
        s3 = int(sample.shape[3]*scale_factors[1])
        transform = transforms.Compose([Resize3D(size=(s1,s2,s3)),
                                        Rotation3D90Degrees(times=1, axis=1),
                                        Rotation3D90Degrees(times=3, axis=3),
                                        Pad3D(size=(192,192,192))
        ])
        return transform(sample)
          
    def __len__(self):
        return len(self.samples_dict[self.sub_dirs[0]])
    
    def __get_samples(self):
        # TO FIX
        ListDict = {}
        for i in range(len(self.sub_dirs)):
            ListFiles=[]
            for c in self.list_classes:
                listofFiles = os.listdir(self.root + '/' + self.sub_dirs[i] + '/' + c)
                listofFiles.sort()
                for file in listofFiles:
                    #ListFiles1.append((self.root + '/' + self.sub_dir_1 + '/' + c + '/' + file, self.list_classes.index(c))) 
                    #img_id = file.replace(".mat","")
                    #if self.filter_list is None or img_id in self.filter_list:
                    if c.isdigit():
                        ListFiles.append((self.root + '/' + self.sub_dirs[i] + '/' + c + '/' + file, int(c)))
                    else:
                        ListFiles.append((self.root + '/' + self.sub_dirs[i] + '/' + c + '/' + file, str(c)))

            ListFiles.sort(key=lambda tup: tup[0])
            ListDict[self.sub_dirs[i]] = ListFiles

        return ListDict
        
    def __getid__(self, index):
        path, target = self.samples_dict[self.sub_dirs[0]][index]

        img_id = os.path.basename(path)
            
        if img_id[:5].isnumeric():
            return img_id[:5]
        else:
            img_id = img_id.replace("_FLAIR","").replace("_T1w","").replace("_T1wCE","").replace("_T2w","").replace(".mat","")
            return img_id
        
    def __gettarget__(self, index):
        path, target = self.samples_dict[self.sub_dirs[0]][index]

        return target
        
    
    def __getitem__(self, index):
        img_id_list = []
        sample_list = []
        target_list = []
        top_idx = -1
        i = 0
        for key, samples in self.samples_dict.items():
            path, target = samples[index]

            if self.ext == "mat":
                if self.dims == 3:
                    sample = loader_fc_3d(path)
                elif self.dims == 2:
                    if self.sel_slices is not None:
                        if i == 0 and key == "KLF" and len(self.samples_dict.keys()) > 1:
                            _, top = loader_fc_filtered(path, self.sel_slices)
                            top_idx = top
                            #print("KLF scans are taken to retrieve top slice index by pixel count, but scans themselves won't be used")
                            i += 1
                            continue
                        elif len(self.samples_dict.keys()) > 1:
                            sample = loader_fc_specific(path, top_idx)
                        else:
                            sample, _ = loader_fc_filtered(path, self.sel_slices)
                    else:
                        sample = loader_fc(path)
            elif self.ext == "nii":
                sample, sf = loader_fc_3d_nifti(path)

            img_id = os.path.basename(path)
            #print(img_id)

            #if self.targets is None:
            #    data = loader_fc(path)
            #else:
            #    if self.augment:
            #        rotation = np.random.randint(0,4)
            #    else:
            #        rotation = 0
            #    data = loader_fc(path, rotate=rotation)

            sample = self.tensor_transform(sample)

            if self.ext == "nii":
                sample = self.isometric_transform(sample, sf)

            if self.transform is not None:
                sample = self.transform(sample)
                
            #if self.dims == 2:
            #    sample = self.to_fake_3d_transform(sample)
                
            img_id_list.append(img_id)
            sample_list.append(sample)
            target_list.append(target)
            i += 1
        
        # TO FIX
        """
        if len(self.samples_dict) == 2: 
            return img_id_list[0], img_id_list[1], sample_list[0], sample_list[1], target_list[0], target_list[1]
        else:
            return img_id_list[0], sample_list[0], target_list[0]
        """
        return_array = []
        return_array.append(img_id_list)
        return_array.append(sample_list)
        return_array.append(target)
        return tuple(return_array)

    
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def plot_slices(dataset, mri_type, out_class):
    mri_type = ""
    if out_class == 0:
        print(f"Plotting {mri_type} slices of patients without tumor...")
    else:
        print(f"Plotting {mri_type} slices of patients with tumor...")
            
    factor = 8
    #fig = plt.figure(figsize=(16,16*int(math.ceil(len(dataset)/factor))))
    #fig.suptitle("FLAIR slices of patients without tumor")
    
    cols = 5
    rows = len(dataset)
    mri_num = dataset.sub_dirs.index(mri_type)
    for i in tqdm(range(rows)):
        fig = plt.figure(figsize=(5*32,1*32))
        dataset_item = dataset.__getitem__(i)
        img = dataset_item[1][mri_num].squeeze(0)
        for j in range(cols):
            ax = fig.add_subplot(1,5,j+1)
            #ax.title.set_text(str(dataset_item[0+idx])+" "+str(32+32*j)+"/192")
            plt.imshow(img[32+32*j], cmap=cm.Greys_r) #32,64,96,128,160
        plt.savefig("plotted_slices/"+str(out_class)+"_"+str(dataset_item[0][mri_num][:-4])+".jpg",dpi=64)
        plt.show()
    #plt.show()

def plot_single_slice(dataset, mri_type, patient_id, slice_num):
    mri_num = dataset.sub_dirs.index(mri_type)
    for i in range(len(dataset)):
        dataset_id = dataset.__getid__(i)
        if dataset_id == patient_id:
            dataset_item = dataset.__getitem__(i)
            img = dataset_item[1][mri_num].squeeze(0)
            fig = plt.figure(dataset_item[0][mri_num]+": "+str(dataset_item[2])+" (slice n."+str(slice_num)+"/"+str(len(img))+")")
            print(dataset_item[0][mri_num]+": "+str(dataset_item[2])+" (slice n."+str(slice_num)+"/"+str(len(img))+")")
            plt.imshow(img[slice_num], cmap=cm.Greys_r)
            plt.show()
            return
    print("Patient not found in this dataset")
    
def get_single_slice(dataset, mri_type, patient_id, slice_num, axis=0):
    mri_num = dataset.sub_dirs.index(mri_type)
    for i in range(len(dataset)):
        dataset_id = dataset.__getid__(i)
        if dataset_id == patient_id:
            dataset_item = dataset.__getitem__(i)
            img = dataset_item[1][mri_num].squeeze(0)
            # Return image and target
            #if mri_type == "KLF":
            #    img = dataset_item[1][mri_num]
            #    return img.numpy().transpose(2,3,1,0)[slice_num,:,:], dataset_item[2]
            if axis == 0:
                return img[slice_num,:,:].numpy(), dataset_item[2]
            elif axis == 1:
                return img[:,slice_num,:].numpy(), dataset_item[2]
            elif axis == 2:
                return img[:,:,slice_num].numpy(), dataset_item[2]
    print("Patient not found in this dataset")
    
def get_patient_data(dataset, mri_type, patient_id):
    mri_num = dataset.sub_dirs.index(mri_type)
    for i in range(len(dataset)):
        dataset_id = dataset.__getid__(i)
        if dataset_id == patient_id:
            dataset_item = dataset.__getitem__(i)
            img = dataset_item[1][mri_num].squeeze(0)
            return img.numpy(), dataset_item[2]
    print("Patient not found in this dataset")
    
class FCM():
    def __init__(self, image, image_bit, n_clusters, m, epsilon, max_iter):
        '''Modified Fuzzy C-means clustering
        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <max_iter>: int, max number of iterations.
        '''

        #-------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.shape = image.shape # image shape
        self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1) 
        self.numPixels = image.size
       
    #--------------------------------------------- 
    def initial_U(self):
        U=np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx%self.n_clusters==ii
            U[idxii,ii] = 1        
        return U
    
    def update_U(self):
        '''Compute weights'''
        c_mesh,idx_mesh = np.meshgrid(self.C,self.X)
        power = 2./(self.m-1)
        p1 = abs(idx_mesh-c_mesh)**power
        p2 = np.sum((1./abs(idx_mesh-c_mesh))**power,axis=1)
        
        return 1./(p1*p2[:,None])

    def update_C(self):
        '''Compute centroid of clusters'''
        numerator = np.dot(self.X,self.U**self.m)
        denominator = np.sum(self.U**self.m,axis=0)
        return numerator/denominator
                       
    def form_clusters(self):      
        '''Iterative training'''        
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:             
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                #print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                #print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        self.segmentImage()


    def deFuzzify(self):
        return np.argmax(self.U, axis = 1)

    def segmentImage(self):
        '''Segment image based on max weights'''

        result = self.deFuzzify()
        self.result = result.reshape(self.shape).astype('int')

        return self.result
    


def create_split_loaders(dataset, dataset_no_tr, split, batch_size):
    g_cpu = torch.Generator()
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    if len(split) == 3:
        test_folds_idx = split[2]
    train_sampler = SubsetRandomSampler(train_folds_idx, g_cpu)
    valid_sampler = SubsetRandomSampler(valid_folds_idx, g_cpu)
    if len(split) == 3:
        test_sampler = SubsetRandomSampler(test_folds_idx, g_cpu)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, worker_init_fn=np.random.seed(0))
    valid_loader = DataLoader(dataset=dataset_no_tr, batch_size=batch_size, sampler=valid_sampler, num_workers=2, worker_init_fn=np.random.seed(0))
    if len(split) == 3:
        test_loader = DataLoader(dataset=dataset_no_tr, batch_size=batch_size, sampler=test_sampler, num_workers=2, worker_init_fn=np.random.seed(0))
        return (train_loader, valid_loader, test_loader) 
    return (train_loader, valid_loader)    

def get_all_split_loaders(dataset, dataset_no_tr, cv_splits, batch_size):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to 
                 be used in each fold for each split.
    batch_size -- batch size.
    
    """
    split_samplers = []
    
    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset,
                                 dataset_no_tr,
                                 cv_splits[i], 
                                 batch_size)
        )
    return split_samplers

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_splits(dataset_0, dataset_1, val_total_ratio, is_k_fold, test_total_ratio=0.1, k=5):
    if type(is_k_fold) == str:
        is_k_fold = str2bool(is_k_fold)
    if not is_k_fold:
        if dataset_1 is not None:
            dataset_0_size = len(dataset_0)
            dataset_1_size = len(dataset_1)
            dataset_0_indices = list(range(dataset_0_size))
            dataset_1_indices = list(range(dataset_0_size, dataset_1_size+dataset_0_size))
        else:
            dataset_0_size = len(dataset_0)
            dataset_1_size = 0
            dataset_0_indices = list(range(dataset_0_size))
            dataset_1_indices = list(range(dataset_0_size, dataset_0_size))
        np.random.seed(0)
        np.random.shuffle(dataset_0_indices)
        np.random.shuffle(dataset_1_indices)
        val_split_0_index = int(np.floor(val_total_ratio * dataset_0_size))
        val_split_1_index = int(np.floor(val_total_ratio * dataset_1_size))
        test_split_0_index = int(np.floor(test_total_ratio * dataset_0_size))
        test_split_1_index = int(np.floor(test_total_ratio * dataset_1_size))
        train_idx = dataset_0_indices[val_split_0_index+test_split_0_index:] + dataset_1_indices[val_split_1_index+test_split_1_index:]
        val_idx = dataset_0_indices[:val_split_0_index] + dataset_1_indices[:val_split_1_index]
        test_idx = dataset_0_indices[val_split_0_index:val_split_0_index+test_split_0_index] + dataset_1_indices[val_split_1_index:val_split_1_index+test_split_1_index]
        print("Train Idx:")
        print(train_idx)
        print("Val Idx:")
        print(val_idx)
        print("Test Idx:")
        print(test_idx)

        split = (train_idx, val_idx, test_idx)
        splits = [split]
    else:
        s_fold = StratifiedKFold(n_splits=k, random_state=518, shuffle=True)
        base_splits = []
        splits = []
        dataset_0_size = len(dataset_0)
        dataset_1_size = len(dataset_1)
        dataset_0_indices = list(range(dataset_0_size))
        dataset_1_indices = list(range(dataset_0_size, dataset_1_size+dataset_0_size))
        X_placeholder = np.array(dataset_0_indices + dataset_1_indices)
        y_placeholder = np.concatenate((np.zeros(dataset_0_size), np.ones(dataset_1_size)), axis=None)
        
        for train_idx, valtest_idx in s_fold.split(X_placeholder, y_placeholder):
            base_splits.append((train_idx, valtest_idx))
            
        splits = base_splits
        """
        i = -1
        for split in base_splits:
            train_idx = split[0]
            val_idx = split[1]
            test_idx = base_splits[i][1]
            train_idx = np.setdiff1d(train_idx,test_idx)
            i+=1
            splits.append((train_idx,val_idx,test_idx))
        """
    return splits