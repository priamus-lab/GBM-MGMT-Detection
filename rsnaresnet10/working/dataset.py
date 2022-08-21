import glob
import os
import re

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config
import utils


class BrainRSNADataset(Dataset):
    def __init__(
        self, data, transform=None, target="MGMT_value", mri_type="FLAIR", is_train=True, ds_type="forgot", do_load=False, folder="train"
    ):
        self.target = target
        self.data = data
        self.type = mri_type

        self.transform = transform
        self.is_train = is_train
        self.folder = f"../../RSNA-BTC-Datasets/{folder}" if self.is_train else "../../RSNA-BTC-Datasets/test"
        self.do_load = do_load
        self.ds_type = ds_type
        self.img_indexes = self._prepare_biggest_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        if str(row.BraTS21ID)[0].isnumeric():
            case_id = int(row.BraTS21ID)
            orig = case_id
        else:
            num = str(row.BraTS21ID).split("-")[2]
            case_id = int(f"1{num}")
            orig = row.BraTS21ID
        target = int(row[self.target])
        
        _3d_images = self.load_dicom_images_3d(orig)
        _3d_images = torch.tensor(_3d_images).float()
        if self.is_train:
            return {"image": _3d_images, "target": target, "case_id": case_id}
        else:
            return {"image": _3d_images, "case_id": case_id}

    def _prepare_biggest_images(self):
        big_image_indexes = {}
        if (f"big_image_indexes_{self.ds_type}.pkl" in os.listdir("."))\
            and (self.do_load) :
            print("Loading the best images indexes for all the cases...")
            big_image_indexes = joblib.load(f"big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes
        else:
            
            print("Caulculating the best scans for every case...")
            for row in tqdm(self.data.iterrows(), total=len(self.data)):
                #case_id = str(int(row[1].BraTS21ID)).zfill(5)
                #print(row)
                if str(row[1].BraTS21ID)[0].isnumeric():
                    case_id = str(int(row[1].BraTS21ID)).zfill(5)
                else:
                    case_id = row[1].BraTS21ID
                if self.folder == "../../RSNA-BTC-Datasets/all":
                    path1 = f"../../RSNA-BTC-Datasets/train/{case_id}/{self.type}/*.dcm"
                    files1 = sorted(
                        glob.glob(path1),
                        key=lambda var: [
                            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                        ],
                    )
                    path2 = f"../../RSNA-BTC-Datasets/UPENN-GBM/{case_id}/*_{self.type}/*.dcm"
                    files2 = sorted(
                        glob.glob(path2),
                        key=lambda var: [
                            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                        ],
                    )
                    files = files1 + files2
                elif self.folder == "../../RSNA-BTC-Datasets/train":
                    path = f"{self.folder}/{case_id}/{self.type}/*.dcm"
                    files = sorted(
                        glob.glob(path),
                        key=lambda var: [
                            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                        ],
                    )
                else:
                    path = f"{self.folder}/{case_id}/*_{self.type}/*.dcm"
                    files = sorted(
                        glob.glob(path),
                        key=lambda var: [
                            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                        ],
                    )
                #print(files)
                resolutions = [utils.extract_cropped_image_size(f) for f in files]
                #print(resolutions)
                if resolutions == []:
                    print(f"\nWARNING: {self.type} in {case_id} not found, skipping...")
                else:
                    middle = np.array(resolutions).argmax()
                    big_image_indexes[case_id] = middle

            joblib.dump(big_image_indexes, f"big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes



    def load_dicom_images_3d(
        self,
        case_id,
        num_imgs=config.NUM_IMAGES_3D,
        img_size=config.IMAGE_SIZE,
        rotate=0,
    ):
        #case_id = str(case_id).zfill(5)
        if str(case_id)[0].isnumeric():
            case_id = str(case_id).zfill(5)
        else:
            case_id = case_id

        #path = f"{self.folder}/{case_id}/{self.type}/*.dcm"
        if self.folder == "../../RSNA-BTC-Datasets/all":
            path1 = f"../../RSNA-BTC-Datasets/train/{case_id}/{self.type}/*.dcm"
            files1 = sorted(
                glob.glob(path1),
                key=lambda var: [
                    int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )
            path2 = f"../../RSNA-BTC-Datasets/UPENN-GBM/{case_id}/*_{self.type}/*.dcm"
            files2 = sorted(
                glob.glob(path2),
                key=lambda var: [
                    int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )
            files = files1 + files2
        elif self.folder == "../../RSNA-BTC-Datasets/train":
            path = f"{self.folder}/{case_id}/{self.type}/*.dcm"
            files = sorted(
                glob.glob(path),
                key=lambda var: [
                    int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )
        else:
            path = f"{self.folder}/{case_id}/*_{self.type}/*.dcm"
            files = sorted(
                glob.glob(path),
                key=lambda var: [
                    int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                ],
            )
        
        middle = self.img_indexes[case_id]

        # # middle = len(files) // 2
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        image_stack = [utils.load_dicom_image(f, rotate=rotate, voi_lut=True) for f in files[p1:p2]]
        
        img3d = np.stack(image_stack).T
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        return np.expand_dims(img3d, 0)
