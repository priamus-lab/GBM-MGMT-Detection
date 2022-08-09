import sys
#sys.path.append("..")
from utils.dataset_utils import *
from utils.classifier_utils import *

import os

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

#import config
#from dataset import BrainRSNADataset

dir_path = "../RSNA-BTC-Datasets/train_mat"
test_dir_path = "../RSNA-BTC-Datasets/test_mat"
tumor_only_dir_path = "../RSNA-BTC-Datasets/ec_train_mat"
tumor_only_test_dir_path = "../RSNA-BTC-Datasets/ec_test_mat"
no_tumor_dir_path = "../RSNA-BTC-Datasets/no_tumor_train_mat"
#ext_test_1_dir_path = "../../RSNA-BTC-Datasets/brats18_mat"
#ext_test_0_dir_path = "../../RSNA-BTC-Datasets/OpenNeuroDS000221_ss_mat"
new_dir_path = "../RSNA-BTC-Datasets/UPENN-GBM_mat"

parser = argparse.ArgumentParser()
parser.add_argument("--even", default=True, type=bool)
args = parser.parse_args()

def generate_datasets(types):
    data_packs = {}
    ext = "mat"
    transform = None
    dims = 3
    sel_slices = None
    for t in types:
        print("Type: "+t)
        # Competition Train + Val + Test
        m_dataset_0 = Dataset(dir_path, [t], list_classes=["0"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

        logging.info("Train/Val datasets size: {}".format(len(m_dataset_0)))

        m_dataset_1 = Dataset(dir_path, [t], list_classes=["1"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)
        
        logging.info("Train/Val datasets size: {}".format(len(m_dataset_1)))

        # External Train + Val + Test
        #t_dataset_0 = Dataset(ext_test_0_dir_path, [t], list_classes=["0"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

        #logging.info("Train/Val datasets size: {}".format(len(t_dataset_0)))

        #t_dataset_1 = Dataset(ext_test_1_dir_path, [t], list_classes=["1"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

        #logging.info("Train/Val datasets size: {}".format(len(t_dataset_1)))

        # UPENN Train + Val + Test
        n_dataset_0 = Dataset(new_dir_path, [t], list_classes=["0"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

        logging.info("Train/Val datasets size: {}".format(len(n_dataset_0)))

        n_dataset_1 = Dataset(new_dir_path, [t], list_classes=["1"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)
        
        logging.info("Train/Val datasets size: {}".format(len(n_dataset_1)))
        
        if t == "KLF" or t == "T1wCE":
            # Competition (Tumor Only) Train + Val + Test
            f_dataset_0 = Dataset(tumor_only_dir_path, [t], list_classes=["0"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

            logging.info("Train/Val datasets size: {}".format(len(f_dataset_0)))

            f_dataset_1 = Dataset(tumor_only_dir_path, [t], list_classes=["1"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

            logging.info("Train/Val datasets size: {}".format(len(f_dataset_1)))
            
            # Competition (No Tumor) Train + Val + Test
            h_dataset_0 = Dataset(no_tumor_dir_path, [t], list_classes=["0"], transform=transform, ext=ext, dims=dims, sel_slices=sel_slices)

            logging.info("Train/Val datasets size: {}".format(len(h_dataset_0)))
            
            # Competition (Tumor Only) + UPENN Train + Val + Test
            fn_dataset_0 = Dataset().concat_datasets(f_dataset_0, n_dataset_0)
            
            logging.info("Train/Val datasets size: {}".format(len(fn_dataset_0)))
            
            fn_dataset_1 = Dataset().concat_datasets(f_dataset_1, n_dataset_1)
            
            logging.info("Train/Val datasets size: {}".format(len(fn_dataset_1)))
        
        if t == "KLF" or t == "T1wCE":
            data_packs[t] = {
                "m_dataset_0": m_dataset_0,
                "m_dataset_1": m_dataset_1,
                "f_dataset_0": f_dataset_0,
                "f_dataset_1": f_dataset_1,
                "h_dataset_0": h_dataset_0,
                #"t_dataset_0": t_dataset_0,
                #"t_dataset_1": t_dataset_1,
                "n_dataset_0": n_dataset_0,
                "n_dataset_1": n_dataset_1,
                "fn_dataset_0": fn_dataset_0,
                "fn_dataset_1": fn_dataset_1
            }
        else:
            data_packs[t] = {
                "m_dataset_0": m_dataset_0,
                "m_dataset_1": m_dataset_1,
                #"t_dataset_0": t_dataset_0,
                #"t_dataset_1": t_dataset_1,
                "n_dataset_0": n_dataset_0,
                "n_dataset_1": n_dataset_1
            }
    return data_packs

import importlib
from utils import dataset_utils
importlib.reload(dataset_utils)

def get_merged_dataset(dataset_0, dataset_1, k=5):
    dataset_merged = Dataset().concat_datasets(dataset_0, dataset_1)
    dataset_merged_no_tr = Dataset().concat_datasets(dataset_0, dataset_1, import_transform=False)

    val_total_ratio = 0.2
    is_5_fold = True
    splits = dataset_utils.get_splits(dataset_0, dataset_1, val_total_ratio, is_5_fold, 0.1, k)
    print(splits)
    return dataset_merged, dataset_merged_no_tr, splits

def train_model(train_dl, validation_dl, fold_number, mri_type, model_name, epochs):
    device = torch.device("cuda")
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)

    model.zero_grad()
    model.to(device)
    best_loss = 9999
    best_auc = 0
    criterion = nn.BCEWithLogitsLoss()
    train_writer = SummaryWriter()
    for counter in range(epochs):

        epoch_iterator_train = tqdm(train_dl)
        tr_loss = 0.0
        for step, batch in enumerate(epoch_iterator_train):
            model.train()
            (img_ids, imgs, labels) = batch
            images, targets = imgs[0].to(device), labels.to(device)
            #images, targets = batch["image"].to(device), batch["target"].to(device)

            outputs = model(images)
            targets = targets  # .view(-1, 1)
            loss = criterion(outputs.squeeze(1), targets.float())

            loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

            tr_loss += loss.item()
            epoch_iterator_train.set_postfix(
                batch_loss=(loss.item()), loss=(tr_loss / (step + 1))
            )

        train_writer.add_scalar('loss', (tr_loss/(step+1)), counter+1)
        scheduler.step()  # Update learning rate schedule

        with torch.no_grad():
            val_loss = 0.0
            preds = []
            true_labels = []
            case_ids = []
            epoch_iterator_val = tqdm(validation_dl)
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                (img_ids, imgs, labels) = batch
                images, targets = imgs[0].to(device), labels.to(device)
                #images, targets = batch["image"].to(device), batch["target"].to(device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())
                val_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                )
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                true_labels.append(targets.cpu().numpy())
                #case_ids.append(batch["case_id"])
                if img_ids[0][0][0].isnumeric():
                    img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                else:
                    img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                case_ids.append(img_ids_fixed)
        preds = np.vstack(preds).T[0].tolist()
        true_labels = np.hstack(true_labels).tolist()
        case_ids = np.hstack(case_ids).tolist()
        auc_score = roc_auc_score(true_labels, preds)
        auc_score_adj_best = 0
        for thresh in np.linspace(0, 1, 50):
            auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
            if auc_score_adj > auc_score_adj_best:
                best_thresh = thresh
                auc_score_adj_best = auc_score_adj

        print(
            f"EPOCH {counter}/{epochs}: Validation average loss: {val_loss/(step+1)} + AUC SCORE = {auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best}"
        )
        train_writer.add_scalar('val_loss', val_loss/(step+1), counter+1)
        train_writer.add_scalar('val_auc', auc_score, counter+1)
        if auc_score > best_auc:
            print("Saving the model...")

            if not os.path.exists("../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/"):
                os.mkdir("../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/")
            all_files = os.listdir("../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/")

            for f in all_files:
                if f"{model_name}_{mri_type}_fold{fold_number}" in f:
                    os.remove(f"../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/{f}")

            best_auc = auc_score
            torch.save(
                model.state_dict(),
                f"../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/3d-{model_name}_{mri_type}_fold{fold_number}_{round(best_auc,3)}.pth",
            )

    print(best_auc)

def train_folded_models(fold, mri_type, model_name, batch_size, epochs, dataset_origin, idx_list):
    #data = pd.read_csv("train.csv")
    #train_df = data[data.fold != fold].reset_index(drop=False)
    #val_df = data[data.fold == fold].reset_index(drop=False)
    
    print(f"train_{mri_type}_{fold}")
    #train_dataset = BrainRSNADataset(data=train_df, mri_type=args.type, ds_type=f"train_{args.type}_{args.fold}")

    #valid_dataset = BrainRSNADataset(data=val_df, mri_type=args.type, ds_type=f"val_{args.type}_{args.fold}")

    packs = generate_datasets([mri_type])
    
    loader_packs = {}
    for t, pack in packs.items():
        print("Type: "+t)
        m_dataset_merged, m_dataset_merged_no_tr, m_splits = get_merged_dataset(pack['m_dataset_0'], pack['m_dataset_1'], fold)
        n_dataset_merged, n_dataset_merged_no_tr, n_splits = get_merged_dataset(pack['n_dataset_0'], pack['n_dataset_1'], fold)
        fn_dataset_merged, fn_dataset_merged_no_tr, fn_splits = get_merged_dataset(pack['fn_dataset_0'], pack['fn_dataset_1'], fold)
        print("SPLITS:")
        print("- folds:")
        print(len(m_splits))
        print("- splits per fold:")
        print(len(m_splits[0]))
        #print(len(n_splits))
        
        if dataset_origin == "m":
            i = 0
            for split in m_splits:
                m_dataloader = get_all_split_loaders(m_dataset_merged, m_dataset_merged_no_tr, [split], batch_size)
                m_dataloaders = list(m_dataloader[0])
                print(f"(M) Train validation splitted: {len(split[0])} {len(split[1])}")
                print(m_dataloaders)
                train_dl = m_dataloaders[0]
                validation_dl = m_dataloaders[1]
                train_model(train_dl, validation_dl, i, mri_type, model_name+"_m", epochs)
                """
                device = torch.device("cuda")
                model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                criterion = nn.BCEWithLogitsLoss()
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)
                trainer = Trainer(
                    model, 
                    device, 
                    optimizer, 
                    criterion,
                    scheduler,
                    1
                )

                history = trainer.fit(
                    device,
                    epochs, 
                    train_dl,
                    validation_dl,
                    model_name+"_m", 
                    15
                )

                trainer.train_writer.flush()
                """
                i += 1
        elif dataset_origin == "n":
            i = 0
            for split in n_splits:
                n_dataloader = get_all_split_loaders(n_dataset_merged, n_dataset_merged_no_tr, [split], batch_size)
                n_dataloaders = list(n_dataloader[0])
                print(f"(M) Train validation splitted: {len(split[0])} {len(split[1])}")
                train_dl = n_dataloaders[0]
                validation_dl = n_dataloaders[1]
                train_model(train_dl, validation_dl, i, mri_type, model_name+"_n", epochs)
                i += 1
        elif dataset_origin == "fn":
            i = 0
            for split in fn_splits:
            	if i in idx_list: 
		            fn_dataloader = get_all_split_loaders(fn_dataset_merged, fn_dataset_merged_no_tr, [split], batch_size)
		            fn_dataloaders = list(fn_dataloader[0])
		            print(f"(FN) Train validation splitted: {len(split[0])} {len(split[1])}")
		            train_dl = fn_dataloaders[0]
		            validation_dl = fn_dataloaders[1]
		            train_model(train_dl, validation_dl, i, mri_type, model_name+"_fn", epochs)
            	i += 1
    
fold = 5
mri_type = "T1wCE"
batch_size = 1
epochs = 15
dataset_origin = "fn"
idx_list = [0,2,4]
idx_list_odd = [1,3]
if args.even:
	train_folded_models(fold, mri_type, "resnet10", batch_size, epochs, dataset_origin, idx_list)
else:
	train_folded_models(fold, mri_type, "resnet10", batch_size, epochs, dataset_origin, idx_list_odd)
