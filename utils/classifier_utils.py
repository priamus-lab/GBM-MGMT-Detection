import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as torch_functional
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import itertools
import seaborn as sns
import logging
import time
from datetime import datetime
import os
import json
#from torchviz import make_dot
#import onnx
from copy import deepcopy
import collections
import gc

def plot_confusion_matrix(y_valid, y_pred,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_valid, y_pred)
    classes = ["0", "1"]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.clim(0,cm.max())
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() - cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion,
        scheduler,
        size
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.size = size

        self.best_valid_score = 0 #np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.train_writer = SummaryWriter()
        self.best_lbl_prd = []
        self.best_lbl_tru = []
        
    def fit(self, device, epochs, train_loader, valid_loader, save_path, patience):        
        for n_epoch in tqdm(range(1, epochs + 1)):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_auc, train_acc, train_time = self.train_epoch(device, train_loader, n_epoch, self.size)
            self.train_writer.add_scalar('loss', train_loss, n_epoch)
            self.train_writer.add_scalar('auc', train_auc, n_epoch)
            self.train_writer.add_scalar('accuracy', train_acc, n_epoch)
            
            valid_loss, valid_auc, valid_acc, valid_time = self.valid_epoch(device, valid_loader, n_epoch, self.size)
            self.train_writer.add_scalar('val_loss', valid_loss, n_epoch)
            self.train_writer.add_scalar('val_auc', valid_auc, n_epoch)
            self.train_writer.add_scalar('val_accuracy', valid_acc, n_epoch)
            
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f}, acc: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_auc, train_acc, train_time
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, acc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_acc, valid_time
            )

            # if True:
            if self.best_valid_score <= valid_acc: #valid_auc 
            #if self.best_valid_score > valid_loss: 
                self.save_model(n_epoch, save_path, valid_loss, valid_acc, valid_auc)
                self.info_message(
                     "acc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_valid_score, valid_acc, self.lastmodel
                )
                self.best_valid_score = valid_acc
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message("\nValid acc didn't improve last {} epochs.", patience)
                break
            
    def train_epoch(self, device, train_loader, n_epoch, size):
        self.model.train()
        t = time.time()
        sum_loss = 0
        sum_acc = 0
        total_size = 0
        
        y_all = [] #true
        outputs_all = [] #predicted
        prob_all = [] #probabilities
        ids = []
        #outputs_merged = []
        for step, elem in enumerate(train_loader):
            step += 1
            if size == 2:
                (img_ids, images, labels) = elem

                X_1, targets = images[0].to(device), labels.to(device)
                X_2, targets = images[1].to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(X_1,X_2)

                if self.model.output_size == 2:
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.criterion(outputs.squeeze(1), targets.float())

                sum_loss += loss.detach().item() * X_1.size(0)
                total_size += X_1.size(0)

                loss.backward()
                y_all.extend(labels.tolist())
                #outputs_all.extend(torch.sigmoid(outputs).tolist())
                if self.model.output_size == 2:
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = (outputs>0.5).int()
                    predicted = torch.reshape(predicted, (-1,))
                    
                outputs_all.extend(predicted.tolist())
                
                if self.model.output_size == 2:
                    prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                else:
                    prob_all.extend(outputs.sigmoid().detach().cpu().numpy())
                    
                sum_acc += torch.sum(predicted == targets.data).item()
                
                if img_ids[0][0][0].isnumeric():
                    img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                else:
                    img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                ids.extend(list(img_ids_fixed))

                self.optimizer.step()

                self.train_writer.add_scalar('train_step_loss', sum_loss/total_size, step+((n_epoch-1)*len(train_loader)))

                message = 'Train Step {}/{}, train_loss: {:.4f}'
                if step % 10 == 0:
                    self.info_message(message, step, len(train_loader), sum_loss/total_size, end="\r")
            else:
                (img_ids, images, labels) = elem

                X, targets = images[0].to(device), labels.to(device)

                self.optimizer.zero_grad()
                modelname = self.model.__class__.__name__
                if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
                    """
                    for i in range(X.shape[1]):
                        x_slice = torch.unsqueeze(X[:,i,:,:], dim=1)
                        outputs = self.model(x_slice)
                        
                        if self.model.output_size == 2:
                            loss = self.criterion(outputs, targets)
                        else:
                            loss = self.criterion(outputs.squeeze(1), targets.float())

                        sum_loss += loss.detach().item() * x_slice.size(0)
                        total_size += x_slice.size(0)

                        loss.backward()
                        y_all.extend(labels.tolist())
                        
                        if self.model.output_size == 2:
                            _, predicted = torch.max(outputs.data, 1)
                        else:
                            predicted = (outputs>0.5).int()
                            predicted = torch.reshape(predicted, (-1,))

                        outputs_all.extend(predicted.tolist())

                        if self.model.output_size == 2:
                            prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                        else:
                            prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                        sum_acc += torch.sum(predicted == targets.data).item()
                        
                        if img_ids[0][0][0].isnumeric():
                            img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                        else:
                            img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                        ids.extend(list(img_ids_fixed))

                        self.optimizer.step()
                    """
                    X = torch.swapaxes(X, 0, 1)
                    outputs = self.model(X)

                    if self.model.output_size == 2:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs.squeeze(1), targets.float())

                    sum_loss += loss.detach().item() * X.size(0)
                    total_size += X.size(0)

                    loss.backward()
                    y_all.extend(labels.tolist())
                    #outputs_all.extend(torch.sigmoid(outputs).tolist())
                    #outputs_all.extend(torch.soft(outputs).tolist())
                    if self.model.output_size == 2:
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        predicted = (outputs>0.5).int()
                        predicted = torch.reshape(predicted, (-1,))

                    outputs_all.extend(predicted.tolist())

                    if self.model.output_size == 2:
                        prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                    else:
                        prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                    sum_acc += torch.sum(predicted == targets.data).item()
                    
                    if img_ids[0][0][0].isnumeric():
                        img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                    else:
                        img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                    ids.extend(list(img_ids_fixed))

                    self.optimizer.step()
                        
                else:
                    outputs = self.model(X)

                    if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs.squeeze(1), targets.float())

                    sum_loss += loss.detach().item() * X.size(0)
                    total_size += X.size(0)

                    loss.backward()
                    y_all.extend(labels.tolist())
                    #outputs_all.extend(torch.sigmoid(outputs).tolist())
                    #outputs_all.extend(torch.soft(outputs).tolist())
                    if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        predicted = (outputs>0.5).int()
                        predicted = torch.reshape(predicted, (-1,))

                    outputs_all.extend(predicted.tolist())

                    if self.model.output_size == 2:
                        prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                    else:
                        prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                    sum_acc += torch.sum(predicted == targets.data).item()
                    
                    if img_ids[0][0][0].isnumeric():
                        img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                    else:
                        img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                    ids.extend(list(img_ids_fixed))

                    self.optimizer.step()

                self.train_writer.add_scalar('train_step_loss', sum_loss/total_size, step+((n_epoch-1)*len(train_loader)))

                message = 'Train Step {}/{}, train_loss: {:.4f}'
                if step % 10 == 0:
                    self.info_message(message, step, len(train_loader), sum_loss/total_size, end="\r")
            
        y_all = [1 if x > 0.5 else 0 for x in y_all]
        
        epoch_sum_loss = sum_loss/total_size
        
        #for (o1, o2) in outputs_all:
        #    outputs_merged.append((o1+o2)/2)
        
        #auc = roc_auc_score(y_all, outputs_merged)
        
        if self.model.output_size == 2:
            prob_all = np.vstack(prob_all).T[1].tolist() #getting probabilities of output=1
            
        modelname = self.model.__class__.__name__
        if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
            ids, outputs_all, prob_all, y_all = mean_voting_y(ids, outputs_all, prob_all, y=y_all)
        
        auc = roc_auc_score(y_all, prob_all)
        #outputs_merged_bin = [1 if x > 0.5 else 0 for x in outputs_merged]
        #acc = [1 if y == out else 0 for (y,out) in zip(y_all,outputs_merged_bin)].count(1)/len(outputs_merged_bin)
        ##acc = [1 if y == out else 0 for (y,out) in zip(y_all,outputs_all)].count(1)/len(outputs_all)
        acc = sum_acc/total_size
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        #return sum_loss/len(train_loader), auc, acc, int(time.time() - t)
        return epoch_sum_loss, auc, acc, int(time.time() - t)
    
    def valid_epoch(self, device, valid_loader, n_epoch, size):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        sum_acc = 0
        total_size = 0
        y_all = [] #true
        outputs_all = [] #predicted
        prob_all = [] #probabilities
        ids = []
        #outputs_merged = []

        for step, elem in enumerate(valid_loader):
        #for step, (img_ids, images, labels) in enumerate(valid_loader, 1):
            step += 1
            if size == 2:
                (img_ids, images, labels) = elem
                with torch.no_grad():
                    X_1, targets = images[0].to(device), labels.to(device)
                    X_2, targets = images[1].to(device), labels.to(device)
                    #X = batch["X"].to(self.device)
                    #targets = batch["y"].to(self.device)
                    #outputs = self.model(X_1,X_2).squeeze(1)
                    outputs = self.model(X_1,X_2)
                    #targets = torch.cat((torch.unsqueeze(targets_1,1), torch.unsqueeze(targets_2,1)),dim=-1)
                    if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                        loss = self.criterion(outputs, targets)
                    else:
                        loss = self.criterion(outputs.squeeze(1), targets.float())
                    #sum_loss += loss.detach().item()
                    sum_loss += loss.detach().item() * X_1.size(0)
                    total_size += X_1.size(0)
                    y_all.extend(labels.tolist())
                    #outputs_all.extend(torch.sigmoid(outputs).tolist())
                    if hasattr(self.model, 'output_size') and  self.model.output_size == 2:
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        predicted = (outputs>0.5).int()
                        predicted = torch.reshape(predicted, (-1,))
                    
                    outputs_all.extend(predicted.tolist())
                    
                    if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                        prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                    else:
                        prob_all.extend(outputs.sigmoid().detach().cpu().numpy())
                    
                    sum_acc += torch.sum(predicted == targets.data).item()
                    
                    if img_ids[0][0][0].isnumeric():
                        img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                    else:
                        img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                    ids.extend(list(img_ids_fixed))

                #self.train_writer.add_scalar('val_step_loss', sum_loss/step, step+((n_epoch-1)*len(valid_loader)))
                self.train_writer.add_scalar('val_step_loss', sum_loss/total_size, step+((n_epoch-1)*len(valid_loader)))

                message = 'Valid Step {}/{}, valid_loss: {:.4f}'
                if step % 5 == 0:
                    self.info_message(message, step, len(valid_loader), sum_loss/total_size, end="\r")
            else:
                (img_ids, images, labels) = elem
                with torch.no_grad():
                    X, targets = images[0].to(device), labels.to(device)
                    #X = batch["X"].to(self.device)
                    #targets = batch["y"].to(self.device)
                    #outputs = self.model(X_1,X_2).squeeze(1)
                    modelname = self.model.__class__.__name__
                    if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
                        X = torch.swapaxes(X, 0, 1)
                        outputs = self.model(X)
                        #targets = torch.cat((torch.unsqueeze(targets_1,1), torch.unsqueeze(targets_2,1)),dim=-1)
                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            loss = self.criterion(outputs, targets)
                        else:
                            loss = self.criterion(outputs.squeeze(1), targets.float())
                        #sum_loss += loss.detach().item()
                        sum_loss += loss.detach().item() * X.size(0)
                        total_size += X.size(0)
                        y_all.extend(labels.tolist())
                        #outputs_all.extend(torch.sigmoid(outputs).tolist())
                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            _, predicted = torch.max(outputs.data, 1)
                        else:
                            predicted = (outputs>0.5).int()
                            predicted = torch.reshape(predicted, (-1,))

                        outputs_all.extend(predicted.tolist())

                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                        else:
                            prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                        sum_acc += torch.sum(predicted == targets.data).item()

                        if img_ids[0][0][0].isnumeric():
                            img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                        else:
                            img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                        ids.extend(list(img_ids_fixed))
                        """
                        for i in range(X.shape[1]):
                            x_slice = torch.unsqueeze(X[:,i,:,:], dim=1)
                            
                            outputs = self.model(x_slice)
                            #targets = torch.cat((torch.unsqueeze(targets_1,1), torch.unsqueeze(targets_2,1)),dim=-1)
                            if self.model.output_size == 2:
                                loss = self.criterion(outputs, targets)
                            else:
                                loss = self.criterion(outputs.squeeze(1), targets.float())
                            #sum_loss += loss.detach().item()
                            sum_loss += loss.detach().item() * x_slice.size(0)
                            total_size += x_slice.size(0)
                            y_all.extend(labels.tolist())
                            #outputs_all.extend(torch.sigmoid(outputs).tolist())
                            if self.model.output_size == 2:
                                _, predicted = torch.max(outputs.data, 1)
                            else:
                                predicted = (outputs>0.5).int()
                                predicted = torch.reshape(predicted, (-1,))

                            outputs_all.extend(predicted.tolist())

                            if self.model.output_size == 2:
                                prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                            else:
                                prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                            sum_acc += torch.sum(predicted == targets.data).item()

                            if img_ids[0][0][0].isnumeric():
                                img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                            else:
                                img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                            ids.extend(list(img_ids_fixed))
                        """
                            
                    else:
                        outputs = self.model(X)
                        #targets = torch.cat((torch.unsqueeze(targets_1,1), torch.unsqueeze(targets_2,1)),dim=-1)
                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            loss = self.criterion(outputs, targets)
                        else:
                            loss = self.criterion(outputs.squeeze(1), targets.float())
                        #sum_loss += loss.detach().item()
                        sum_loss += loss.detach().item() * X.size(0)
                        total_size += X.size(0)
                        y_all.extend(labels.tolist())
                        #outputs_all.extend(torch.sigmoid(outputs).tolist())
                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            _, predicted = torch.max(outputs.data, 1)
                        else:
                            predicted = (outputs>0.5).int()
                            predicted = torch.reshape(predicted, (-1,))

                        outputs_all.extend(predicted.tolist())

                        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
                            prob_all.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                        else:
                            prob_all.extend(outputs.sigmoid().detach().cpu().numpy())

                        sum_acc += torch.sum(predicted == targets.data).item()

                        if img_ids[0][0][0].isnumeric():
                            img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                        else:
                            img_ids_fixed = [img_id[:-4] for img_id in img_ids[0]]
                        ids.extend(list(img_ids_fixed))

                #self.train_writer.add_scalar('val_step_loss', sum_loss/step, step+((n_epoch-1)*len(valid_loader)))
                self.train_writer.add_scalar('val_step_loss', sum_loss/total_size, step+((n_epoch-1)*len(valid_loader)))

                message = 'Valid Step {}/{}, valid_loss: {:.4f}'
                if step % 5 == 0:
                    self.info_message(message, step, len(valid_loader), sum_loss/total_size, end="\r")
            
        y_all = [1 if x > 0.5 else 0 for x in y_all]
        #auc = roc_auc_score(y_all, outputs_merged)
        
        if hasattr(self.model, 'output_size') and self.model.output_size == 2:
            prob_all = np.vstack(prob_all).T[1].tolist() #getting probabilities of output=1
            
        modelname = self.model.__class__.__name__
        if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
            ids, outputs_all, prob_all, y_all = mean_voting_y(ids, outputs_all, prob_all, y=y_all)
        
        preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_real_value": y_all, "MGMT_value": outputs_all}) 
        #preddf = preddf.set_index("BraTS21ID")
        preddf = preddf.sort_values(by="BraTS21ID")#.reset_index(drop=True)
        preddf.style.apply(self.highlight_equal, column1="MGMT_real_value", column2= "MGMT_value", axis=1)
        print("First 64 results:")
        print(preddf[:64])
        
        auc = roc_auc_score(y_all, prob_all)
        #acc = [1 if y == out else 0 for (y,out) in zip(y_all,outputs_merged_bin)].count(1)/len(outputs_merged_bin)
        ##acc = [1 if y == out else 0 for (y,out) in zip(y_all,outputs_all)].count(1)/len(outputs_all)
        epoch_sum_loss = sum_loss/total_size
        acc = sum_acc/total_size
        plot_confusion_matrix(y_all, outputs_all)
        
        #return sum_loss/len(valid_loader), auc, acc, int(time.time() - t)
        return epoch_sum_loss, auc, acc, int(time.time() - t)
    
    def save_model(self, n_epoch, save_path, loss, acc, auc):
        self.lastmodel = f"{save_path}-e{n_epoch}-loss{loss:.3f}-acc{acc:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )
        
    def highlight_equal(s, column1, column2):
        is_eq = pd.Series(data=False, index=s.index)
        is_eq[column1] = s.loc[column1] == s.loc[column2]
        return ['background-color: green' if is_eq.any() else '' for v in is_eq]
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        #print(message.format(*args), end=end)
        logging.info(message.format(*args))
        
#def train_mri_type(model, device, lr, epochs, pat, train_loader, valid_loader, is_multistep, size):
def train_mri_type(model, device, info, epochs, pat, train_loader, valid_loader, sub_idx=None):
    #train_loader = train_loaders[mri_type]
    #valid_loader = valid_loaders[mri_type]

    #model = RSNAClassifier()
    model.to(device)
    print(model)

    #checkpoint = torch.load("best-model-all-auc0.555.pth")
    #model.load_state_dict(checkpoint["model_state_dict"])

    patience = pat

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if info["is_adam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=info["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=info["lr"], momentum=info["momentum"])
    
    if info["is_multistep"]:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)
    else:
        scheduler = None
    
    date_time = datetime.now()
    date_str = date_time.strftime("%b%d_%H-%M-%S")
    
    modelname = model.__class__.__name__
    info["net"] = modelname
    if model.is_dw:
        modelname += "-DW"
        
    if model.output_size == 1:
        modelname += "-SO"
      
    remove_first = 0
    if info["mri_types"][0] == "KLF" and len(info["mri_types"]) > 1:
        modelname += "-KLF"
        remove_first = 1
    elif info["mri_types"][0] == "KLF":
        modelname += "-KLF"
        
    if info["train_origin"] is not None:
        modelname = modelname + "-" + info["train_origin"]
        
    if sub_idx is not None:
        modelname += f"-10F({sub_idx})"

    if model.output_size == 2:
        # Since there are two neuron outputs, cross entropy and not binary cross entropy must be used
        criterion = torch_functional.cross_entropy
    else:
        criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model, 
        device, 
        optimizer, 
        criterion,
        scheduler,
        len(info["mri_types"])-remove_first
    )
        
    foldername = f"../RSNA-BTC-Datasets/out_models/{modelname}_{date_str}"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    name = f"{foldername}/{modelname}"
    
    with open(f"{foldername}/training_info.txt", 'w') as file:
        #file.write(json.dumps(info))
        file.write("{\n")
        for k in info.keys():
            file.write(f"{k}: {info[k]}\n")
        file.write("}")

    history = trainer.fit(
        device,
        epochs, 
        train_loader,
        valid_loader,
        name, 
        patience
    )
    
    trainer.train_writer.flush()
    
    return trainer.lastmodel

def predict(model, device, modelfile, data_loader, size, is_target_included=True, labelfile=None, outputname=""):
    #data_loader = data_loaders[mri_type]
    logging.info("Predict: {} {}".format(modelfile, len(data_loader.dataset)))
    #df.loc[:,"MRI_Type"] = mri_type
    #data_retriever = Dataset(
    #    df.index.values, 
    #    mri_type=df["MRI_Type"].values,
    #    split=split
    #)

    #data_loader = torch_data.DataLoader(
    #    data_retriever,
    #    batch_size=4,
    #    shuffle=False,
    #    num_workers=8,
    #)
   
    #model = RSNAClassifier()
    model.to(device)
    
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    X_list = []
    y = []
    y_pred = []
    y_prob = []
    #y_pred_single = []
    ids = []

    if size == 2:
        for e, elem in enumerate(data_loader,1):
            (img_ids, images, labels) = elem
            print(f"{e}/{len(data_loader)}", end="\r")
            with torch.no_grad():
                #tmp_pred = torch.sigmoid(model(images_1.to(device),images_2.to(device))).cpu().numpy().squeeze()

                #tmp_pred_single = [(x[0]+x[1])/2 for x in tmp_pred]
                #y_pred_single.extend(tmp_pred_single)
                #tmp_pred_bin = [1 if x > 0.5 else 0 for x in tmp_pred_single]
                #y_pred.extend(tmp_pred_bin)
                if is_target_included:
                    X_1, targets = images[0].to(device), labels.to(device)
                    X_2, targets = images[1].to(device), labels.to(device)
                else:
                    X_1 = images[0].to(device)
                    X_2 = images[1].to(device)
                #X = batch["X"].to(self.device)
                #targets = batch["y"].to(self.device)

                #outputs = self.model(X_1,X_2).squeeze(1)
                outputs = model(X_1,X_2)
                """
                if e == 1:
                    input_names = ['MRI Scan']
                    output_names = ['Tumor']
                    if "BinaryEfficientNet3D" in modelfile:
                        model.net.set_swish(memory_efficient=False)
                    torch.onnx.export(model, (X_1,X_2), f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
                    #make_dot(outputs, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
                """
                if model.output_size is not None and model.output_size == 2:
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    predicted = (outputs>0.5).int()
                    predicted = torch.reshape(predicted, (-1,))
                    
                y_pred.extend(predicted.tolist())

                if model.output_size is not None and model.output_size == 2:
                    y_prob.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                else:
                    y_prob.extend(outputs.sigmoid().detach().cpu().numpy())

                #if tmp_pred_bin.size == 1:
                #   y_pred.append(tmp_pred_bin)
                #else:
                #    y_pred.extend(tmp_pred_bin.tolist())
                #print(img_ids_1)
                if img_ids[0][0][0].isnumeric():
                    img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                else:
                    img_ids_fixed = [img_id[:-4].replace("_FLAIR","").replace("_T1w","").replace("_T1wCE","").replace("_T2w","").replace("_KLF","") for img_id in img_ids[0]]
                ids.extend(list(img_ids_fixed))
                if is_target_included:
                    y.extend(labels.tolist())
                #print(y)
                #print(y_pred)
    else:
        for e, elem in enumerate(data_loader,1):
            (img_ids, images, labels) = elem
            print(f"{e}/{len(data_loader)}", end="\r")
            with torch.no_grad():
                #tmp_pred = torch.sigmoid(model(images_1.to(device),images_2.to(device))).cpu().numpy().squeeze()

                #tmp_pred_single = [(x[0]+x[1])/2 for x in tmp_pred]
                #y_pred_single.extend(tmp_pred_single)
                #tmp_pred_bin = [1 if x > 0.5 else 0 for x in tmp_pred_single]
                #y_pred.extend(tmp_pred_bin)
                if is_target_included:
                    X, targets = images[0].to(device), labels.to(device)
                else:
                    X = images[0].to(device)
                #X = batch["X"].to(self.device)
                #targets = batch["y"].to(self.device)
                modelname = model.__class__.__name__
                if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
                    X = torch.swapaxes(X, 0, 1)
                    outputs = model(X)
                    """
                    if e == 1:
                        input_names = ['MRI Scan']
                        output_names = ['Tumor']
                        if "BinaryEfficientNet3D" in modelfile:
                            model.net.set_swish(memory_efficient=False)
                        torch.onnx.export(model, X, f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
                        #make_dot(outputs, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
                    """
                    if model.output_size is not None and model.output_size == 2:
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        predicted = (outputs.sigmoid()>0.5).int() #sigmoid
                        predicted = torch.reshape(predicted, (-1,))

                    X_list.extend(X)
                    y_pred.extend(predicted.tolist())

                    if model.output_size is not None and model.output_size == 2:
                        y_prob.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                    else:
                        y_prob.extend(outputs.sigmoid().detach().cpu().numpy())

                    #if tmp_pred_bin.size == 1:
                    #   y_pred.append(tmp_pred_bin)
                    #else:
                    #    y_pred.extend(tmp_pred_bin.tolist())
                    #print(img_ids_1)
                    if img_ids[0][0][0].isnumeric():
                        img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                    else:
                        img_ids_fixed = [img_id[:-4].replace("_FLAIR","").replace("_T1w","").replace("_T1wCE","").replace("_T2w","").replace("_KLF","") for img_id in img_ids[0]]
                    ids.extend(list(img_ids_fixed))
                    if is_target_included:
                        y.extend(labels.tolist())
                    #print(y)
                    #print(y_pred)
                    """
                    for i in range(X.shape[1]):
                        x_slice = torch.unsqueeze(X[:,i,:,:], dim=1)
                        
                        outputs = model(x_slice)
                        
                        #if e == 1 and i == 1:
                        #    input_names = ['MRI Scan']
                        #    output_names = ['Tumor']
                        #    torch.onnx.export(model, x_slice, f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
                            #make_dot(outputs, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
                        
                        if model.output_size == 2:
                            _, predicted = torch.max(outputs.data, 1)
                        else:
                            predicted = (outputs>0.5).int()
                            predicted = torch.reshape(predicted, (-1,))

                        y_pred.extend(predicted.tolist())

                        if model.output_size == 2:
                            y_prob.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                        else:
                            y_prob.extend(outputs.sigmoid().detach().cpu().numpy())

                        #if tmp_pred_bin.size == 1:
                        #   y_pred.append(tmp_pred_bin)
                        #else:
                        #    y_pred.extend(tmp_pred_bin.tolist())
                        #print(img_ids_1)
                        if img_ids[0][0][0].isnumeric():
                            img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                        else:
                            img_ids_fixed = [img_id[:-4].replace("_FLAIR","").replace("_T1w","").replace("_T1wCE","").replace("_T2w","").replace("_KLF","") for img_id in img_ids[0]]
                        ids.extend(list(img_ids_fixed))
                        if is_target_included:
                            y.extend(labels.tolist())
                        #print(y)
                        #print(y_pred)
                    """
                else:
                    #outputs = self.model(X_1,X_2).squeeze(1)
                    outputs = model(X)
                    """
                    if e == 1:
                        input_names = ['MRI Scan']
                        output_names = ['Tumor']
                        if "BinaryEfficientNet3D" in modelfile:
                            model.net.set_swish(memory_efficient=False)
                        torch.onnx.export(model, X, f'{modelfile}.onnx', input_names=input_names, output_names=output_names)
                        #make_dot(outputs, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
                    """
                    if model.output_size is not None and model.output_size == 2:
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        predicted = (outputs>0.5).int()
                        predicted = torch.reshape(predicted, (-1,))

                    #X_list.extend(X)
                    y_pred.extend(predicted.tolist())

                    if model.output_size is not None and model.output_size == 2:
                        y_prob.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())
                    else:
                        y_prob.extend(outputs.sigmoid().detach().cpu().numpy())

                    #if tmp_pred_bin.size == 1:
                    #   y_pred.append(tmp_pred_bin)
                    #else:
                    #    y_pred.extend(tmp_pred_bin.tolist())
                    #print(img_ids_1)
                    if img_ids[0][0][0].isnumeric():
                        img_ids_fixed = [img_id[:5] for img_id in img_ids[0]]
                    else:
                        img_ids_fixed = [img_id[:-4].replace("_FLAIR","").replace("_T1w","").replace("_T1wCE","").replace("_T2w","").replace("_KLF","") for img_id in img_ids[0]]
                    ids.extend(list(img_ids_fixed))
                    if is_target_included:
                        y.extend(labels.tolist())
                        
                del outputs
                del X
                del predicted
                gc.collect()
                torch.cuda.empty_cache()
                    
                #print(y)
                #print(y_pred)
    if labelfile is not None:
        hl = pd.read_csv(labelfile, dtype={"BraTS21ID": object})
        hl["BraTS21ID"] = hl["BraTS21ID"].apply(lambda x: str(x))
        hl = hl.rename(columns={"MGMT_value": "MGMT_real_value"})
        y.extend(hl["MGMT_real_value"])
        
    modelname = model.__class__.__name__
    if modelname == "RSNAClassifierSingleVoting2D" or modelname == "RSNAAlternativeClassifierSingle2D":
        if labelfile is not None:
            ids, y_pred, y_prob = mean_voting_y(ids, y_pred, y_prob, hl)
        elif is_target_included:
            ids, y_pred, y_prob, y = mean_voting_y(ids, y_pred, y_prob, y=y)
        else:
            ids, y_pred, y_prob = mean_voting_y(ids, y_pred, y_prob)
        
    if is_target_included:
        print(f"Dataset: Labeled Test Set")
        preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_real_value": y, "MGMT_value": y_pred}) 
        #preddf = preddf.set_index("BraTS21ID")
        preddf = preddf.sort_values(by="BraTS21ID")#.reset_index(drop=True)
        preddf.style.apply(highlight_equal, column1="MGMT_real_value", column2= "MGMT_value", axis=1)
    elif labelfile is not None:
        print(f"Dataset: Hand Labeled Test Set")
        preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred}) 
        #preddf = preddf.set_index("BraTS21ID")
        preddf = preddf.sort_values(by="BraTS21ID")#.reset_index(drop=True)
        #hl = pd.read_csv("../hl_test_set.csv")
        
        ###preddf["MGMT_real_value"] = y
        #print(preddf)
        #print(hl)
        preddf = pd.merge(preddf, hl, on="BraTS21ID")
        # Fare join (fatto)
        # train su dataset esterno (fatto)
        # 10-fold (fatto)
        # modelli migliori train+val
        preddf.style.apply(highlight_equal, column1="MGMT_real_value", column2= "MGMT_value", axis=1)
    else:
        print(f"Dataset: Unlabeled Test Set")
        preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred}) 
        #preddf = preddf.set_index("BraTS21ID")
        preddf = preddf.sort_values(by="BraTS21ID")#.reset_index(drop=True)
    
    print(preddf)
      
    #auc = roc_auc_score(pred["MGMT_real_value"], pred["MGMT_value"])
    if is_target_included or labelfile is not None:
        print(f"Values:")
        sns.displot(y)
        #auc = roc_auc_score(y, y_pred_single)
  
        #if modelname == "RSNAClassifierSingleVoting2D":
            
            
        if model.output_size == 2:
            print("Zeros prob:")
            print(np.vstack(y_prob).T[0].tolist())
            print("Ones prob:")
            print(np.vstack(y_prob).T[1].tolist())
            y_prob = np.vstack(y_prob).T[1].tolist()
        else:
            print("Prob:")
            print(y_prob)
        
        auc = roc_auc_score(y, y_prob)
        print(f"Prediction AUC: {auc:.4f}")
        acc = [1 if yy == out else 0 for (yy,out) in zip(y,y_pred)].count(1)/len(y_pred)
        total_0_count = y.count(0)
        total_1_count = y.count(1)
        total_1_pred_count = y_pred.count(1)
        true_0 = [1 if yy == out and yy == 0 else 0 for (yy,out) in zip(y,y_pred)].count(1)
        true_1 = [1 if yy == out and yy == 1 else 0 for (yy,out) in zip(y,y_pred)].count(1)
        spec = true_0/total_0_count
        sens = true_1/total_1_count
        if total_1_pred_count != 0:
            prec = true_1/total_1_pred_count
        else:
            prec = 0
        print(f"Prediction Accuracy: {acc:.4f}")
        print(f"Prediction Specificity: {spec:.4f}")
        print(f"Prediction Sensitivity: {sens:.4f}")
        print(f"Prediction Precision: {prec:.4f}")
        #sns.displot(y_pred_single)
        print(f"Predictions:")
        sns.displot(y_pred)
        #auc_bin = roc_auc_score(y, y_pred)
        #print(f"Validation binary AUC: {auc_bin:.4f}")
        #sns.displot(y_pred)
        
        y_pred_0_count = [1 if y == 0 else 0 for y in y_pred].count(1)
        y_pred_1_count = [1 if y == 1 else 0 for y in y_pred].count(1)
        print(f"Predictions: {y_pred_0_count} without tumor, {y_pred_1_count} with tumor")
        
        comps = modelfile.split("-")
        val_loss = comps[-3][4:]
        val_acc = comps[-2][3:]
        val_auc = comps[-1][3:8]
        inference_info = {}
        inference_info["val_loss"] = val_loss
        inference_info["val_acc"] = val_acc
        inference_info["val_auc"] = val_auc
        inference_info["test_acc"] = acc
        inference_info["test_auc"] = auc
        inference_info["test_spec"] = spec
        inference_info["test_sens"] = sens
        inference_info["test_prec"] = prec
        inference_info["test_0_count"] = total_0_count
        inference_info["test_1_count"] = total_1_count

        foldername = os.path.dirname(modelfile)

        name = "_"+outputname
        with open(f"{foldername}/inference_info{name}.txt", 'w') as file:
            #file.write(json.dumps(info))
            file.write("{\n")
            for k in inference_info.keys():
                file.write(f"{k}: {inference_info[k]}\n")
            file.write("}")
    else:
        print(f"Predictions:")
        sns.displot(y_pred)
        y_pred_0_count = [1 if y == 0 else 0 for y in y_pred].count(1)
        y_pred_1_count = [1 if y == 1 else 0 for y in y_pred].count(1)
        print(f"Predictions: {y_pred_0_count} without tumor, {y_pred_1_count} with tumor")
        

    del checkpoint
    del model
    torch.cuda.empty_cache()
    return preddf, ids, X_list, y, y_pred, y_prob

def mean_voting_y(ids, y_pred, y_prob, hl=None, y=None):
    original_ids = deepcopy(ids)
    ids_pred_dict = collections.defaultdict(list)
    ids_prob_dict = collections.defaultdict(list)
    ids_y_dict = collections.defaultdict(list)
    if y is not None:
        for (single_id, single_y_pred, single_y_prob, single_y) in zip(ids, y_pred, y_prob, y):
            ids_pred_dict[single_id].append(single_y_pred)
            ids_prob_dict[single_id].append(single_y_prob)
            ids_y_dict[single_id].append(single_y)
    else:
        for (single_id, single_y_pred, single_y_prob) in zip(ids, y_pred, y_prob):
            ids_pred_dict[single_id].append(single_y_pred)
            ids_prob_dict[single_id].append(single_y_prob)

    ids = ids_pred_dict.keys()
    pred_vals = []
    prob_vals = []
    y_vals = []
    print("KEY |", "True VAL |", "Array of preds VAL |", "Pred Avg VAL |", "is correct")
    for key,val in ids_pred_dict.items():
        out_val = int((sum(val)/len(val)) > 0.5)
        if hl is not None:
            y_val = hl[hl["BraTS21ID"]==key]["MGMT_real_value"].values[0]
            print(key, y_val, val, out_val, y_val==out_val)
        #else:
        #    print(key, val, out_val)
        #weighted_sum = 0
        #k = 1
        #weighted_len = 0
        #for v in val:
        #    weighted_sum += v*k
        #    weighted_len += k
        #    k += 1
        pred_vals.append(out_val)
        #vals.append(int((weighted_sum/weighted_len) > 0.5))
    y_pred = pred_vals

    for key,val in ids_prob_dict.items():
        #weighted_sum = 0
        #k = 1
        #weighted_len = 0
        #for v in val:
        #    weighted_sum += v*k
        #    weighted_len += k
        #    k += 1
        prob_vals.append(sum(val)/len(val))
        #vals.append(weighted_sum/weighted_len)
    y_prob = prob_vals
    
    if y is not None:
        for key,val in ids_y_dict.items():
            #weighted_sum = 0
            #k = 1
            #weighted_len = 0
            #for v in val:
            #    weighted_sum += v*k
            #    weighted_len += k
            #    k += 1
            y_vals.append(val[0])
            #vals.append(weighted_sum/weighted_len)
        y = y_vals

        return ids, y_pred, y_prob, y
    else:
        return ids, y_pred, y_prob

def highlight_equal(s, column1, column2):
    is_eq = pd.Series(data=False, index=s.index)
    is_eq[column1] = s.loc[column1] == s.loc[column2]
    return ['background-color: green' if is_eq.any() else '' for v in is_eq]

def get_metrics(y, y_pred, y_prob, name):
    auc = roc_auc_score(y, y_prob)
    acc = [1 if yy == out else 0 for (yy,out) in zip(y,y_pred)].count(1)/len(y_pred)
    total_0_count = y.count(0)
    total_1_count = y.count(1)
    total_1_pred_count = list(y_pred).count(1)
    true_0 = [1 if yy == out and yy == 0 else 0 for (yy,out) in zip(y,y_pred)].count(1)
    true_1 = [1 if yy == out and yy == 1 else 0 for (yy,out) in zip(y,y_pred)].count(1)
    spec = true_0/total_0_count
    sens = true_1/total_1_count
    if total_1_pred_count != 0:
        prec = true_1/total_1_pred_count
    else:
        prec = 0
    print(f"Prediction AUC: {auc:.4f}")
    print(f"Prediction Accuracy: {acc:.4f}")
    print(f"Prediction Specificity: {spec:.4f}")
    print(f"Prediction Sensitivity: {sens:.4f}")
    print(f"Prediction Precision: {prec:.4f}")
    return pd.DataFrame({"model": [name], "AUC": [auc], "acc": [acc], "spec": [spec], "sens": [sens], "prec": [prec]})