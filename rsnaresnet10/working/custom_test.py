import os
import argparse

import monai
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from dataset import BrainRSNADataset

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", default="train.csv", type=str)
parser.add_argument("--type", default="FLAIR", type=str)
parser.add_argument("--model_name", default="resnet10_m", type=str)
args = parser.parse_args()

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

data = pd.read_csv(f"../../RSNA-BTC-Datasets/{args.csv_file}")

targets = data.MGMT_value.values

device = torch.device("cuda")
model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
model.to(device)

tta_true_labels = []
tta_preds = []
preds_f = np.zeros(len(data))

test_df = data[data.split == 2].reset_index(drop=False)

for type_ in [args.type]:
    preds_type = np.zeros(len(data))
    all_weights = os.listdir("../../RSNA-BTC-Datasets/rsnaresnet10_output_weights")
    fold_files = [f for f in all_weights if args.model_name+"_"+type_+"_" in f]
    #val_df = data[data.fold == fold]
    #val_index = val_df.index
    #val_df = val_df.reset_index(drop=True)
    test_index = test_df.index

    test_dataset = BrainRSNADataset(data=test_df, mri_type=type_, is_train=True, do_load=False, ds_type=f"test_{type_}_holdout")
    test_dl = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=4
        )
    image_ids = []
    model.load_state_dict(torch.load(f"../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/{fold_files[0]}"))
    preds = []
    y_pred = []
    true_labels = []
    case_ids = []
    with torch.no_grad():
        for  step, batch in enumerate(test_dl):
            model.eval()
            images, targets = batch["image"].to(device), batch["target"].to(device)

            outputs = model(images)
            targets = targets
            preds.append(outputs.sigmoid().detach().cpu().numpy())
            y_pred.append((outputs.sigmoid().detach().cpu().numpy()>0.5).astype(int))
            true_labels.append(targets.cpu().numpy())
            case_ids.append(batch["case_id"])

    case_ids = np.hstack(case_ids).tolist()

    #preds_f[val_index] += np.vstack(preds).T[0]/5
    #preds_type[val_index] += np.vstack(preds).T[0]
    #score_fold = roc_auc_score(targets[test_index], np.vstack(preds).T[0])
    #print(f"the score of the fold number {fold} and the type {type_}: {score_fold}")
        

    print(f"the final socre of the type {type_}")
    #print(roc_auc_score(targets, preds_type))
    print(roc_auc_score(true_labels, np.vstack(preds).T[0]))
    print("\n"*2)
    preds = np.vstack(preds).T[0].tolist()
    true_labels = np.hstack(true_labels).tolist()
    y_pred = np.hstack(y_pred).tolist()[0]
    preddf = get_metrics(true_labels, y_pred, preds, args.model_name)
    print(preddf)
    preddf.to_csv(f"tunisiaai_metrics_{args.model_name}.csv", index=False)
#print("the final score is")
#print(roc_auc_score(targets, preds_f))
