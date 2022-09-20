import os
import argparse
import monai
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from dataset import BrainRSNADataset

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

parser = argparse.ArgumentParser()
parser.add_argument("--models_folder", default=".", type=str)
parser.add_argument("--csv_file", default="train_fold.csv", type=str)
parser.add_argument("--full_set", default=False, type=bool)
args = parser.parse_args()

comb = "unknown"
if args.models_folder == "tunisiaai_A":
    if args.csv_file == "train_fold.csv":
        comb = "A"
    else:
        comb = "A_B"
elif args.models_folder == "tunisiaai_B":
    if args.csv_file == "train_fold.csv":
        comb = "B_A"
    else:
        comb = "B"
elif args.models_folder == "tunisiaai_AB":
    comb = "AB"
    
if args.csv_file != "all":
    data = pd.read_csv(f"../../RSNA-BTC-Datasets/{args.csv_file}")
    if args.csv_file == "upenn_train_fold_t1wce.csv":
        data["BraTS21ID"] = data["BraTS21ID"].apply(lambda x :f"UPENN-GBM-{str(x).zfill(5)}")
else:
    data1 = pd.read_csv(f"../../RSNA-BTC-Datasets/train_fold.csv")
    data2 = pd.read_csv(f"../../RSNA-BTC-Datasets/upenn_train_fold_t1wce.csv")
    data2["BraTS21ID"] = data2["BraTS21ID"].apply(lambda x :f"UPENN-GBM-{str(x).zfill(5)}")
    data = data1.append(data2, ignore_index=True)

all_ids = data.BraTS21ID.values
targets = data.MGMT_value.values

device = torch.device("cuda")
model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
model.to(device)

tta_true_labels = []
tta_preds = []
preds_f = np.zeros(len(data))
pd.set_option("display.max_rows", None, "display.max_columns", None)

for type_ in ["T1wCE"]:
    preds_type = np.zeros(len(data))
    y_pred = np.zeros(len(data))
    all_weights = os.listdir(f"../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/{args.models_folder}")
    fold_files = [f for f in all_weights if "3d-resnet10_"+type_+"_" in f]
    acc_list = []
    spec_list = []
    sens_list = []
    prec_list = []
    for fold in range(5):
        if args.full_set:
            val_df = data
        else:
            val_df = data[data.fold == fold]
        val_index = val_df.index
        val_df = val_df.reset_index(drop=True)

        if args.csv_file == "all":
            test_dataset = BrainRSNADataset(data=val_df, mri_type=type_, is_train=True, do_load=False, ds_type=f"val_{type_}_{fold}", folder="all")
        elif args.csv_file == "train_fold.csv":
            test_dataset = BrainRSNADataset(data=val_df, mri_type=type_, is_train=True, do_load=False, ds_type=f"val_{type_}_{fold}")
        else:
            test_dataset = BrainRSNADataset(data=val_df, mri_type=type_, is_train=True, do_load=False, ds_type=f"val_{type_}_{fold}", folder="UPENN-GBM")
        test_dl = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
        image_ids = []
        model.load_state_dict(torch.load(f"../../RSNA-BTC-Datasets/rsnaresnet10_output_weights/{args.models_folder}/{fold_files[fold]}"))
        preds = []
        y_pred_fold = []
        case_ids = []
        with torch.no_grad():
            for  step, batch in enumerate(test_dl):
                model.eval()
                images = batch["image"].to(device)

                outputs = model(images)
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                y_pred_fold.append((outputs.sigmoid()>0.5).detach().cpu().numpy())
                case_ids.append(batch["case_id"])

        case_ids = np.hstack(case_ids).tolist()

        preds_f[val_index] += np.vstack(preds).T[0]/5
        preds_type[val_index] += np.vstack(preds).T[0]
        y_pred[val_index] += np.vstack(y_pred_fold).T[0]
        y_pred_fold_int = 1*np.vstack(y_pred_fold).T[0]
        df = pd.DataFrame({"BraTS21ID": np.hstack(case_ids).tolist(), "MGMT_value": targets[val_index].tolist(), "pred": y_pred_fold_int, "prob": np.vstack(preds).T[0]})
        print(f"Fold {fold}:")
        print(df)
        score_fold = roc_auc_score(targets[val_index], np.vstack(preds).T[0])
        metrics = get_metrics(targets[val_index].tolist(), np.vstack(y_pred_fold).T[0], np.vstack(preds).T[0], f"{fold}")
        print(f"the score of the fold number {fold} and the type {type_}: {score_fold}")
        print(metrics)
        acc_list.append(metrics["acc"][0])
        spec_list.append(metrics["spec"][0])
        sens_list.append(metrics["sens"][0])
        prec_list.append(metrics["prec"][0])

    #print(targets.tolist())
    #print(y_pred)
    #print(preds_type)
    preddfA = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_real_value": targets.tolist(), "MGMT_pred_value": y_pred, "MGMT_prob_value": preds_type}) 
    #preddfA = preddfA.sort_values(by="BraTS21ID")
    #print(preddfA)
    preddfA.to_csv(f"../pred_metrics/{args.models_folder}_{comb}.csv", index=False)
    
    metrics_ = get_metrics(targets.tolist(), y_pred, preds_type, f"all")
    if args.full_set:
        metrics = pd.DataFrame({"model": ["all"], "AUC": [metrics_["AUC"][0]], "acc": [sum(acc_list)/len(acc_list)], "spec": [sum(spec_list)/len(spec_list)], "sens": [sum(sens_list)/len(sens_list)], "prec": [sum(prec_list)/len(prec_list)]})
    else:
        metrics = get_metrics(targets.tolist(), y_pred, preds_type, f"all")
    print(f"the final socre of the type {type_}")
    print(roc_auc_score(targets, preds_type))
    print(metrics)
    print("\n"*2)
print("the final score is")
print(roc_auc_score(targets, preds_f))
