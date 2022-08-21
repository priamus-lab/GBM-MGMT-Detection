import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--in_csv_file", default="train_labels.csv", type=str)
parser.add_argument("--out_csv_file", default="train_fold.csv", type=str)
parser.add_argument("--n_folds", default=5, type=int)
args = parser.parse_args()

train = pd.read_csv(f"../../RSNA-BTC-Datasets/{args.in_csv_file}")

skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=518)
oof = []
targets = []
target = "MGMT_value"

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train, train[target])
):
    train.loc[val_idx, "fold"] = int(fold)


train.to_csv(f"../../RSNA-BTC-Datasets/{args.out_csv_file}", index=False)
