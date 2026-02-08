# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
from params import BASE

def load5foldData(obj, shuffle_flag=True, seed=2020):
    random.seed(seed)
    cwd = os.getcwd()
    n_splits = 5
    target_column = 'student_id'

    # Define the path for the Excel dataset
    file_path = os.path.join(BASE, f"./{obj}")

    try:
        file_path = f"{BASE}/{obj}"
        print(file_path)
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        file_path = f"D:\\Acode\\labo\\aiprj\\{obj}"
        df = pd.read_excel(file_path)

    print("Loading dataset")

    # Shuffle the dataset if shuffle_flag is True
    if shuffle_flag:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Extract features and target
    X = df.drop(columns=[target_column])  # Feature part
    y = df[target_column]  # Target column

    grouped = df.groupby(target_column)
    groups = list(grouped.groups.keys())

    # Shuffle the groups if shuffle_flag is True
    if shuffle_flag:
        random.shuffle(groups)

    # Create indices for the splits
    fold_sizes = [len(groups) // n_splits] * n_splits
    for i in range(len(groups) % n_splits):
        fold_sizes[i] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(groups[start:stop])
        current = stop

    # Generate five-fold data
    fold_data = []
    train_d = []
    val_d = []
    for i in range(n_splits):
        val_groups = folds[i]
        train_groups = [group for fold in folds if fold != folds[i] for group in fold]
        train_data = df[df[target_column].isin(train_groups)]
        val_data = df[df[target_column].isin(val_groups)]
        fold_data.append((train_data, val_data))

    for i, (train, val) in enumerate(fold_data):
        train_d.append(train)
        val_d.append(val)
    return train_d, val_d

if __name__ == '__main__':
    fold_data = load5foldData("desensitized_data_Final.xlsx")
    for i, (train, val) in enumerate(fold_data):#某一行的数据：train.iloc[0]
        print(f"Fold {i+1}")
        print(f"Validation data shape: {val.shape}")
        print(fold_data)