"""
Shared sampling utility — dung chung cho ca 4 frameworks.
Dam bao tat ca framework lay cung subset du lieu.
"""

import numpy as np
from datasets import Dataset, DatasetDict


def stratified_sample(dataset_split, label_col, n, seed=42):
    """
    Lay n mau phan tang (stratified) tu mot split cua dataset.
    Dam bao ti le label duoc giu nguyen.

    Args:
        dataset_split: HuggingFace Dataset (1 split, vd: dataset['train'])
        label_col: ten cot label
        n: so mau can lay
        seed: random seed

    Returns:
        Dataset subset voi n mau, phan tang theo label
    """
    labels = np.array(dataset_split[label_col])
    unique_labels = np.unique(labels)

    rng = np.random.default_rng(seed)
    selected_indices = []

    n_per_label = n // len(unique_labels)
    remainder = n % len(unique_labels)

    for i, label in enumerate(unique_labels):
        label_indices = np.where(labels == label)[0]
        n_this_label = n_per_label + (1 if i < remainder else 0)
        n_this_label = min(n_this_label, len(label_indices))
        chosen = rng.choice(label_indices, size=n_this_label, replace=False)
        selected_indices.extend(chosen.tolist())

    rng.shuffle(selected_indices)
    return dataset_split.select(selected_indices)


def sample_dataset(dataset, label_col, n, seed=42):
    """
    Apply stratified sampling cho train/validation/test splits.

    Args:
        dataset: DatasetDict voi cac splits
        label_col: ten cot label
        n: so mau cho train split
        seed: random seed

    Returns:
        DatasetDict voi cac splits da duoc sample
    """
    train_n = n
    val_n = max(n // 5, 50)
    test_n = max(n // 3, 100)

    result = DatasetDict({
        "train": stratified_sample(dataset["train"], label_col, train_n, seed),
        "validation": stratified_sample(dataset["validation"], label_col,
                                        min(val_n, len(dataset["validation"])), seed),
        "test": stratified_sample(dataset["test"], label_col,
                                  min(test_n, len(dataset["test"])), seed),
    })

    return result
