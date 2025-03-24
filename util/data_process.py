# =============================================================================
# Utils --- Data Processing
# ============================================================================

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def pad_tensor_to_match(tensor, target_shape):
    """
    Pads a tensor to match the target shape.
    """
    padding = [(0, target - current) for current, target in zip(tensor.size(), target_shape)]
    padding = [p for pair in reversed(padding) for p in pair]  # Flatten the padding list
    return F.pad(tensor, padding)

def align_tensor_shapes(tensor_list, current_data_batch):
    """
    Aligns all tensors in the list to the current shape.
    """
    current_shape = [current_data_batch, tensor_list[0].size(1), tensor_list[0].size(2), tensor_list[0].size(3)]
    aligned_tensors = [pad_tensor_to_match(t, current_shape) for t in tensor_list]
    return aligned_tensors

def estimate_benign_updates(global_updates, alpha, current_data_batch):
    """
    Estimates benign global model updates by aligning mismatched tensor shapes and computing a weighted mean.
    """
    aligned_tensors = align_tensor_shapes(global_updates, current_data_batch)
    stacked_tensors = torch.stack(aligned_tensors, dim=0)
    mean_tensor = torch.mean(stacked_tensors, dim=0)
    return mean_tensor * alpha

def estimate_benign_updates_backup(global_updates, alpha, current_data_batch):
    """
    Estimates benign global model updates by aligning mismatched tensor shapes and computing a weighted mean.
    """
    aligned_tensors = align_tensor_shapes(global_updates, current_data_batch)
    stacked_tensors = torch.stack(aligned_tensors, dim=0)
    mean_tensor = torch.mean(stacked_tensors, dim=0)
    return mean_tensor * alpha