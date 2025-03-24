import torch
import torch.nn.functional as F

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


def euclidean_distance(tensor1, tensor2):
    """Calculate the Euclidean distance between two tensors."""
    return torch.norm(tensor1 - tensor2).item()


def krum(global_updates):
    """
    Apply the Krum algorithm to select the most central update.

    Args:
    - global_updates (list of torch.Tensor): List of model updates (e.g., gradients or weights).

    Returns:
    - torch.Tensor: The selected update based on the Krum algorithm.
    """
    num_updates = len(global_updates)
    distances = []
    for i in range(num_updates):
        sum_distances = 0
        for j in range(num_updates):
            if i != j:
                sum_distances += euclidean_distance(global_updates[i], global_updates[j])
        distances.append((i, sum_distances))
    distances.sort(key=lambda x: x[1])
    selected_update_index = distances[0][0]
    return global_updates[selected_update_index]


def poison_benign_updates_1(global_updates, current_data_batch):
    """
    Estimates poisoning attacks published by Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning
    """
    alpha = 1
    alpha_min = 0  # Set the lower bound for alpha
    alpha_max = 1  # Set the upper bound for alpha
    step_size = 0.5  # Initial step size
    aligned_tensors = align_tensor_shapes(global_updates, current_data_batch)
    stacked_tensors = torch.stack(aligned_tensors, dim=0)
    mean_tensor = torch.mean(stacked_tensors, dim=0)
    selected_tensor = krum(aligned_tensors)
    count1 = 0
    while alpha > alpha_min:
        if count1 < 2:
            alpha -= step_size
            step_size /= 2
            current_poisoned_tensor = mean_tensor + alpha * selected_tensor
            tmp = aligned_tensors.copy()
            tmp.append(current_poisoned_tensor)
            selected_tensor = krum(tmp)
            if torch.equal(selected_tensor, current_poisoned_tensor):
                break
            else:
                count1 += 1
        else:
            break
    min_alpha = alpha

    count2 = 0
    while alpha < alpha_max:
        if count2 < 2:
            alpha += step_size
            step_size /= 2
            current_poisoned_tensor = mean_tensor + alpha * selected_tensor
            tmp = aligned_tensors.copy()
            tmp.append(current_poisoned_tensor)
            selected_tensor = krum(tmp)
            if not torch.equal(selected_tensor, current_poisoned_tensor):
                break
            else:
                count2 += 1
        else:
            break
    max_alpha = alpha
    final_alpha = (min_alpha+max_alpha)/2
    current_poisoned_tensor = mean_tensor + final_alpha * selected_tensor
    return current_poisoned_tensor