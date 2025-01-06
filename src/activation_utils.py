import torch
import numpy as np
from utils import select_device
from torch.utils.data import DataLoader

global activation
activation = {}
def hook_fn(module, input, output):
    output.retain_grad()
    activation[module] = output

def get_activations(
        model,
        concept_dataloader,
        layer_idx,
        grad:bool = False
):  
    # hook to get access to intermediate activations
    """
    Obtain activations and optionally gradients from a specified layer of a model.

    This function registers a forward hook to capture intermediate activations
    from a specified layer of the model while processing a dataset. If `grad` is
    set to True, it also computes the gradients of the activations with respect 
    to the model's predictions for each target class.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to extract activations.
    concept_dataloader : torch.utils.data.DataLoader
        DataLoader providing the data for which to calculate activations.
    layer_idx : int
        The index of the layer to extract activations from.
    grad : bool, optional
        Whether to compute gradients of the activations. Defaults to False.

    Returns
    -------
    activations : np.ndarray
        The computed activations for each sample in the dataloader.
    activations_grad : np.ndarray, optional
        The gradients of the activations for each target class, returned only if `grad` is True.
    labels : np.ndarray
        The labels corresponding to the activations.
    """

    hook_handle = model.backbone.features[layer_idx].register_forward_hook(hook_fn)
    device = select_device()
    model.to(device)
    activations = []
    activations_grad = {i: [] for i in range(model.fc.out_features)}
    list_labels = []
    for images, labels in concept_dataloader:
        images = images.to(device)
        pred = model(images)
        list_labels.append(labels.detach().cpu().numpy())
        activation_i = list(activation.values())[0]
        # linear classifier input is samples, features = n, c * h * w
        activation_i = activation_i.view(activation_i.size(0), -1).detach().cpu().numpy()
        activations.append(activation_i)
        # conpute gradients for each class
        if grad:
            for target_class in range(model.fc.out_features):
                if target_class > 0:
                    pred = model(images)
                for sample_i in range(images.size(0)):
                    pred[sample_i, target_class].backward()
                    activations_gradi = list(activation.values())[0].grad
                    activations_gradi = activations_gradi.view(activations_gradi.size(0), -1).detach().cpu().numpy()
                    activations_grad[target_class].append(activations_gradi)
        activation.clear()
    activations = np.vstack(activations)
    activations_grad = np.vstack(activations_grad)
    labels = np.vstack(list_labels)
    hook_handle.remove()
    activation.clear()
    if grad:
        return activations, activations_grad, labels
    else:
        return activations, labels
    
def get_activation_gradients(
        model,
        concept_dataset,
        layer_idx,
):  
    """
    Compute the gradients of the activations of a specific layer.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to get the gradients from.
    concept_dataset : TerminatorDataset
        The dataset containing examples of the concept.
    layer_idx : int
        The index of the layer to get the gradients from.
    
    Returns
    -------
    activations_grad : dict of numpy arrays
        A dictionary where the keys are the target class indices and the values are the gradients of the activations.
    """
    hook_handle = model.backbone.features[layer_idx].register_forward_hook(hook_fn)
    concept_dataloader = DataLoader(concept_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = select_device()
    model.to(device)
    num_classes = model.fc.out_features
    activations_grad = {i: [] for i in range(num_classes)}
    for images, labels in concept_dataloader:
        images = images.to(device)
        for target_class in range(num_classes):
            pred = model(images)
            pred[:, target_class].backward()
            activations_gradi = list(activation.values())[0].grad
            activations_gradi = activations_gradi.view(activations_gradi.size(0), -1).detach().cpu().numpy()
            activations_grad[target_class].append(activations_gradi)
            activation.clear()
    for target_class in range(num_classes):
        activations_grad[target_class] = np.vstack(activations_grad[target_class])
    hook_handle.remove()
    activation.clear()
    return activations_grad
