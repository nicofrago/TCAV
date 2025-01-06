#!/usr/bin/env python
# coding: utf-8
import os
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
import os.path as osp   
import matplotlib.pyplot as plt
from transforms import val_transform
from dataset import TerminatorDataset
from torch.utils.data import DataLoader
from models.model_factory import load_model
from utils import select_device, read_labels

from visualization import visualize_cav

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression

import umap


from activation_utils import get_activations, get_activation_gradients

def train_linear_classifier_from_concept(
        model, 
        concept_dataloader, 
        non_concept_dataloader,
        layer_idx:int = 8,
        visualize:bool = False,
        save_in:str = '../outputs/weights/logistic_regresion',
        classifier:str = 'logistic'
):
    """
    Train a linear classifier to distinguish between the concept and non-concept examples.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to get the activations from.
    concept_dataloader : torch.utils.data.DataLoader
        The dataloader for the concept examples.
    non_concept_dataloader : torch.utils.data.DataLoader
        The dataloader for the non-concept examples.
    layer_idx : int, optional
        The index of the layer to get the activations from. Defaults to 8.
    visualize : bool, optional
        Whether to visualize the learned weights. Defaults to False.
    save_in : str, optional
        The directory to save the weights in. Defaults to '../outputs/weights/logistic_regresion'.
    classifier : str, optional
        The type of classifier to train. Defaults to 'logistic'.

    Returns
    -------
    clf : sklearn.linear_model._base.LinearClassifierMixin
        The trained classifier.
    """
    
    os.makedirs(save_in, exist_ok=True)
    # train linear classifier
    arnies_activations, labels = get_activations(model, concept_dataloader, layer_idx)
    enemies_activations, labels = get_activations(model, non_concept_dataloader, layer_idx)
    X = np.vstack((arnies_activations, enemies_activations))
    y = np.concatenate((np.ones(arnies_activations.shape[0]), np.zeros(enemies_activations.shape[0])))
    if classifier == 'svm':
        clf = SVC(kernel='linear').fit(X, y)
    elif classifier == 'logistic':
        clf = LogisticRegression().fit(X, y)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    training_accuracy = float(clf.score(X, y))  # Explicitly convert to float
    print(f'Training accuracy: {training_accuracy:.4f}')
    # save the classifier
    # weights = {"coef": clf.coef_, "intercept": clf.intercept_, 'classes': clf.classes_}
    # np.save(f"{save_in}/weights.npy", weights)
    # Create a new model and set the weights
    # new_clf.coef_ = loaded_weights["coef"]
    # new_clf.intercept_ = loaded_weights["intercept"]
    if visualize:
        visualize_cav(clf, X, len(arnies_activations))
    return clf

def get_tcav_score(
        model,
        classifier,
        layer_idx:int,
        concept_dataset:TerminatorDataset
):
    """
    Calculate the TCAV (Testing with Concept Activation Vectors) scores for each target class.

    This function computes the TCAV scores that quantify the influence of a specific concept
    on the predictions of a model for each target class. It does so by calculating the 
    directional derivative of the activations with respect to a concept classifier's 
    coefficients.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which the TCAV scores are calculated.
    classifier : sklearn.linear_model._base.LinearClassifierMixin
        The linear classifier trained to distinguish between concept and non-concept examples.
    layer_idx : int
        The index of the layer from which to obtain activations for TCAV calculation.
    concept_dataset : TerminatorDataset
        The dataset containing examples of the concept.

    Returns
    -------
    tcva_scores : list of float
        A list containing the TCAV scores for each target class, indicating the influence
        of the concept on the model's predictions for each class.
    """

    device = select_device()
    model.to(device)
    num_classes = model.fc.out_features
    signs = [0] * num_classes
    activations_grad = get_activation_gradients(model, concept_dataset, layer_idx)
    for target_class in range(num_classes):
        gradi = activations_grad[target_class]
        for sample_i in range(len(gradi)):
            directional_derivative = np.dot(gradi[sample_i], classifier.coef_[0])
            if directional_derivative > 0:
                signs[target_class] += 1
    tcva_scores= []
    for target_class in range(num_classes):
        tcav_score = signs[target_class] / len(concept_dataset)
        tcva_scores.append(tcav_score)
        print(f"TCAV score for target class {target_class}: {tcav_score} ")
    return tcva_scores
def correct_labels_from_concept(
        model,
        layer_idx:int,
        clf,
        dataset:TerminatorDataset,
        label_path:str = '../Dataset/train.csv'
):
    """
    Correct labels in a dataset by using a concept classifier to distinguish between Arnies and non-Arnies examples.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the activations from.
    layer_idx : int
        The index of the layer from which to obtain activations.
    clf : sklearn.linear_model._base.LinearClassifierMixin
        The linear classifier trained to distinguish between concept and non-concept examples.
    dataset : TerminatorDataset
        The dataset containing examples of the concept.
    label_path : str, optional
        The path to the csv file containing the original labels. Defaults to '../Dataset/train.csv'.
    
    Returns
    -------
    save_in : str
        The path to the new csv file containing the corrected labels.
    arnies_indexes : list of int
        A list of the indices of the examples that were labeled as Arnies.
    """
    set_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    arnies_indexes = []
    device = select_device()
    model.to(device)
    activations, labels = get_activations(model, set_loader, layer_idx)
    for idx, act in enumerate(activations):
        prediction = int(clf.predict(act.reshape(1, -1))[0])
        if prediction == 1:
            arnies_indexes.append(idx)
    # correct labels and create a new csv file
    labels = read_labels(label_path)
    corrected_labels = len(labels.iloc[arnies_indexes][labels['class']==0])
    labels.loc[arnies_indexes, 'class'] = 1
    save_in = label_path.replace('.csv', '_corrected.csv')
    labels.to_csv(save_in, index=False, header=None)
    print('labels saved in:', save_in)
    print(f'{corrected_labels} labels corrected')
    return save_in, arnies_indexes