import os
import torch
import logging
from val import val
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import TerminatorDataset
from torch.utils.data import DataLoader
from models.model_factory import model_factory
from transforms import train_transform, val_transform
from utils import select_device, get_num_classes, plot_training

def train(
        model_name:str='regnet400',
        epochs:int=10,
        images_dir:str='../Dataset/',
        train_labels:str='../Dataset/train.csv',
        val_labels:str='../Dataset/test.csv',
        add_info:str=None
):
    
    """
    Train a model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to be trained. Defaults to 'regnet400'.
    epochs : int, optional
        Number of epochs to train the model. Defaults to 10.
    images_dir : str, optional
        Directory where dataset images are stored. Defaults to '../Dataset/'.
    train_labels : str, optional
        CSV file containing the training labels. Defaults to '../Dataset/train.csv'.
    val_labels : str, optional
        CSV file containing the validation labels. Defaults to '../Dataset/test.csv'.
    add_info : str, optional
        Additional information to be added to the weights directory. Defaults to None.

    Returns
    -------
    str
        Directory where the trained model weights are saved.
    """
    weights_dir = f'../outputs/weights/{model_name}_epochs{epochs}'
    if add_info is not None:
        weights_dir += f'_{add_info}'
    os.makedirs(weights_dir, exist_ok=True)

    # Check if GPU is available and assign device accordingly
    device = select_device()

    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    # Initialize dataset and data loader
    train_dataset = TerminatorDataset(images_dir, train_labels, transform=train_transform)
    val_dataset = TerminatorDataset(images_dir, val_labels, transform=val_transform)
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=32, shuffle=False, drop_last=False)
    # Get number of classes for each label
    num_classes = get_num_classes(train_labels)
    logging.info(f'num_classes: {num_classes}')
    # Initialize model, optmizer and criterion
    model = model_factory(model_name, num_classes)
    model.to(device)
    logging.info(f'Model: {model_name} loaded')
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_loss = []
    val_loss = []
    trainacc = []
    valacc = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_samples = 0
        epoch_accuracy = 0
        model.train()
        logging.info(f'Training Epoch {epoch+1}/{epochs}')
        for images, label in train_loader:
            
            images = images.to(device)
            label = label.to(device)
            # Forward pass
            output = model(images)
            # Compute losses
            loss = criterion(output, label)
            epoch_loss += loss.item() * images.size(0)
            epoch_samples += images.size(0)
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_accuracy+= (predicted == label).sum().item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= epoch_samples
        epoch_accuracy /= epoch_samples
        train_loss.append(epoch_loss)
        trainacc.append(epoch_accuracy)
        # import pdb; pdb.set_trace()
        message = f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}'
        logging.info(message)

        # Save the model checkpoint
        save_model_in = f'{weights_dir}/{epoch+1}.pth'
        torch.save(model.state_dict(), save_model_in)
        logging.info(f'Model saved in {save_model_in}')

        # Validation
        message = f'Validation Epoch {epoch+1}/{epochs}'
        logging.info(message)
        epoch_loss, epoch_accuracy = val(model, val_loader, criterion)
        val_loss.append(epoch_loss)        
        valacc.append(epoch_accuracy)
        message = f'Epoch {epoch+1}/{epochs}, Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}'
        logging.info(message)
    results_dict = {
        'train_loss': train_loss, 
        'val_loss': val_loss, 
        'trainacc': trainacc,
        'valacc': valacc, 
    }
    pd.DataFrame(results_dict).to_csv(f'{weights_dir}/results.csv', index=False)

    # Plot training and validation loss
    plot_training(
        train_loss,
        val_loss,
        trainacc,
        valacc,
        epochs,
        weights_dir
    )
    return weights_dir


if __name__ == '__main__':
    images_dir = '../Dataset/'
    val_labels = '../Dataset/test.csv'
    model_name = 'efficientnetb0'
    epochs = 3

    # train_labels = '../Dataset/train.csv'
    # add_info = None

    train_labels = '../Dataset/train_corrected.csv'
    add_info = 'corrected'

    train(model_name, epochs, images_dir, train_labels, val_labels, add_info)