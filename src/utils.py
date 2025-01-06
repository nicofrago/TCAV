import os
import torch
import yaml
import glob
import pandas as pd
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt

def list_images(
        images_folder_path: str, 
        extension_list: list = ['.jpg', '.jpeg', '.png']
):
    """
    Lists all images in a given folder with specific extensions.

    Args:
        images_folder_path (str): Path to folder containing images.
        extension_list (list, optional): List of extensions to include. Defaults to ['.jpg', '.jpeg', '.png'].
    
    Returns:
        list: List of paths to all images in the given folder with the specified extensions.
    """
    assert osp.exists(images_folder_path), f"Path {images_folder_path} does not exist"
    images = []
    for ext in extension_list:
        images_search = f"{images_folder_path}/*{ext}"
        list_files = glob.glob(images_search)
        images.extend(list_files)
    return images

def select_device():
    """
    Selects and returns the appropriate device for computation.

    This function checks if CUDA is available and selects a GPU device if possible.
    Otherwise, it defaults to using a CPU device for computation. It also prints 
    the selected device.

    Returns:
        torch.device: The selected device ('cuda' or 'cpu').
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_num_classes(label_path:str):
    """
    Get the number of classes from a labels csv file.

    Args:
        label_path (str): Path to csv file with labels. The csv file should have two columns, the first column 
            should be the class label and the second column should be the filename of the image.

    Returns:
        int: The number of unique classes found in the labels file.
    """
    df = pd.read_csv(
        label_path, 
        header=None,
        names=['class', 'filename']
    )
    num_classes = df['class'].nunique()
    return num_classes

def read_yaml(file_path):
    """
    Reads a yaml file and returns its contents as a dictionary.

    Parameters:
        file_path (str): The path to the yaml file

    Returns:
        dict: A dictionary containing the contents of the yaml file

    Raises:
        AssertionError: If the file does not exist
    """
    assert osp.exists(file_path), f"File not found: {file_path}"
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_labels_csv(dataset_dir:str, setname:str):
    """
    Create a labels csv file from a directory structure

    Parameters:
        dataset_dir (str): The root directory of the dataset
        setname (str): The name of the set (e.g. 'train', 'val', 'test')

    Returns:
        pd.DataFrame: A DataFrame containing the labels

    The function will create a csv file in the same directory as the set directory with the same name as the set
    but with a .csv extension. The csv file will contain two columns, 'class' and 'filename', where 'class' is the
    name of the class and 'filename' is the name of the image relative to the dataset directory

    The function will return the DataFrame containing the labels
    """
    dataset_dir = osp.normpath(dataset_dir)
    set_dir = f'{dataset_dir}/{setname}'
    labels = []
    for i in os.listdir(set_dir):
        imdir = f'{set_dir}/{i}'
        images = list_images(imdir)
        labelsi = [[i, im.replace(dataset_dir + '/', '')] for im in images]
        labels += labelsi
    labels = pd.DataFrame(labels, columns=['class', 'filename'])
    save_in = f'{osp.dirname(set_dir)}/{osp.basename(set_dir)}.csv'
    labels.to_csv(save_in, header=None, index=False)
    print('labels saved in:', save_in)
    return labels

def create_labels(dataset_dir:str='../dataset'):
    
    """
    Create labels csv files for the dataset

    Parameters:
        dataset_dir (str): The root directory of the dataset

    Returns:
        tuple: A tuple containing two DataFrames, the first one for the training set and the second one for the test set

    The function will create two csv files, one for the training set and one for the test set, in the same directory as the dataset directory
    with the same name as the set but with a .csv extension. The csv file will contain two columns, 'class' and 'filename', where 'class' is the
    name of the class and 'filename' is the name of the image relative to the dataset directory

    The function will return a tuple of two DataFrames, one for each set
    """
    labels_train = create_labels_csv(dataset_dir, 'train')
    labels_test = create_labels_csv(dataset_dir, 'test')
    return labels_train, labels_test

def read_labels(labels_path:str):
    """
    Reads a CSV file containing image labels and returns a DataFrame.

    Args:
        labels_path (str): Path to the CSV file with labels. The CSV file should have two columns:
                           the first column should be the class label and the second column should 
                           be the filename of the image.

    Returns:
        pd.DataFrame: A DataFrame containing the labels with columns 'class' and 'filename'.
    """
    df = pd.read_csv(
            labels_path, 
            header=None,
            names=['class', 'filename']
        )
    return df

def plot_training(
        train_loss,
        val_loss,
        trainacc,
        valacc,
        epochs,
        weights_dir
):

    # Plot training and validation loss
    """
    Plots training and validation loss and accuracy during training

    Args:
        train_loss (list): List of training losses
        val_loss (list): List of validation losses
        trainacc (list): List of training accuracy
        valacc (list): List of validation accuracy
        epochs (int): Number of epochs
        weights_dir (str): Directory to save the plots
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{weights_dir}/loss_plot.png')
    plt.show()
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), trainacc, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{weights_dir}/train_accuracy_plot.png')
    plt.show()
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), valacc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{weights_dir}/val_accuracy_plot.png')
    plt.show()

def get_image(index, dataset_dir, labels):
    """
    Reads an image from the dataset.

    Args:
        index (int): Index of the image in the dataset.
        dataset_dir (str): Directory of the dataset.
        labels (pd.DataFrame): DataFrame containing the labels of the dataset.

    Returns:
        PIL.Image.Image: The image at the given index.
    """
    image_path = f'{dataset_dir}/{labels.iloc[index].filename}'
    assert osp.exists(image_path), f'Image not found: {image_path}'
    image = Image.open(image_path)
    return image

def get_conceptdf(
        labels:pd.DataFrame, 
        listnames:list, 
        dataset_name:str = 'train/0\\', 
        images_dir:str = '../Dataset/' ,
        debug:bool = False
):
    """
    Selects a subset of the dataset based on the given list of names.

    Args:
        labels (pd.DataFrame): DataFrame containing the labels of the dataset.
        listnames (list): List of names to select from the dataset.
        dataset_name (str, optional): Directory name of the dataset. Defaults to 'train/0\\'.
        images_dir (str, optional): Root directory of the dataset. Defaults to '../Dataset/'.
        debug (bool, optional): Whether to display the selected images. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the selected subset of the dataset.
    """

    
    listnames = [f'{dataset_name}{f}.jpg' for f in listnames]
    index = [] 
    fig, axs = plt.subplots(1, len(listnames), figsize=(len(listnames) * 2, 2))
    for f in listnames:
        idx = labels[labels['filename'] == f].index[0]
        index.append(idx)
        if debug:
            for i, idx in enumerate(index):
                image = get_image(idx, images_dir, labels)
                axs[i].imshow(image)
                axs[i].axis('off')
    if debug:
        plt.show()
    return labels.iloc[index]

if __name__ == '__main__':
    dataset_dir = '../dataset'
    labels_train, labels_test = create_labels()