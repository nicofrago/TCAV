from activation_utils import get_activations
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from torch.utils.data import DataLoader
def visualize_hidden_activation(
        model,
        reduction_method_name:str,
        layers_idx:list, 
        set_dataloader:DataLoader,
):  
    """
    Visualize the activations of a specific layer in a model using PCA.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to extract activations.
    reduction_method_name : str
        The name of the dimensionality reduction method. Currently supports 'pca' and 'tsne'.
    layers_idx : list of int
        The indices of the layers to visualize.
    set_dataloader : torch.utils.data.DataLoader
        The DataLoader providing the data for which to calculate activations.

    Returns
    -------
    None
    """
    
    if reduction_method_name.lower() == 'pca':
        reduction_method = PCA(n_components=2)
    elif reduction_method_name.lower() == 'tsne':
        reduction_method = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method_name}")

    fig, axes = plt.subplots(1, len(layers_idx), figsize=(20, 6))
    for i, layer in enumerate(layers_idx):
        # Extract activations from a specific layer
        layer_activation, labels = get_activations(model, set_dataloader, layer)
        # # Flatten the activations for dimensionality reduction
        # flattened_activations = layer_activation.reshape(layer_activation.shape[0], -1)

        # Apply PCA
        reduced_activations = reduction_method.fit_transform(layer_activation)
        # Plot the reduced activations
        axes[i].scatter(reduced_activations[:, 0], reduced_activations[:, 1], alpha=0.7, c=labels)
        axes[i].set_title(f"PCA of Layer {layer} Activations")
        axes[i].set_xlabel("Principal Component 1")
        axes[i].set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()

def visualize_cav(clf, X, len_concept_activations):
    """
    Visualize Concept Activation Vectors (CAV) in a 2D space using PCA.

    This function performs dimensionality reduction on activations and the 
    CAV using PCA, then visualizes the concept activations, random activations, 
    and the CAV direction in a 2D scatter plot.

    Parameters
    ----------
    clf : sklearn.linear_model._base.LinearClassifierMixin
        The trained linear classifier with a coefficient representing the CAV.
    X : np.ndarray
        The activations from which to derive concept and random activations.
    len_concept_activations : int
        The number of activations corresponding to the concept examples 
        in the dataset, used to differentiate between concept and random activations.

    Returns
    -------
    None
    """

    cav = clf.coef_.flatten()
    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(X)
    reduced_cav = pca.transform(cav.reshape(1, -1))
    # separate reduced activations
    concept_reduced = reduced_activations[:len_concept_activations]
    random_reduced = reduced_activations[len_concept_activations:]
    # plot concept and random activations
    plt.scatter(concept_reduced[:, 0], concept_reduced[:, 1], label="Concept", alpha=0.7)
    plt.scatter(random_reduced[:, 0], random_reduced[:, 1], label="Random", alpha=0.7)

    # add the CAV as an arrow
    cav_direction = reduced_cav[0]
    plt.arrow(0, 0, cav_direction[0], cav_direction[1], color="red", label="CAV", 
            head_width=0.5, head_length=0.5, alpha=0.9)

    # customize the plot
    plt.title("CAV Visualization in Hidden Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()