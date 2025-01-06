import torch
from utils import select_device
def val(
    model, 
    val_loader, 
    criterion, 
):
    
    """
    This function validates a model on a given validation loader.

    It computes the loss and accuracy on the validation set and returns them.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): The loader for the validation set.
        criterion (nn.Module): The criterion to use to compute the loss.

    Returns:
        epoch_loss (float): The loss on the validation set.
        epoch_accuracy (float): The accuracy on the validation set.
    """
    device = select_device()
    epoch_loss = 0
    epoch_accuracy = 0
    epoch_samples = 0
    model.eval()
    with torch.no_grad():
        for images, label in val_loader:
            images = images.to(device)
            label = label.to(device)
            # Forward pass
            output = model(images)
            # Compute losses
            loss = criterion(output, label)
            epoch_loss += loss.item() * images.size(0)
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            epoch_accuracy+= (predicted == label).sum().item()
            epoch_samples += images.size(0)
    epoch_loss /= epoch_samples
    epoch_accuracy /= epoch_samples
    return epoch_loss, epoch_accuracy