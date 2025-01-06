from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.2, 
        hue=0.1
    ),  # Adjust brightness, contrast, saturation, and hue
    transforms.RandomRotation(
        degrees=(-30, 30),  # Rotate images randomly between -30 to 30 degrees
        interpolation=transforms.InterpolationMode.BILINEAR,  # Interpolation method
        expand=False,       # Whether to expand the image to fit the new orientation
        fill=0              # Fill color for areas outside the original image (default is black)
    ),
    transforms.RandomHorizontalFlip(),  # Flip images horizontally
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])