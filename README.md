# Tenkys

This repository answer the following tasks:

- Train a vision classification model 
- Visualise the model’s hidden activation space preferably by plotting dimensionality
- Implement the TCAV 
- Train TCAV to recognise the “Arnie concept”  
- Visualise the Arnie CAV vector in the model’s hidden space.
- Use the trained TCAV model to find images in the train set for which the Arnie
  concept is high. These should represent the images with Arnie present.
- Change the labels of these images from “enemy” to “friend”.
- Train a new vision classification model on the fixed dataset and report its (hopefully
 improved) test set performanc

Tenkys is a PyTorch-based project for image classification, training, and concept-based analysis (TCAV).

## Project Structure
- train.py, val.py, utils.py: Training, validation, utility functions  
- dataset.py: Custom dataset class  
- transforms.py: Image transforms using torchvision  
- tcav.py, activation_utils.py: Concept-based analysis utilities  
- main.ipynb: End-to-end workflow demonstration  

## Setup
1. Install dependencies (PyTorch, torchvision, scikit-learn, pandas, etc.).
3. Adjust paths in train.py, val.py, and main.ipynb as needed.

### Installation via pip Requirements
```bash
pip install -r requirements.txt
```

## Usage
- Run the main.ipynb notebook for a complete demonstration.
- Use train.py to train a model, and val.py to validate it.
- Use tcav.py to perform concept-based analysis.

## Additional Notes
- Models supported: EfficientNet, RegNet, ResNet, ViT
- Customizable transforms in transforms.py
- Logging and checkpoint saving integrated in train.py
- The images in the dataset's folder are not moved during the correction process. Instead, a dataframe is created containing the filenames and their corresponding classes.

##### Contact 
Nicolas Franco Gonzalez
nicolas.franco-gonzalez24@imperial.ac.uk