import torch
from models.resnet import MultiOutputResNet
from models.regnet import MultiOutputRegNet
from models.tinyvit import MultiOutputTinyVit

from models.efficientnet import( 
    MultiOutputEfficientNetB0, 
    MultiOutputEfficientNetB1, 
    MultiOutputEfficientNetB2,
)

def get_models_dict():
    models = {
        'resnet18': MultiOutputResNet,
        'regnet400': MultiOutputRegNet,
        'efficientnetb0': MultiOutputEfficientNetB0,
        'efficientnetb1': MultiOutputEfficientNetB1,
        'efficientnetb2': MultiOutputEfficientNetB2,
        'tinyvit': MultiOutputTinyVit
    }    
    return models
def model_factory(network_name:str, num_classes:int):
    models = get_models_dict()
    if network_name not in models:
        raise ValueError(f"Invalid network name: {network_name}")
    model_class = models[network_name]
    model = model_class(num_classes)
    return model

def load_model(model_name:str, num_classes:int, weights_path:str):
    model = model_factory(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path))
    return model