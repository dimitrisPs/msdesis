from .multinet_light.multinet import Multinet as  multitask_light
from .multinet_resnet34.multinet import Multinet as  multitask_resnet32


def get_model(model:str):
    model_request=model.lower()
    
    if model_request == 'light':
        return multitask_light    
    elif model_request == 'resnet34':
        return multitask_resnet32   
    else:
        raise NotImplementedError