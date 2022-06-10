from pathlib import Path
import json

from .RIS_17_binary_evaluation_monocular import RIS17 as ris17_monocular_eval, ris17_stats
from .SCARED_keyframes_clean import SCARED, scared_stats
from .FlyingThings3D import FlyingThings3D, imagenet_stats

from .RIS17_binary import RIS17 as RIS17, ris17_stats
from .RIS17_binary_124 import RIS17 as RIS17_124, ris17_stats
from .RIS_17_binary_monocular import RIS17 as ris17_monocular, ris17_stats

path_json = Path(__file__).parent / "dataset_paths.json"

def get_dataset(dataset:str):
    dataset_request=dataset.lower()
    if dataset_request == 'flyingthings3d':
        return FlyingThings3D, imagenet_stats
    elif dataset_request == 'scared':
        return SCARED, scared_stats
    elif dataset_request == 'ris17':
        return RIS17, ris17_stats
    elif dataset_request == 'ris17_124':
        return RIS17_124, ris17_stats
    elif dataset_request == 'ris17_train_monocular':
        return ris17_monocular, ris17_stats
    elif dataset_request == 'ris17_eval_monocular':
        return ris17_monocular_eval, ris17_stats
    else:
        raise NotImplementedError

def get_data_paths(dataset:str):
    with open(path_json, 'r') as paths:
        paths = json.load(paths)
    if dataset in paths.keys():
        return paths[dataset]
    else:
        raise NotImplementedError