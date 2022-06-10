from glob import escape
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

ris17_stats = {'mean': [0.4899,	0.3078,	0.3474],
                'std': [0.1800,	0.1592,	0.1763]}

scared_stats = {'mean': [0.5339, 0.3666, 0.4537],
                'std': [0.2006, 0.1978, 0.2128]}

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}



class FileList(Dataset):

    def __init__(self, root_input_dir, root_output_dir, file_list, transforms=None):
        root_input_dir = Path(root_input_dir).resolve()
        root_output_dir = Path(root_output_dir).resolve()
        f_lst = Path(file_list).resolve() 

        with open(f_lst, 'r') as path_csv:
            paths = path_csv.readlines()
        
        self.left_image_paths=[]
        self.right_image_paths=[]
        self.out_disparity_paths=[]
        self.out_segmentation_paths=[]

        for sample_paths in paths:
            paths= sample_paths.strip().split(',')
            self.left_image_paths.append(root_input_dir/paths[0])
            self.right_image_paths.append(root_input_dir/paths[1])
            if len(paths)>2:
                self.out_disparity_paths.append(root_output_dir/paths[2])
            if len(paths)>3:
                self.out_segmentation_paths.append(root_output_dir/paths[3])

        assert(len(self.left_image_paths) == len(self.right_image_paths))

        self.transforms = transforms

    def __getitem__(self, index):
        left = Image.open(self.left_image_paths[index]).convert('RGB')
        right = Image.open(self.right_image_paths[index]).convert('RGB')

        if self.transforms:
            left = self.transforms(left)
            right = self.transforms(right)

        sample = left, right, index
        return sample
    
    def get_paths(self, index):
        paths = [self.left_image_paths[index], self.right_image_paths[index]]
        
        if self.out_disparity_paths:
            paths.append(self.out_disparity_paths[index])
        else:
            paths.append(None)
        
        if self.out_segmentation_paths:
            paths.append(self.out_segmentation_paths[index])
        else:
            paths.append(None)
        
        return paths
    def __len__(self):
        return len(self.right_image_paths)
