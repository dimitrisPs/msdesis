from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import re
import sys



imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}

class FlyingThings3D(Dataset):

    def __init__(self, root_dir, train=True, transforms=None):
        
        self.root_dir = Path(root_dir).resolve()
        self.transforms = transforms
        self.left_paths = []
        self.right_paths = []
        self.disparity_paths = []

        self.left_paths, self.right_paths, self.disparity_paths = self.collect_samples(self.root_dir, train=train)
        assert((len(self.left_paths) == len(self.right_paths))and (len(self.left_paths) == len(self.disparity_paths)))

    def __getitem__(self, index):
        left = Image.open(self.left_paths[index]).convert('RGB')
        right = Image.open(self.right_paths[index]).convert('RGB')
        
        disparity = self.read_sceneflow_disparity(self.disparity_paths[index])
        sample = left, right, disparity 
        if self.transforms:
            sample = self.transforms(sample)
        sample = sample[0], sample[1], np.expand_dims(sample[2], 0)

        return sample
    
    def __len__(self):
        return len(self.left_paths)
    
    def read_sceneflow_disparity(self, path):
        file = open(path, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if (sys.version[0]) == '3':
            header = header.decode('utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        if (sys.version[0]) == '3':
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        else:
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        if (sys.version[0]) == '3':
            scale = float(file.readline().rstrip().decode('utf-8'))
        else:
            scale = float(file.readline().rstrip())
            
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return np.abs(np.ascontiguousarray(data,dtype=np.float32))

    def collect_samples(self, root_dir, train):
        root_dir = Path(root_dir)
        if train:
            root_dir = root_dir/'train'
        else:
            root_dir= root_dir/'val'

        images_files_paths = [str(p) for p in root_dir.rglob('*.png')]
        left_paths = sorted([p for p in images_files_paths if 'left' in p])
        right_paths = sorted([p for p in images_files_paths if 'right' in p])      
        
        disparity_files_paths = [str(p) for p in root_dir.rglob('*.pfm')]
        disparity_paths = sorted([p for p in disparity_files_paths if 'left' in p])
        assert((len(left_paths) == len(right_paths))and (len(left_paths) == len(disparity_paths)))

        return left_paths, right_paths, disparity_paths
