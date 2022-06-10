from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

scared_stats = {'mean': [0.5339, 0.3666, 0.4537],
                'std': [0.2006, 0.1978, 0.2128]}


class SCARED(Dataset):

    def __init__(self, root_dir, dataset_id, keyframe_id, transforms=None):
        root_dir = Path(root_dir).resolve()
        left_img_dir = root_dir /('dataset_'+str(dataset_id))/('keyframe_'+str(keyframe_id))/'data'/ 'left_rectified'
        right_img_dir = root_dir /('dataset_'+str(dataset_id))/('keyframe_'+str(keyframe_id))/ 'data'/ 'right_rectified'
        left_disparity = root_dir /('dataset_'+str(dataset_id))/('keyframe_'+str(keyframe_id))/ 'data'/ 'disparity'

        assert left_img_dir.is_dir() and right_img_dir.is_dir() and left_disparity.is_dir()

        self.left_paths = sorted([str(file_path) for file_path in left_img_dir.iterdir()])
        self.right_paths = sorted([str(file_path) for file_path in right_img_dir.iterdir()])
        self.disparity_paths = sorted([str(file_path) for file_path in left_disparity.iterdir()])
        assert((len(self.left_paths) == len(self.right_paths))and (len(self.left_paths) == len(self.disparity_paths)))

        self.transforms = transforms

    def __getitem__(self, index):
        left = Image.open(self.left_paths[index]).convert('RGB')
        right = Image.open(self.right_paths[index]).convert('RGB')
        disparity = self.read_disparity(self.disparity_paths[index])
        sample = left, right, disparity 
        if self.transforms:
            sample = self.transforms(sample)
        sample = sample[0], sample[1], np.expand_dims(sample[2], 0)
        return sample
    
    def __len__(self):
        return len(self.left_paths)
        
    def read_disparity(self, path):
        disparity = Image.open(path)
        disparity = np.ascontiguousarray(disparity,dtype=np.float32)/128.
        return disparity