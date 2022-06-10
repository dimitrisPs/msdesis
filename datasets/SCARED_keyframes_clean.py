from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

scared_stats = {'mean': [0.5339, 0.3666, 0.4537],
                'std': [0.2006, 0.1978, 0.2128]}


class SCARED(Dataset):

    def __init__(self, root_dir, train=True, transforms=None):
        root_dir = Path(root_dir).resolve()
        left_img_dir = root_dir / 'left_rectified'
        right_img_dir = root_dir / 'right_rectified'
        left_disparity = root_dir / 'disparity'
        assert left_img_dir.is_dir() and right_img_dir.is_dir() and left_disparity.is_dir()

        left_paths = sorted([str(file_path) for file_path in left_img_dir.iterdir()])
        right_paths = sorted([str(file_path) for file_path in right_img_dir.iterdir()])
        disparity_paths = sorted([str(file_path) for file_path in left_disparity.iterdir()])
        assert((len(left_paths) == len(right_paths))and (len(left_paths) == len(disparity_paths)))

        self.transforms = transforms


        # split without tools in training, datasets 4,5,6 do not have good enough
        # calibration parameters to stereo rectify input frames. During training
        # using samples from those datasets decreases the performance of the
        # resulting model.
        good = [0,1,2,4,5,6,8,9,11,12,13,14,30,31,32,33,34]
        eval_samples = [3,7,10,25,26,27,28,29]
        if train:
            self.left_paths = [left_paths[i] for i in good]
            self.right_paths = [right_paths[i] for i in good]
            self.disparity_paths = [disparity_paths[i] for i in good]
        else:
            self.left_paths = [left_paths[i] for i in eval_samples]
            self.right_paths = [right_paths[i] for i in eval_samples]
            self.disparity_paths = [disparity_paths[i] for i in eval_samples]

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