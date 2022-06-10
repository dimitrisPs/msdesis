from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

ris17_stats = {'mean': [0.4899,	0.3078,	0.3474],
                'std': [0.1800,	0.1592,	0.1763]}

class RIS17(Dataset):

    def __init__(self, root_dir,train=True, transforms=None, gt_transforms=None):
        if gt_transforms is not None:
            raise NotImplementedError
        self.transforms = transforms
        root_dir = Path(root_dir)
        
        left_paths = []
        right_paths = []
        binary_mask_paths = []

        
        sub_datasets_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

        
        for sub_dataset in sub_datasets_dirs:
            left_rgb_dir = sub_dataset/'left_frame'
            ground_truth_dir = sub_dataset/'ground_truth'/'binary'
            
            left_paths.extend(sorted([p for p in left_rgb_dir.iterdir()]))
            binary_mask_paths.extend(sorted([p for p in ground_truth_dir.iterdir()]))
        
        assert len(left_paths) == len(binary_mask_paths)
        # keep only dataset 8 for testing.
        if train:
            self.left_paths = left_paths[:-224]
            self.right_paths = right_paths[:-224]
            self.binary_mask_paths = binary_mask_paths[:-224]
        else:
            # use dataset 8 for testing
            self.left_paths = left_paths[-224:]
            self.right_paths = right_paths[-224:]
            self.binary_mask_paths = binary_mask_paths[-224:]

        assert len(self.left_paths) == len(self.binary_mask_paths)


        for l_p, gt_p in zip(self.left_paths, self.binary_mask_paths):
            assert l_p.parents[1] == gt_p.parents[2]
            assert l_p.name == gt_p.name  

    def __getitem__(self, index):
        left = Image.open(self.left_paths[index]).convert('RGB')
        bin_mask = self.read_bin_mask(self.binary_mask_paths[index])
        if self.transforms:
            left = self.transforms(left)
        sample = left, bin_mask.transpose(2,0,1)
        return sample
    
    def __len__(self):
        return len(self.left_paths)
    
    def read_bin_mask(self, path):
        bin_mask = Image.open(path)
        bin_mask = np.ascontiguousarray(bin_mask,dtype=np.float32)/255
        return np.expand_dims(bin_mask,-1)


