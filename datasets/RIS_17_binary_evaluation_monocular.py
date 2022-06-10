from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

ris17_stats = {'mean': [0.4899,	0.3078,	0.3474],
                'std': [0.1800,	0.1592,	0.1763]}

class RIS17(Dataset):

    def __init__(self, root_dir, dataset_id, transforms=None):
        assert isinstance(dataset_id, int)
        assert (dataset_id>0) and (dataset_id<=10)
        self.transforms = transforms
        root_dir = Path(root_dir)

        
        sub_dataset = root_dir / ('instrument_dataset_'+str(dataset_id))


        left_rgb_dir = sub_dataset/'left_frame'
        ground_truth_dir = sub_dataset/'ground_truth'/'binary'
        
        self.left_paths = sorted([p for p in left_rgb_dir.iterdir()])
        self.binary_mask_paths = sorted([p for p in ground_truth_dir.iterdir()])
        
        assert len(self.left_paths) == len(self.binary_mask_paths)

        # assert paths

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


