import argparse
from pathlib import Path
from tqdm import tqdm
from utils.iotools import save_seg
from models import get_model
from datasets.RIS_17_binary_evaluation_monocular import RIS17
from datasets import get_data_paths
import torch
from torchvision import transforms as tv_transforms
from utils import RunningAvg
from utils.error_metrics import mIoU

import os
import pandas as pd

from datasets import scared_stats




parser = argparse.ArgumentParser('Evaluate binary segmentation on RIS2017 dataset')
parser.add_argument('model_type', help='select model to load', choices=['light', 'resnet34'])
parser.add_argument('model_weights', help='path to model weights')
parser.add_argument('--save_predictions', help='directory to save network output,  if not specified, the script will only save the score.')
parser.add_argument('--csv', help='path to save evalution scores', default='./ris17_scores.csv')
parser.add_argument('--cuda_id', help='cuda device to use', default=0)
parser.add_argument('--segmentation_threshold', help='segmentation binary threshold', default=0.5, type=float)

if __name__ == '__main__':
    args = parser.parse_args()

    #setup cuda
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        from torch.backends import cudnn
        cudnn.benchmark=True
    



    # configure the network
    Model = get_model(args.model_type)
    model = Model(max_disparity=320,n_classes=1, mode='segmentation')
    model.load_state_dict(torch.load(args.model_weights), strict=True)
    model.to(device)
    model.eval()

    # load data
    eval_transforms = tv_transforms.Compose([tv_transforms.ToTensor(),
                                             tv_transforms.Normalize(scared_stats['mean'],scared_stats['std'])])
    dataset_path = get_data_paths('ris17_eval_monocular')

    scores = {'dataset':[], 'IoU':[]}


    for dataset_id in tqdm(range(1,11), desc='evaluation'):
        # evaluate on dataset: dataset_id
        # get dataset and create a dataloader
        dataset = RIS17(dataset_path, dataset_id=dataset_id, transforms=eval_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        accuracy = RunningAvg()
   
        for frame_id, sample in tqdm(enumerate(dataloader), total= len(dataloader), leave=False):
            # load left, right, and augment one by one
            left_img, gt_binary_seg = sample
            left_img = left_img.to(device)
            gt_binary_seg = gt_binary_seg.to(device)

            # run inference, since the multitask model is designed to accept stereo inputs
            # we need to provide a second image as input. Because the segmentation
            # head only shared the feature extruction layer with the disparity head
            # passing two time the left image does not affect the segmentation performance.
            with torch.cuda.amp.autocast(enabled=True):
                _, segmentation_logits = model(left_img, left_img)

            segmentation_prediction = torch.sigmoid(segmentation_logits)>args.segmentation_threshold


            # save outputs
            if args.save_predictions is not None:
                segmentation_out_p = Path(args.save_predictions)/('instrument_dataset_'+str(dataset_id))/'binary'/('frame{:03d}.png'.format(frame_id))
                segmentation_out_p.parent.mkdir(exist_ok=True, parents=True)
                save_seg(segmentation_out_p, segmentation_logits, threshold=args.segmentation_threshold)

            accuracy.append(mIoU(segmentation_prediction, gt_binary_seg))

        scores['dataset'].append('instrument_dataset_'+str(dataset_id))
        scores['IoU'].append(accuracy.get_val().item())



    scores_df = pd.DataFrame(scores)
    scores_df.loc['10']=['final_mean',*list(scores_df.mean(numeric_only=True))]
    Path(args.csv).parent.mkdir(exist_ok=True, parents=True)
    scores_df.to_csv(args.csv)
    print(scores_df)

