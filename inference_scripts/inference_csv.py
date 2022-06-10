import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from utils.iotools import save_disp, save_seg
from models import get_model
from datasets.filelist_inference import FileList, scared_stats
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import concurrent


parser = argparse.ArgumentParser('run inference of multinet model')
parser.add_argument('root_input_path', help='path to append add in front of all input paths in csv')
parser.add_argument('root_output_path', help='path to append add in front of all output paths in csv')
parser.add_argument('filelist_path', help='path to csv file containing input and output paths')
parser.add_argument('model_type', help='encoder variation',choices=['resnet34', 'light'])
parser.add_argument('model_weights', help='path to model weights')
parser.add_argument('--mode', help='mode to run the multitask network', choices=['multitask', 'segmentation', 'disparity'], default='multitask')
parser.add_argument('--cuda_id', help='cuda device to use', default=0, type=int)
parser.add_argument('--segmentation_threshold', help='segmentation binary threshold', default=0.5, type=float)
parser.add_argument('--disparity_scale_factor', help='disparity scale factor', default=128.0, type=float)
parser.add_argument('--jobs','-j', help='concurent workeres to save images', default=2, type=int)
parser.add_argument('--use_amp', help='run in mixed precision mode', default=False, action='store_true')

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
    model = Model(max_disparity=320,n_classes=1, mode=args.mode)
    model.load_state_dict(torch.load(args.model_weights), strict=False)
    model.to(device)
    model.eval()

    evaluation_transforms = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(scared_stats['mean'], scared_stats['std'])])
    dataset = FileList(args.root_input_path, args.root_output_path, args.filelist_path, evaluation_transforms)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=10,
                            drop_last=False,
                            pin_memory=True)

    total_time=0

    parallel_saver = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)

    with torch.no_grad():
    # iterate over each path, compute output and save it, optionally collect statistics
        for batch in tqdm(dataloader, total =len(dataloader), desc='sampled processed'):
            left_tensor, right_tensor, sample_idx = batch
            
            *_, disparity_out_p, segmentation_out_p = dataset.get_paths(sample_idx)


            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)

            # run inference
            with torch.cuda.amp.autocast(enabled=(args.use_amp and device=='cuda')):
                disparity_scales, segmentation_logits = model(left_tensor, right_tensor)


            # save outputs
            if disparity_scales is not None and disparity_out_p is not None:
                parallel_saver.submit(save_disp,disparity_out_p, disparity_scales[0], size=None, scale_factor=args.disparity_scale_factor)
            
            if segmentation_logits is not None and segmentation_out_p is not None:
                parallel_saver.submit(save_seg,segmentation_out_p, segmentation_logits, size=None, threshold=args.segmentation_threshold)