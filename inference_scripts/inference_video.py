import os
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from utils.iotools import post_process_disparity, post_process_segmentation, prepare_cv_frame
from utils.calibrator import StereoCalibrator
from models import get_model
from datasets.SCARED_evaluation import scared_stats

parser = argparse.ArgumentParser('run inference of multinet model')
parser.add_argument('video_path', help='Path to input stereo video')
parser.add_argument('out_path', help=' Path to save video showing the network\'s output')
parser.add_argument('model_type', help='Type of encoder to use', choices =['light', 'resnet34'], default='light')
parser.add_argument('model_weights', help='path to model weights')
parser.add_argument('--max_disparity', help='Maximum disparity search range, default=320. For our 2D network this values should be the same as the one used during training', default=320, type=int)
parser.add_argument('--mode', help='mode to run the multitask network', choices=['multitask', 'segmentation', 'disparity'], default='multitask')
parser.add_argument('--cuda_id', help='cuda device to use', default=0, type=int)
parser.add_argument('--segmentation_threshold', help='segmentation binary threshold', default=0.5, type=float)
parser.add_argument('--disparity_scale_factor', help='Disparity scale factor to multiply the original disparity, default=1, because we save the output as a video and not as png, this is only useful when the disparity range is over 255', default=1, type=float)
parser.add_argument('--calib', help='path to stereo calibration .json file. If this is not set, the input video is considered rectified and its frames are fed directly to the network')
parser.add_argument('--rectification_alpha', help='Stereo rectification alpha to use in the stereo rectification', default=-1.0, type=float)
parser.add_argument('--stacking',help='Specify if left and right stereo frames are stacked horizontally or vertically in the input video', default='horizontal', choices=['horizontal', 'vertical'])
parser.add_argument('--save_images', action='store_true', help='save output as pngs instead of video.')


if __name__ == '__main__':
    args = parser.parse_args()

    #setup cuda
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        from torch.backends import cudnn
        cudnn.benchmark=True
    


    # configure io files.
    invideo = cv2.VideoCapture(args.video_path)
    Path(args.out_path).parent.mkdir(exist_ok=True, parents=True)
    video_fps = invideo.get(cv2.CAP_PROP_FPS)
    num_frames = int(invideo.get(cv2.CAP_PROP_FRAME_COUNT))
    stereo_frame_size = (int(invideo.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(invideo.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if args.save_images:
        if args.mode !='segmentation':
            disp_path = Path(args.out_dir)/'disparity'
            disp_path.mkdir(exist_ok=True, parents=True)
        if args.mode !='disparity':
            seg_path =  Path(args.out_dir)/'segmentation'
            seg_path.mkdir(exist_ok=True, parents=True)
    else:
        if args.stacking =='horizontal':
            o_h, o_w = stereo_frame_size[:2]
        else:
            o_h, o_w = [stereo_frame_size[0]//2, stereo_frame_size[1]*2]
        frame_size = (o_h, o_w)
        outvideo = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), video_fps, (o_w, o_h))



        
    
    # configure rectifier if needed.
    if args.calib:
        calibrator = StereoCalibrator()
        calib = calibrator.load(args.calib)


    # configure the network
    model_variant = args.model_type
    Model = get_model(model_variant)
    model = Model(max_disparity=args.max_disparity,n_classes=1, mode=args.mode)
    model.load_state_dict(torch.load(args.model_weights), strict=False)
    model.to(device)
    model.eval()



    # iterate over each path, compute output and save it, optionally collect statistics
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            # load left, right, and augment one by one

            ret, stereo_frame = invideo.read()
            if not ret:
                print('finished')
                break

            # convert to rgb color space
            stereo_frame = cv2.cvtColor(stereo_frame, cv2.COLOR_BGR2RGB)
            # split frame in left and right and apply augmentations
            if args.stacking =='horizontal':
                left_frame = stereo_frame[:, :frame_size[1]//2]
                right_frame = stereo_frame[:, frame_size[1]//2:]
            else:
                left_frame = stereo_frame[:frame_size[0],:]
                right_frame = stereo_frame[frame_size[0]:,:]



            if args.calib:
                left_frame, right_frame = calibrator.rectify(left_frame, right_frame, alpha = args.rectification_alpha)


            left_tensor = prepare_cv_frame(left_frame, (scared_stats['std'], scared_stats['mean']))
            right_tensor = prepare_cv_frame(right_frame, (scared_stats['std'], scared_stats['mean']))

            
            
            
            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)

            disparity_scales, segmentation_logits = model(left_tensor, right_tensor)
            

            # post_process outputs and save them as video or pngs
            if args.mode !='segmentation':
                disp = post_process_disparity(disparity_scales[0], scale_factor=args.disparity_scale_factor, size=frame_size)
            if args.mode !='disparity':
                seg = post_process_segmentation(segmentation_logits, size=frame_size, threshold=args.segmentation_threshold)

            if args.save_images:
                if args.mode !='segmentation':
                    cv2.imwrite(str(disp_path/f"{i:06d}.png"), disp)
                if args.mode !='disparity':
                    cv2.imwrite(str(seg_path/f"{i:06d}.png"), seg)

            if args.mode == 'multitask':
                out_frame = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                out_frame[:,:,2:][seg!=0]=0

            elif args.mode == 'disparity':
                out_frame = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            else:
                out_frame = seg


            out_frame = np.hstack((cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR), out_frame))

            outvideo.write(out_frame)

    invideo.release()
    outvideo.release()