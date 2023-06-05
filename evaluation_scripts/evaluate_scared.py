import argparse
from pathlib import Path
from tqdm import tqdm
from utils.iotools import save_disp
from models import get_model
from datasets.SCARED_evaluation import SCARED, scared_stats
from datasets import get_data_paths
import torch
from torchvision import transforms as tv_transforms
from datasets.transforms import Normalize, ToTensor
from utils import RunningAvg
from utils.error_metrics import disparity_bad3, disparity_epe
import cv2
import numpy as np

import os
import pandas as pd



parser = argparse.ArgumentParser('Evaluate disparity/depth on SCARED dataset')
parser.add_argument('model_type', help='select model to load', choices=['light', 'resnet34'])
parser.add_argument('model_weights')
parser.add_argument('--save_predictions', help='directory to save network output,  if not specified, the script will only save the score.')
parser.add_argument('--csv', help='path to save evalution scores', default='./scared_scores.csv')
parser.add_argument('--cuda_id', help='cuda device to use', default=0)
parser.add_argument('--disparity_scale_factor', help='Disparity scale factor to multiply the original disparity before saving it as 16bit .png, default=128', default=128.0)
parser.add_argument('--depth', help='calculate depth mean absolute error', action='store_true')


def load_Q_mat(calib_path):
    calib = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    Q = calib.getNode("Q").mat()
    calib.release()
    return Q



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
    model = Model(max_disparity=320,n_classes=1, mode='disparity')
    model.load_state_dict(torch.load(args.model_weights), strict=False)
    model.to(device)
    model.eval()

    # load data
    eval_transforms = tv_transforms.Compose([ToTensor(),
                                             Normalize(scared_stats)])
    dataset_path = get_data_paths('scared_test')

    scores = {'dataset':[], 'EPE':[], 'Bad3':[]}

    if args.depth:
        scores['Depth_MAE']=[]


    for dataset_id in tqdm(range(8,10), desc='datasets'):
        for keyframe_id in tqdm(range(0,5), leave=False, desc='keyframes'):
            # evaluate on dataset: dataset_id
            # get dataset and create a dataloader
            dataset = SCARED(dataset_path, dataset_id=dataset_id, keyframe_id=keyframe_id, transforms=eval_transforms)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)

            if args.depth:
                Q = load_Q_mat(Path(dataset_path)/('dataset_'+str(dataset_id))/('keyframe_'+str(keyframe_id))/'stereo_calib.json')

            epe_accumulator = RunningAvg()
            bad3_accumulator = RunningAvg()
            depth_error_accumulator = RunningAvg()
    
            for frame_id, sample in tqdm(enumerate(dataloader), total= len(dataloader), leave=False,):
                # load left, right, and augment one by one
                left_img, right_img, gt_disparity = sample
                left_img = left_img.to(device)
                right_img = right_img.to(device)
                gt_disparity = gt_disparity.to(device)
                if gt_disparity.count_nonzero()/torch.numel(gt_disparity)<0.07:
                    continue
                with torch.cuda.amp.autocast(enabled=True):
                    disparity_scales, _ = model(left_img, right_img)

                # because the network does not have relu activation in the outpus we clip negative values
                disparity_prediction = disparity_scales[0]
                disparity_prediction[disparity_prediction<0]=0


                # save outputs
                if args.save_predictions is not None:
                    disparity_out_p = Path(args.save_predictions)/('dataset_'+str(dataset_id))/('keyframe_'+str(keyframe_id))/('frame{:06d}.png'.format(frame_id))
                    disparity_out_p.parent.mkdir(exist_ok=True, parents=True)
                    save_disp(disparity_out_p, disparity_prediction, scale_factor=args.disparity_scale_factor)


                bad3_accumulator.append(disparity_bad3(disparity_prediction, gt_disparity))
                epe_accumulator.append(disparity_epe(disparity_prediction, gt_disparity))

                if args.depth:
                    # calculate depth error
                    # create gt depth using Q matrix
                    # create inferred depth using Q matrix.
                    # take the mean absolute difference at pixels were gt > 0
                    cv_disparity = gt_disparity.detach().cpu().numpy()[0,0]
                    cv_prediction =  disparity_prediction.detach().cpu().numpy()[0,0].astype(np.float32)

        
                    ref_depth = cv2.reprojectImageTo3D(cv_disparity, Q)[:,:,2]
                    predicted_depth = cv2.reprojectImageTo3D(cv_prediction, Q)[:,:,2]


                    predicted_depth = np.nan_to_num(predicted_depth, posinf=0, neginf=0)
                    ref_depth = ref_depth[cv_disparity>0]
                    predicted_depth = predicted_depth[cv_disparity>0]
                    e= np.abs(ref_depth-predicted_depth)
                    e[predicted_depth<=0]=40
                    depth_error_accumulator.append(np.mean(e))


            scores['dataset'].append('dataset_'+str(dataset_id)+'_keyframe_'+str(keyframe_id))
            scores['EPE'].append(epe_accumulator.get_val().item())
            scores['Bad3'].append(bad3_accumulator.get_val().item())
            if args.depth:
                scores['Depth_MAE'].append(depth_error_accumulator.get_val())


    scores_df = pd.DataFrame(scores)
    scores_df.loc['10']=['dataset8_mean',*list(scores_df[:5].mean(numeric_only=True))]
    scores_df.loc['11']=['dataset9_mean',*list(scores_df[5:10].mean(numeric_only=True))]
    scores_df.loc['12']=['final_mean',*list(scores_df[-2:].mean(numeric_only=True))]
    Path(args.csv).parent.mkdir(exist_ok=True, parents=True)
    scores_df.to_csv(args.csv)
    print(scores_df)


