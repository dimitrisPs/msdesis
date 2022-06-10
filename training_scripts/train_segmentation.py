import argparse
import os
from PIL import Image


import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as tv_transforms
from tqdm import tqdm
import wandb


from models import get_model
from datasets import get_data_paths, get_dataset
from datasets import transforms
from losses import get_segmentation_loss
from utils import Params, set_random_seed, RunningAvg
from utils.error_metrics import mIoU
# from datasets import imagenet_stats as stats
from datasets import scared_stats as stats




parser = argparse.ArgumentParser(description= 'fully supervised binary segmentation training')
parser.add_argument('--default_config', help= '.json file containing hyperparameter configuration')
parser.add_argument('--cuda_id', help='gpu number you want to use.', default=0, type=int)


if __name__ == "__main__":

    args, unknown = parser.parse_known_args()
    lower_eval_loss=None
    force_log=False
    config = Params(args.default_config)
    wandb.init(config=config.dict, project=config.project, name=config.experiment_tag)
    config = wandb.config
    if config.device !='cpu':
        print(config.device)
        if args.cuda_id is None:
            exit()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
    
    if config.seed:
        set_random_seed(config.seed)

    # load data 

    TrainingDataset, _ = get_dataset(config.dataset)
    EvaluationDataset, _ = get_dataset(config.dataset)
    training_transforms = tv_transforms.Compose([tv_transforms.ToTensor(), 
                                                 tv_transforms.Normalize(stats['mean'], stats['std'])])

    evaluation_transforms = tv_transforms.Compose([tv_transforms.ToTensor(), 
                                                   tv_transforms.Normalize(stats['mean'], stats['std'])])
    
    training_dataset = TrainingDataset(get_data_paths(config.dataset),
                                                   transforms=training_transforms,
                                                   train=True)
    eval_dataset = EvaluationDataset(get_data_paths(config.dataset),
                                               transforms=evaluation_transforms,
                                               train=False)



    training_dl = torch.utils.data.DataLoader(training_dataset,
                                              batch_size=config.training_bs,
                                              shuffle=(not config.overfit),
                                              num_workers=config.dl_workers,
                                              drop_last=False,
                                              pin_memory=True)
    
    eval_dl = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=config.eval_bs,
                                          shuffle=(not config.overfit),
                                          num_workers=config.dl_workers,
                                          drop_last=False,
                                          pin_memory=True)
    

    model = get_model(config.model_type)(max_disparity=config.max_disparity, n_classes=config.n_classes, mode='segmentation')
    if config.log_gradients:
        if config.overfit:
            wandb.watch(model, log='all',log_freq=2)
        else:

            wandb.watch(model, log='all',log_freq=len(training_dl)+len(eval_dl))

    if config.load_model:
        print('loading pretrained model: {}'.format(config.load_model))
        try:
            model.load_state_dict(torch.load(config.load_model))
        except:
            pass
        finally:
            print('loading only disparity weights, the rest of the model has random weights')
            model.load_state_dict(torch.load(config.load_model), strict=False)
    
    model = model.to(config.device)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    SegmentationLoss = get_segmentation_loss(config.segmentation_criterion)
    segmentation_criterion = SegmentationLoss(alpha=config.segmentation_alpha).to(config.device)
    
    
    for epoch in tqdm(range(config.epochs), desc= 'training_progress'):
        
        train_epoch_loss_acc = RunningAvg()
        eval_epoch_loss_acc = RunningAvg()
        miou_error_acc = RunningAvg()

        model.train()
        for batch in tqdm(training_dl, total=len(training_dl), desc='training (epoch:{:04d})'.format(epoch), leave=False):
            left, reference_seg = batch
            left = left.to(config.device)
            reference_seg = reference_seg.to(config.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                # the network accepts stereo inputs, for segmentation the right 
                # channel does not matter, hence we pass two time the left one.
                # internally the network will not compute features for the second
                # input because is configured in segmentation mode
                _, segmentation_out = model(left, left)
                training_loss = segmentation_criterion(segmentation_out, reference_seg)

            scaler.scale(training_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss_acc.append(training_loss.detach())
            del training_loss
            if config.overfit:
                break
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dl, total =len(eval_dl), desc='evaluating (epoch:{:04d})'.format(epoch),leave=False):
                left, reference_seg = batch
                left = left.to(config.device)
                reference_seg = reference_seg.to(config.device)

                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    _, segmentation_out = model(left, left)
                    eval_loss = segmentation_criterion(segmentation_out, reference_seg)

                thresholded_segmentation = (torch.sigmoid(segmentation_out.detach())>0.5).float()
                error = 1- mIoU(thresholded_segmentation, reference_seg)

                eval_epoch_loss_acc.append(eval_loss.detach())
                miou_error_acc.append(error.detach())

                del eval_loss, error
                if config.overfit:
                    break

        

        current_lr = optimizer.param_groups[0]['lr']
        
            
        #telemetry, write to tensorboard and to log files. save latest model 
        tqdm.write(' epoch: {:05d}\t lr: {:.02E}\t training_loss: {:.03f}\t evaluation_loss:{:.03f}\t miou: {:.03f}'.format(epoch,
                                                                                                        current_lr,
                                                                                                        train_epoch_loss_acc.get_val(),
                                                                                                        eval_epoch_loss_acc.get_val(),
                                                                                                        miou_error_acc.get_val()))
        if lower_eval_loss is None:
            lower_eval_loss = eval_epoch_loss_acc.get_val()
        elif lower_eval_loss > eval_epoch_loss_acc.get_val():
            lower_eval_loss = eval_epoch_loss_acc.get_val()
            torch.save(model.state_dict(), str(config.experiment_tag+'_lowest_eval_loss.pt'))
            tqdm.write(f'lowest_eval_loss={eval_epoch_loss_acc.get_val()} at epoch {epoch}')
            force_log=True
        if (epoch+1)%config.wandb_log == 0:
            log_dict={'learning_rate':optimizer.param_groups[0]['lr'],
                    'Loss/Training':train_epoch_loss_acc.get_val().item(),
                    'Loss/Evaluation':eval_epoch_loss_acc.get_val().item(),
                    'Error/mIoU':miou_error_acc.get_val().item(),
                    "sample": [wandb.Image(tv_transforms.ToPILImage()(transforms.NormalizeInverse(stats['mean'], stats['std'])(left[0])), caption="input"),
                                wandb.Image(Image.fromarray(reference_seg[0,0].detach().cpu().numpy().astype(np.uint8)*255), caption="reference_segmentation"),
                                wandb.Image(Image.fromarray(thresholded_segmentation[0,0].cpu().numpy().astype(np.uint8)*255), caption="predicted_segmentation")]}
            wandb.log(log_dict, step=epoch+1)
            force_log=False
        
        if (epoch+1)%config.model_save_period == 0:
            torch.save(model.state_dict(), str(config.experiment_tag+'_epoch_'+str(epoch+1) + '__error_' + str(miou_error_acc.get_val().item()) + '.pt'))
    
    # save final model
    torch.save(model.state_dict(), config.experiment_tag+'_lr'+str(config.lr)+'_final_error__' + str(miou_error_acc.get_val().item()) + '.pt')
    
        
        
    
