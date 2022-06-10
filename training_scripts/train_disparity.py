import os
import argparse
from PIL import Image


from tqdm import tqdm
import numpy as np
import wandb
import torch
import torch.optim as optim
import torchvision.transforms as tv_transforms

from models import get_model
from datasets import get_data_paths, get_dataset
from datasets import transforms
from losses import get_disparity_loss
from utils import Params, set_random_seed, RunningAvg
from utils.error_metrics import disparity_epe, disparity_bad3


parser = argparse.ArgumentParser(description= 'fully supervised disparity training')
parser.add_argument('--default_config', help= '.json file containing hyperparameter configuration')
parser.add_argument('--cuda_id', help='gpu number you want to use.', default=0, type=int)


if __name__ == "__main__":

    args, unknown = parser.parse_known_args()
    lower_eval_loss=None
    force_log=False
    config = Params(args.default_config)
    wandb.init(config=config.dict, project=config.project, name=config.experiment_tag)
    config = wandb.config
    
    if config.dataset =='flyingthings3d':
        from datasets import imagenet_stats as stats
    else:
        from datasets import scared_stats as stats
    
    
    
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
    training_transforms = tv_transforms.Compose([transforms.RandomCrop((config.tr_crop_h,config.tr_crop_w)),
                                                 transforms.RandomFlipVertical(),
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize(stats)])

    evaluation_transforms = tv_transforms.Compose([transforms.RandomCrop((config.eval_crop_h,config.eval_crop_w)),
                                                 transforms.ToTensor(), 
                                                 transforms.Normalize(stats)])
    
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
    

    Model = get_model(config.model)
    model = Model(max_disparity=config.max_disparity, n_classes=1, mode='disparity')

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


    DisparityLoss = get_disparity_loss(config.disparity_loss)
    disparity_criterion = DisparityLoss(config.supervision_scales, config.max_disparity, scale_weights=config.scale_weights).to(config.device)
    
    if config.use_onecycle_lr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr*10, steps_per_epoch=len(training_dl), epochs=config.epochs)
        
    
    for epoch in tqdm(range(config.epochs), desc= 'training_progress'):
        
        train_epoch_loss_acc = RunningAvg()
        eval_epoch_loss_acc = RunningAvg()
        epe_error_acc = RunningAvg()
        bad3_error_acc = RunningAvg()

        model.train()
        for batch in tqdm(training_dl, total=len(training_dl), desc='training (epoch:{:04d})'.format(epoch), leave=False):
            left, right, reference_disparity = batch
            
            left = left.to(config.device)
            right = right.to(config.device)
            reference_disparity = reference_disparity.to(config.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                disparity_scales, _ = model(left, right)
                training_loss = disparity_criterion(disparity_scales, reference_disparity)

            scaler.scale(training_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss_acc.append(training_loss.detach())
            del training_loss
            if config.use_onecycle_lr:
                scheduler.step()
            if config.overfit:
                break
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dl, total =len(eval_dl), desc='evaluating (epoch:{:04d})'.format(epoch),leave=False):
                left, right, reference_disparity = batch

                left = left.to(config.device)
                right = right.to(config.device)
                reference_disparity = reference_disparity.to(config.device)
                with torch.cuda.amp.autocast(enabled=config.use_amp):             
                    disparity_scales,_ = model(left, right)
                    eval_loss = disparity_criterion(disparity_scales, reference_disparity)
                

                eval_epoch_loss_acc.append(eval_loss.detach())

                epe = disparity_epe(disparity_scales[0], reference_disparity, max_disparity=config.max_disparity)
                bad3 = disparity_bad3(disparity_scales[0], reference_disparity, max_disparity=config.max_disparity)
                epe_error_acc.append(epe.detach())
                bad3_error_acc.append(bad3.detach())

                del eval_loss, epe, bad3
                if config.overfit:
                    break

        current_lr = optimizer.param_groups[0]['lr']
        
        
        tqdm.write(' epoch: {:05d}\t lr: {:.02E}\t training_loss: {:.03f}\t evaluation_loss:{:.03f}\t epe: {:.03f}\t bad3: {:.02f}'.format(epoch,
                                                                                                        current_lr,
                                                                                                        train_epoch_loss_acc.get_val(),
                                                                                                        eval_epoch_loss_acc.get_val(),
                                                                                                        epe_error_acc.get_val(),
                                                                                                        bad3_error_acc.get_val()))
        if lower_eval_loss is None:
            lower_eval_loss = eval_epoch_loss_acc.get_val()
        elif lower_eval_loss > eval_epoch_loss_acc.get_val():
            lower_eval_loss = eval_epoch_loss_acc.get_val()
            torch.save(model.state_dict(), str(config.experiment_tag+'_lowest_eval_loss.pt'))
            tqdm.write(f'lowest_eval_loss={eval_epoch_loss_acc.get_val()} at epoch {epoch}')
            force_log=True

        if ((epoch+1)%config.wandb_log == 0) or force_log:
            log_dict={'learning_rate':optimizer.param_groups[0]['lr'],
                    'Loss/Training':train_epoch_loss_acc.get_val(),
                    'Loss/Evaluation':eval_epoch_loss_acc.get_val(),
                    'Error/Epe':epe_error_acc.get_val(),
                    'Error/Bad3':bad3_error_acc.get_val(),
                    "sample": [wandb.Image(Image.fromarray(reference_disparity[0,0].detach().cpu().numpy().astype(np.uint8)), caption="gt"),
                                wandb.Image(tv_transforms.ToPILImage()(transforms.NormalizeInverse(stats['mean'], stats['std'])(left[0])), caption="input"),
                                wandb.Image(Image.fromarray(disparity_scales[0][0,0].detach().cpu().numpy().astype(np.uint8)), caption="prediction_f")]}
            wandb.log(log_dict, step=epoch+1)
            force_log=False
            
        
        if (epoch+1)%config.model_save_period == 0:
            torch.save(model.state_dict(), str(config.experiment_tag+'_epoch_'+str(epoch+1) + '__error_' + str(epe_error_acc.get_val().item()) + '.pt'))

    # save final model
    torch.save(model.state_dict(), config.experiment_tag+'_lr'+str(config.lr)+'_final_error__' + str(epe_error_acc.get_val().item()) + '.pt')
        
