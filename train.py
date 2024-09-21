from glob import glob
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch

from utils.util_inout import getinout, getdatainfo, labeldist
from utils.util_traintest import EarlyStopping, CDB_loss, getCDBmetric_multiclass


def validate(dataloader_val, output_size, model, device, args):
    model.eval()
    gt_val = torch.empty(len(dataloader_val), dtype=torch.float32, device=device) if output_size == 1 else torch.empty(len(dataloader_val), dtype=torch.long, device=device)
    pred_val = torch.empty( len(dataloader_val), dtype=torch.float32, device=device) if output_size == 1 else torch.empty((len(dataloader_val), output_size), dtype=torch.float32, device=device)
    info_val = []
    with torch.no_grad():
        for i, data_val in enumerate(tqdm(dataloader_val, desc='validation', total=len(dataloader_val), leave=False)):
            input, gt = getinout(data_val, args.input, args.pred, args.slice_len, device)
            studyid, numslices = getdatainfo(data_val)
            if type(input) != list:
                pred = model(input)
            else:
                pred = model(input[0], input[1])
            gt_val[i] = gt
            if output_size == 1:
                pred_val[i] = pred
            else:
                pred_val[i,:] = pred
            info_val.append([studyid, numslices])

    return gt_val, pred_val, info_val



def train_validate(args, dataloader_train, dataloader_val, model, optimizer, criterion, output_size, writer, device, save_path):
    auc_val_max = -1
    early_stopping = EarlyStopping(args.earlystopping_patience)
    for epoch in tqdm(range(1, args.epoch + 1)):
        # Train
        model.train()
        loss_list = []
        gt_list = []
        pred_list = []
        for data_train in dataloader_train:
            input, gt = getinout(data_train, args.input, args.pred, args.slice_len, device)
            optimizer.zero_grad()
            if type(input) != list:
                pred = model(input)
            else:
                pred = model(input[0], input[1])
            if args.pred == 'goseldl' or args.pred == 'gose4clsldl': #label distribution learning
                gt_ldl = labeldist(gt, output_size, device)
                loss_train = criterion(pred, gt_ldl)
            else:
                loss_train = criterion(pred, gt)
            loss_list.append(loss_train.item())
            gt_list.extend(gt.detach().cpu().numpy())
            if output_size == 1: #binary class
                pred = torch.sigmoid(pred) 
            else: #multi class
                m = torch.nn.Softmax(dim=1) 
                pred = m(pred)
            pred_list.extend(pred.detach().cpu().numpy().tolist())
            loss_train.backward()
            optimizer.step()  
        if output_size == 1: #binary class
            auc = roc_auc_score(y_true=np.array(gt_list), y_score=np.array(pred_list))
        else: #multi class
            auc = roc_auc_score(y_true=np.array(gt_list), y_score=np.array(pred_list), multi_class='ovr')
        if args.mist == 0:
            writer.add_scalar("loss/train", np.mean(np.array(loss_list)), epoch)
            writer.add_scalar("AUC/train", auc, epoch)
        
        # Validation
        gt_val, pred_val, _ = validate(dataloader_val, output_size, model, device, args)
        loss_val = criterion(pred_val, gt_val).item()

        if output_size == 1: #binary class
            pred_val = torch.sigmoid(pred_val)
            auc_val = roc_auc_score(y_true=gt_val.detach().cpu().numpy(), y_score=pred_val.detach().cpu().numpy())
        else: #multi class
            m = torch.nn.Softmax(dim=1) 
            pred_val = m(pred_val)
            auc_val = roc_auc_score(y_true=gt_val.detach().cpu().numpy(), y_score=pred_val.detach().cpu().numpy(), multi_class='ovr')
            
            if (args.pred == 'goseCDBloss' or args.pred == 'gose4clsCDBloss') and epoch % 3 == 0:
                class_wise_metric = getCDBmetric_multiclass(y_true=np.array(gt_list), y_pred=np.array(pred_list))
                print(f'class_wise_metric: {class_wise_metric}')
                class_difficulty = 1- np.array(class_wise_metric)
                criterion = CDB_loss(class_difficulty=class_difficulty)
            
        if args.mist == 0:
            writer.add_scalar("loss/validation", loss_val, epoch)
            writer.add_scalar("AUC/validation", auc_val, epoch)
        early_stopping(loss_val) #Early stopping 
        if early_stopping.early_stop:
            break
        
        # Snapshot
        if not os.path.exists(os.path.join(save_path, 'ckpt', 'snapshot')):
            os.path.os.makedirs(os.path.join(save_path, 'ckpt', 'snapshot'))
        if not os.path.exists(os.path.join(save_path, 'ckpt', 'best')):
            os.path.os.makedirs(os.path.join(save_path, 'ckpt', 'best'))
            
        if epoch % args.snapshot_interval == 0:
            modelinfo = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_val': loss_val,
                'auc_val' : auc_val
                }
            savefilename = os.path.join(save_path, 'ckpt', 'snapshot', args.input+'_'+args.pred+'_epoch{}_{:.6}.pth'.format(str(epoch).zfill(4), str(float(auc_val)).replace('.', '_')))
            torch.save(modelinfo, savefilename)

        #Save the best model
        if epoch == 1 or auc_val > auc_val_max:
            auc_val_max = auc_val
            modelinfo = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_val': loss_val,
                'auc_val' : auc_val
                }
            #Remove currnt pth file if any
            for f in glob(os.path.join(save_path, 'ckpt', 'best', args.input+'_'+args.pred+'_epoch*.pth')):
                os.remove(f)
            savefilename = os.path.join(save_path, 'ckpt', 'best', args.input+'_'+args.pred+'_epoch{}_{:.6}.pth'.format(str(epoch).zfill(4), str(float(auc_val)).replace('.', '_')))
            torch.save(modelinfo, savefilename)
        





