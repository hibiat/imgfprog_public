import datetime
from glob import glob
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import parameters
from train import train_validate
from test import test
from utils.util_traintest import crossval, CDB_loss
from utils.util_inout import getinputsize
from dataset import dataset
from models import model_selector

if __name__ == '__main__':
    args = parameters.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(device)

    dataset_all = dataset(input = args.input,
                        path_imaginginput=args.path_imaginginput,
                        summary_csv=args.summary_csv,
                        brainarea_csv=args.brainarea_csv,
                        treatment_csv = args.treatment_csv,
                        impact_csv=args.impact_csv,
                        feature_extracted_model=args.feature_extracted_model,
                        slice_len=args.slice_len,
                        targetimgsize = args.targetimgsize,
                        maxbrainarearatiothr=args.maxbrainarearatiothr,
                        disposition_csv = args.disposition_csv,
                        exclude_dead = args.exclude_dead)
    dataset_trainval = crossval(dataset_all=dataset_all, k_fold=args.k_fold, random_seed=args.seed)

    input_size = getinputsize(args)
    
    time = datetime.datetime.now()
    time_format = '{}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second)
    boardlayout = {
        args.input+'_'+args.pred: {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "AUC": ["Multiline", ["AUC/train", "AUC/validation"]],
        },
    }

    
    if args.foldrun is not None:
        run_k = [args.foldrun]
    else:
        run_k = range(args.k_fold)
    for k in run_k:
        print(f'Cross validation ({k})')
        if args.pred == 'gose' or args.pred == 'goseldl' :
            criterion = torch.nn.CrossEntropyLoss()
            output_size = 7
            print('7-class (Cross Entropy Loss)')
        elif args.pred == 'goseCDBloss' :
            output_size = 7
            criterion = CDB_loss(class_difficulty = np.ones(output_size, dtype=np.float32))
            print('7-class (Class-Wise Difficulty-Balanced Loss)')
        elif args.pred == 'disposition_5cls':
            criterion = torch.nn.CrossEntropyLoss()
            output_size = 5
            print('5-class (Cross Entropy Loss)')
        elif args.pred == 'gose4clsCDBloss' :
            output_size = 4
            criterion = CDB_loss(class_difficulty = np.ones(output_size, dtype=np.float32))
            print('4-class (Class-Wise Difficulty-Balanced Loss)')  
        elif args.pred == 'gose4cls' or args.pred == 'gose4clsldl':
            criterion = torch.nn.CrossEntropyLoss()
            output_size = 4
            print('4-class (Cross Entropy Loss)')
        elif args.pred == 'deadhomenohome':
            criterion = torch.nn.CrossEntropyLoss()
            output_size = 3
            print('3-class (Cross Entropy Loss)')
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            output_size = 1
            print('Binary class (Binary Cross Entropy with Logit Loss)')    

        model = model_selector(model_name=args.arc, input=args.input, input_size=input_size, output_size=output_size)
        if args.device is None and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        if args.device is not None:
            device = torch.device("cuda:{}".format(args.device))
            torch.cuda.set_device(device)
            print(f'Using {device} only')
        model.to(device)
        optimizer= optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        if args.pretrained_ckpt is not None:
            checkpoint = torch.load(args.pretrained_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Weights loaded from "{}"'.format(args.pretrained_ckpt))
        
        dataset_train = dataset_trainval[k][0] 
        #dataset_train = dataset_all #train using all data
        dataset_val = dataset_trainval[k][1]
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch, pin_memory=False, num_workers=4, shuffle=True) 
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False, num_workers=4, shuffle=False) 
        
        if args.jobidentifier is not None:
            save_path = os.path.join(args.path_saveroot, args.input+'_'+args.pred, args.arc, args.feature_extracted_model, args.jobidentifier, 'fold_'+str(k), time_format)
        else:
            save_path = os.path.join(args.path_saveroot, args.input+'_'+args.pred, args.arc, args.feature_extracted_model, 'fold_'+str(k), time_format)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(file=os.path.join(save_path, 'parameters.txt'), mode='w') as f:
            for key, value in vars(args).items():
                f.write('%s:%s\n' % (key, value))
        if args.mist == 0: 
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter('runs/' + args.input+'_'+args.pred + '_'+ args.arc + '_' + args.feature_extracted_model + '_' + time_format + '_fold' + str(k))
            writer.add_custom_scalars(boardlayout)
        else:
            writer = None
        
        train_validate(args=args,
                        dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        output_size=output_size,
                        writer=writer, device=device, save_path=save_path)
        if args.mist == 0: 
            writer.close()

        bestmodel = list(glob(os.path.join(save_path, 'ckpt', 'best', args.input+'_'+args.pred+'_epoch*.pth')))[0]
        print(f'Best model "{os.path.basename(bestmodel)}" is loaded')
        checkpoint = torch.load(bestmodel)
        model.load_state_dict(checkpoint['model_state_dict'])
        test(args=args,
                dataloader_val=dataloader_val,
                model=model,
                output_size=output_size,
                device=device,
                save_path=save_path,
                time_format = time_format)


