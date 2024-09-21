import os
import csv
import torch
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.util_inout import getinout, getdatainfo
from utils.util_traintest import getmetrics, getmetrics_multiclass
import parameters
from dataset import dataset
from utils.util_traintest import crossval


def test(args, dataloader_val, model, output_size, device, save_path, time_format):
    model.eval()
    gt_test = torch.empty(len(dataloader_val), dtype=torch.float32, device=device) if output_size == 1 else torch.empty(len(dataloader_val), dtype=torch.long, device=device)
    pred_test = torch.empty(len(dataloader_val), dtype=torch.float32, device=device) if output_size == 1 else torch.empty((len(dataloader_val), output_size), dtype=torch.float32, device=device)
    info_list = []
    with torch.no_grad():
        for i, data_val in enumerate(tqdm(dataloader_val, desc='report', total=len(dataloader_val), leave=True)):
            input, gt = getinout(data_val, args.input, args.pred, args.slice_len, device)
            studyid, numslices = getdatainfo(data_val)
            if type(input) != list:
                pred = model(input)
            else:
                pred = model(input[0], input[1])
            gt_test[i] = gt
            if output_size == 1: #binary class
                pred_test[i] = torch.sigmoid(pred)
                info_list.append([studyid, numslices, gt.item(), pred_test[i].item()])
            else:
                m = torch.nn.Softmax(dim=1) 
                pred_test[i,:] = m(pred)
                info_list.append([studyid, numslices, gt.item()] + pred_test[i,:].tolist())

    if output_size == 1: #binary class
        metrics = getmetrics(y_true=gt_test.detach().cpu().numpy(), y_pred=pred_test.detach().cpu().numpy()) #y_pred needs not to be binarized
        
        with open(os.path.join(save_path, 'report'+time_format+'.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['thr', 'confm', 'auc', 'sensitivity_recall', 'specificity', 'precision', 'f1', 'accuracy'])
            writer.writerows(metrics)
            writer.writerow([''])
            writer.writerow(['studyid', 'numslices', 'groundtruth', 'prediction'])
            writer.writerows(info_list)
    else: #multi class
        metrics = getmetrics_multiclass(y_true=gt_test.detach().cpu().numpy(), y_pred=pred_test.detach().cpu().numpy()) #y_pred needs not to be binarized
        with open(os.path.join(save_path, 'report'+time_format+'.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['confm', 'auc'])
            writer.writerow([metrics[0], metrics[1]])
            header = ['']
            for i in range(output_size):
                header.append('class'+str(i))
            writer.writerow(header)
            writer.writerow(['sensitivity_recall'] + metrics[2])
            writer.writerow(['specificity'] + metrics[3])
            writer.writerow(['precision'] + metrics[4])
            writer.writerow(['accuracy'] + metrics[5])
            writer.writerow([''])
            header = ['studyid', 'numslices', 'groundtruth']
            for i in range(output_size):
                header.append('prediction'+str(i))
            writer.writerow(header)
            writer.writerows(info_list)
        

        

#Test if weights in the original paper of CT scores are used
def test_ctscores(dataloader_val, device, input_type, output_type, save_path):
    pred_list = []
    gt_list = []
    info_list = []
    for data_val in tqdm(dataloader_val, desc='report', total=len(dataloader_val), leave=True):
        input, gt = getinout(data_val, input_type, output_type, None, device)
        studyid, numslices = getdatainfo(data_val)
        if input_type == 'h' and output_type == 'outcome':
            pred = 1.0/(1.0 + math.exp(1.636-0.319*input.item()))
        elif input_type == 'h' and output_type == 'mortality':
            pred = 1.0/(1.0 + math.exp(2.666-0.287*input.item()))
        elif input_type == 'r' and output_type == 'mortality':
            pred = 1.0/(1.0 + math.exp(2.60-0.80*input.item()))
        else:
            raise NotImplementedError()
                        
        gt_list.append(gt.item())
        pred_list.append(pred)
        info_list.append([studyid, numslices, gt.item(), pred])
    metrics = getmetrics(y_true=np.array(gt_list), y_pred=np.array(pred_list)) #y_pred needs not to be binarized
    
    with open(os.path.join(save_path, 'report.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['thr', 'confm', 'auc', 'sensitivity_recall', 'specificity', 'precision', 'f1', 'accuracy'])
        writer.writerows(metrics)
        writer.writerow([''])
        writer.writerow(['studyid', 'numslices', 'groundtruth', 'prediction'])
        writer.writerows(info_list)

# Test by using values written in papers of Helsinki and Rotterdam scores
def test_existing_score():
    args = parameters.parser.parse_args()
    device = torch.device("cuda:0")
    dataset_all = dataset(input = 'v',
                        path_imaginginput=None,
                        summary_csv=args.summary_csv,
                        brainarea_csv=args.brainarea_csv,
                        feature_extracted_model=args.feature_extracted_model,
                        slice_len=args.slice_len,
                        targetimgsize = args.targetimgsize,
                        maxbrainarearatiothr=args.maxbrainarearatiothr)
    dataset_trainval = crossval(dataset_all=dataset_all, k_fold=args.k_fold)
    save_path = os.path.join(args.path_saveroot)
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    for k in range(args.k_fold):
        print(f'Cross validation ({k})')    
        dataset_val = dataset_trainval[k][1]
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False, num_workers=4, shuffle=False) 
        for input in ['r', 'h']:
            for output in ['outcome', 'mortality']:
                if not (input == 'r' and output == 'outcome'):
                    print(f'input:{input}, output:{output}')
                    save_path =  os.path.join(args.path_saveroot, input+'_'+output, 'value_from_paper', 'fold_'+str(k))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        test_ctscores(dataloader_val, device, input, output, save_path)
                

if __name__ == '__main__':
    test_existing_score()