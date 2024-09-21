import random
import numpy as np
np.seterr(divide='ignore')
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from torch.utils.data.dataset import Subset


def crossvalidx(total_size, k_fold, random_seed=None):
    train_idx_dict =dict()
    val_idx_dict = dict()
    if random_seed is not None:
        random.seed(random_seed)
        shuffle_idx = random.sample(list(range(total_size)), total_size)
        print(f'Random shuffle seed {random_seed}') 
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        print('Cross validation({}) train indices: [{},{}),[{},{}), test indices: [{},{})'.format(i, trll,trlr,trrl,trrr,vall,valr))
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        if random_seed is not None:
            train_idx_s = []
            val_idx_s = []
            for j in train_indices:
                train_idx_s.append(shuffle_idx[j])
            for j in val_indices:
                val_idx_s.append(shuffle_idx[j])
            train_idx_dict[i] = train_idx_s
            val_idx_dict[i] = val_idx_s
        else:
            train_idx_dict[i] = train_indices
            val_idx_dict[i] = val_indices
    return train_idx_dict, val_idx_dict

def crossval(dataset_all, k_fold, random_seed=None):
    output = []
    total_size = len(dataset_all)
    train_idx_dict, val_idx_dict = crossvalidx(total_size, k_fold, random_seed)
    for i in range(k_fold):
        train_set = Subset(dataset_all, train_idx_dict[i])
        val_set = Subset(dataset_all, val_idx_dict[i])
        output.append([train_set, val_set])
    return output

def getmetrics(y_true, y_pred, detailed=False): #numpy array (not list)
    if np.sum(y_true < 0.5) == len(y_true) or np.sum(y_true > 0.5) == len(y_true): # auc cannot be computed in a single class 
        auc = None
    else:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    
    metrics = []
    for thr in np.arange(0.01, 1, 0.01):
        bin_pred = np.where(y_pred > thr, 1, 0) 
        confm = confusion_matrix(y_true=y_true, y_pred=bin_pred, labels=[0,1])
        tn, fp, fn, tp = confm.flatten()
        if tp + fn > 0:
            sensitivity_recall = tp / (tp + fn)
        else:
            sensitivity_recall = None

        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = None
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = None
        
        if tp + fp + fn > 0:
            f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            f1 = None
        accuracy = (tp + tn)/ (tn + fp + fn + tp)
        if detailed:
            eer = tp/ (tp + fn) # Experimental event rate = precision
            cer = fn / (fn + tn) #Control event rate
            arr = abs(cer - eer) #Absolute risk reduction
            nnt = 1 / arr #Number needed to treat
            metrics.append([thr, confm, auc, sensitivity_recall, specificity, precision, f1, accuracy, cer, arr, nnt])
        else:
            metrics.append([thr, confm, auc, sensitivity_recall, specificity, precision, f1, accuracy])
    return metrics

def getmetrics_multiclass(y_true, y_pred): #numpy array (not list)
    auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr')
    y_pred_amax = np.argmax(y_pred, axis=1)
    confm = confusion_matrix(y_true=y_true, y_pred=y_pred_amax)
    #https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    FP = confm.sum(axis=0) - np.diag(confm) 
    FN = confm.sum(axis=1) - np.diag(confm)
    TP = np.diag(confm)
    TN = confm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)

    sensitivity_recall = TPR
    specificity = TNR
    precision = PPV
    accuracy = ACC

    metrics = [confm.tolist(), auc.tolist(), sensitivity_recall.tolist(), specificity.tolist(), precision.tolist(), accuracy.tolist()]
    return metrics

def getCDBmetric_multiclass(y_true, y_pred): #numpy array (not list) for Class-Wise Difficulty-Balanced Loss
    y_pred_amax = np.argmax(y_pred, axis=1)
    confm = confusion_matrix(y_true=y_true, y_pred=y_pred_amax)
    #https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    FP = confm.sum(axis=0) - np.diag(confm) 
    FN = confm.sum(axis=1) - np.diag(confm)
    TP = np.diag(confm)
    TN = confm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Overall accuracy for each class
    accuracy = (TP+TN)/(TP+FP+FN+TN)

    sensitivity = TP/(TP+FN) 
    specificity = TN/(TN+FP) 
    ave_senspe = np.nan_to_num(0.5 * (sensitivity + specificity)) #remove nan

    #return accuracy.tolist()
    return ave_senspe

def import_part(model_now, dict_pretrained): #pretrainの重みのうち、今回のネットワーク構造に一致する部分の重みのみインポート
    dict_now = model_now.state_dict()

    # 1. filter out unnecessary keys
    for k in dict_pretrained.keys():
        if not k in dict_now:
            print(f'{k} in pretrained model is not loaded')
    for k in dict_now.keys():
        if not k in dict_pretrained:
            print(f'{k} in current model is not updated')
    dict_pretrained = {k: v for k, v in dict_pretrained.items() if k in dict_now}
    # 2. overwrite entries in the existing state dict
    dict_now.update(dict_pretrained) 
    # 3. load the new state dict
    model_now.load_state_dict(dict_now)

    return model_now

def import_part_pretrain(model_now, model_pretrained): 
    dict_pretrained = model_pretrained.state_dict()
    model_updated = import_part(model_now, dict_pretrained)
    return model_updated

def import_part_pretrain_pth(model_now, model_pretrained_pth):
    dict_pretrained = torch.load(model_pretrained_pth)
    model_updated = import_part(model_now, dict_pretrained)
    return model_updated

def import_part_pretrain_pth2(model_now, model_pretrained_pth):
    dict_pretrained = torch.load(model_pretrained_pth)
    dict_pretrained = dict_pretrained['model_state']
    model_updated = import_part(model_now, dict_pretrained)
    return model_updated

def sigmoid(x):
  return (1/(1+np.exp(-x)))

class CDB_loss(torch.nn.Module):
    def __init__(self, class_difficulty, tau='dynamic', reduction='mean'): #tau='dynamic'
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        print(f'loss weights updated:{self.weights}')
        self.reduction = reduction
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda() #Make sure to use arguments of "--device=0" 
    def forward(self, input, target):
        return self.loss(input, target)
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True