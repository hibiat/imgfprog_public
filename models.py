import os
import torch
import torch.nn as nn
from torchvision import models

from utils.util_traintest import import_part_pretrain, import_part_pretrain_pth, import_part_pretrain_pth2

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    def forward(self, x):
        #outputs = torch.sigmoid(self.linear(x))
        outputs = self.linear(x)
        return outputs

class LogisticRegressionWithMiddleLayer384(torch.nn.Module):
    def __init__(self, input_size, in_v_len, output_size):
        super(LogisticRegressionWithMiddleLayer384, self).__init__()
        middle_layer_dim = 384
        self.linear1 = torch.nn.Linear(input_size, middle_layer_dim)
        self.linear2 = torch.nn.Linear(middle_layer_dim + in_v_len, output_size)
    def forward(self, x1, x2):
        y = self.linear1(x1)
        outputs = self.linear2(torch.cat((y, x2), dim=1))
        return outputs

class LogisticRegressionWithMiddleLayer192(torch.nn.Module):
    def __init__(self, input_size, in_v_len, output_size):
        super(LogisticRegressionWithMiddleLayer192, self).__init__()
        middle_layer_dim = 192
        self.linear1 = torch.nn.Linear(input_size, middle_layer_dim)
        self.linear2 = torch.nn.Linear(middle_layer_dim + in_v_len, output_size)
    def forward(self, x1, x2):
        y = self.linear1(x1)
        outputs = self.linear2(torch.cat((y, x2), dim=1))
        return outputs

class LogisticRegressionWithMiddleLayer96(torch.nn.Module):
    def __init__(self, input_size, in_v_len, output_size):
        super(LogisticRegressionWithMiddleLayer96, self).__init__()
        middle_layer_dim = 96
        self.linear1 = torch.nn.Linear(input_size, middle_layer_dim)
        self.linear2 = torch.nn.Linear(middle_layer_dim + in_v_len, output_size)
    def forward(self, x1, x2):
        y = self.linear1(x1)
        outputs = self.linear2(torch.cat((y, x2), dim=1))
        return outputs

class LogisticRegressionWithMiddleLayer48(torch.nn.Module):
    def __init__(self, input_size, in_v_len, output_size):
        super(LogisticRegressionWithMiddleLayer48, self).__init__()
        middle_layer_dim = 48
        self.linear1 = torch.nn.Linear(input_size, middle_layer_dim)
        self.linear2 = torch.nn.Linear(middle_layer_dim + in_v_len, output_size)
    def forward(self, x1, x2):
        y = self.linear1(x1)
        outputs = self.linear2(torch.cat((y, x2), dim=1))
        return outputs

def model_selector(model_name, input, input_size, output_size):
    print(f'Model: {model_name}')
    if model_name == 'logistic_regression':
        model = LogisticRegression(input_size=input_size, output_size=output_size)
    elif model_name == 'logistic_regression_with_middle_layer384':
        model = LogisticRegressionWithMiddleLayer384(input_size=input_size[0], in_v_len=input_size[1], output_size=output_size)
    elif model_name == 'logistic_regression_with_middle_layer192':
        model = LogisticRegressionWithMiddleLayer192(input_size=input_size[0], in_v_len=input_size[1], output_size=output_size)
    elif model_name == 'logistic_regression_with_middle_layer96':
        model = LogisticRegressionWithMiddleLayer96(input_size=input_size[0], in_v_len=input_size[1], output_size=output_size)
    elif model_name == 'logistic_regression_with_middle_layer48':
        model = LogisticRegressionWithMiddleLayer48(input_size=input_size[0], in_v_len=input_size[1], output_size=output_size)

    else:
        assert type(input_size) == list, f'{model_name} only accepts list type input size, but its type is {type(input_size)}'
        homedir = os.environ['HOME']
        if len(input_size) == 3:
            in_img_h, in_img_w, in_v_len = input_size
        elif len(input_size) == 4:
            in_img_h, in_img_w, in_img_d, in_v_len = input_size
        else:
            raise NotImplementedError(model_name +' has not been implemented yet...')

    return model

if __name__ == '__main__':
    feature_dim = 768
    slice_len = 32
    in_v_len = 1
    output_size= 1
    batch = 5
    imgsize = 96
       
    v = torch.randn(batch, in_v_len)
    tffeature = torch.randn(batch, feature_dim)
    model = model_selector(model_name='logistic_regression_with_middle_layer192',  input='core_tffeaturemid', input_size=[feature_dim, in_v_len], output_size=output_size)
    pred = model(tffeature, v)

    print(model)
    print(pred)
