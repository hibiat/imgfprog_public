import argparse
import os

parser = argparse.ArgumentParser(description='TBI prognosis using imaging features')
rootdir = '/mnt/c/work/tbidataset/center_tbi/'

#Usage of GPU cluster
parser.add_argument('--mist', type=int, default=0, help='1: run on GPU cluster like mist 0: run on premise')
#Hardware
parser.add_argument('--device', type=int, default=0, help='non-parallel, set device number if using specific GPU device. Set None otherwise')
#Input and output
parser.add_argument('--input', type=str, default='core_tffeature_vinflated768', help='v:clinical variables, \
                                                                i:image, iv: image+cv,\
                                                                f:imaging features, fv:cv+features, \
                                                                mv:Marshall+cv, rv:Rotterdam+cv: hv: Helsinki+cv,\
                                                                m:Marshall, r:Rotterdam, h:Helsinki')
parser.add_argument('--pred', type=str, default='home', help='"outcome" (favirable/unfavorable) or "mortality" or "gose" or "gose4cls" or "disposition_5cls" or "home" or "deadhomenohome"')
parser.add_argument('--jobidentifier', type=str, default=None, help='job identifier')
#Architecture
parser.add_argument('--arc', type=str, default='logistic_regression', help='architecture of network')
parser.add_argument('--feature_extracted_model', type=str, default='TimeSformer_divST_96x4_224_K600_norm_minus1_plus1_dim768', help='type of feature extractor') 
#Path to data
parser.add_argument('--path_imaginginput', type=str, default=rootdir+'04_npy_resize_orgimg', help='path to feature file (.npy) or image directory') 
parser.add_argument('--summary_csv', type=str, default=rootdir+'datasummary_all.csv', help='path to datasummary_all.csv')
parser.add_argument('--brainarea_csv', type=str, default=rootdir+'brainarea.csv', help='path to brainarea.csv')
parser.add_argument('--treatment_csv', type=str, default=None, help='path to csv describing surgical treatment')
parser.add_argument('--impact_csv', type=str, default=rootdir+'20221228IMPACT_stratum.csv', help='path to csv describing IMPACT inclusion criteria') 
parser.add_argument('--disposition_csv', type=str, default=rootdir+'20230615Disposition.csv', help='path to csv describing disposition') 
parser.add_argument('--exclude_dead', action='store_true') #if --exclude_dead is in the command, it will be treated as true and exclude dead cases in patient disposition. if --exclude_dead is not put in the command, dead cases will be included.
parser.add_argument('--path_saveroot', type=str, default='../results', help='path to saving directory')

#Pre-processing parameters
parser.add_argument('--slice_len', type=int, default=96, help='number of slices used in input') 
parser.add_argument('--targetimgsize', type=int, default=12, help='height and width of input image after resized') 
parser.add_argument('--maxbrainarearatiothr', type=float, default=0.15, help='threshod of max brain area ratio')

#Learning parameters
parser.add_argument('--epoch', type=int, default=300, help='number of epochs') 
parser.add_argument('--batch', type=int, default=32, help='batch size') 
parser.add_argument('--k_fold', type=int, default=5, help='number of fold for cross validation')
parser.add_argument('--foldrun', type=int, default=None, help='run specific fold. None otherwise')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate (default: 0.0001)')
parser.add_argument('--wd', type=float, default=0.0005,help='weight decay (default: 0.0005)')
parser.add_argument('--earlystopping_patience', type=int, default=300, help='Patience of early stopping')
parser.add_argument('--snapshot_interval', type=int, default=150, help='Interval of saving weights file') 
#Initial model weights (if applicable)
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
#Random seed
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
