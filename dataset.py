import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from utils.util_inout import checkfeaturesize, selectstudyid, labelmap_gose
import parameters

class dataset(Dataset):
    def __init__(self, input, path_imaginginput, summary_csv, brainarea_csv, treatment_csv, impact_csv,feature_extracted_model, slice_len, targetimgsize, maxbrainarearatiothr, disposition_csv, exclude_dead):
        self.input = input
        self.path_imaginginput = path_imaginginput
        self.summary_csv = summary_csv
        self.brainarea_csv = brainarea_csv
        self.treatment_csv = treatment_csv
        self.impact_csv = impact_csv
        self.feature_extracted_model = feature_extracted_model
        self.feature_dim = checkfeaturesize(feature_extracted_model)
        self.targetimgsize = targetimgsize
        self.slice_len = slice_len
        self.maxbrainarearatiothr = maxbrainarearatiothr
        self.disposition_csv = disposition_csv
        self.exclude_dead = exclude_dead

        self.df_summary, self.df_brain = selectstudyid(self.summary_csv, self.brainarea_csv, self.treatment_csv, self.impact_csv, self.maxbrainarearatiothr, self.disposition_csv, self.exclude_dead)
        self.df_summary = labelmap_gose(self.df_summary)

    def __getitem__(self, index):
        #General info
        studyid = str(self.df_summary['subjectId'][index])
        numslices = int(self.df_summary['NumSlices'][index])
        if self.input.startswith('f'):
            #CT features (Fixed size considering resolution)
            feature = np.zeros((self.slice_len, self.feature_dim), dtype=np.float32)
            image = -1
            filedir = os.path.join(self.path_imaginginput, self.feature_extracted_model, studyid, '*.npy')
            assert len(list(glob(filedir))) > 0, f'no npy file found in {filedir}'
            for count, featurefile in enumerate(sorted(glob(filedir))):
                f =  np.squeeze(np.load(featurefile))
                feature[count, :] = f
        elif self.input.startswith('i') or 'image' in self.input:
            #CT 3D images (Fixed size considering resolution)
            image = np.zeros((3, self.slice_len, self.targetimgsize, self.targetimgsize), dtype=np.float32)
            feature = -1
            filedir = os.path.join(self.path_imaginginput, studyid, '*.png')
            assert len(list(glob(filedir))) > 0, f'no image file found in {filedir}'
            for count, imgfile in enumerate(sorted(glob(filedir))):
                i =  Image.open(imgfile)
                if not hasattr(Image, 'Resampling'):  # Pillow<9.0
                    Image.Resampling = Image
                i_resized = i.resize((self.targetimgsize, self.targetimgsize), resample=Image.Resampling.BICUBIC)
                tmp = np.array(i_resized) #[targetimgsize (h),targetimgsize (w), 3]
                imgarray = np.moveaxis(tmp, 2, 0) #[3,targetimgsize (h), targetimgsize (w)]
                image[:, count, :, :] = imgarray/ 255.0 #[3, slice_len, targetimgsize (h), targetimgsize (w)]
        elif 'tffeature' in self.input:
            #Features extracted from 3D-TimeFormer
            feature = -1
            image = -1
            npyfilename = os.path.join(self.path_imaginginput, self.feature_extracted_model, studyid+'.npy')
            assert os.path.exists(npyfilename), f'{npyfilename} not found'
            tffeature = np.load(npyfilename)
        else:
            image = -1
            feature = -1
            tffeature = -1

        #Clinical variables
        age = self.df_summary['Subject.Age'][index]
        pupil = self.df_summary['InjuryHx.PupilsBaselineDerived'][index]
        gcs_motor = self.df_summary['InjuryHx.GCSMotorBaselineDerived'][index]
        gcs_score = self.df_summary['InjuryHx.GCSScoreBaselineDerived'][index]
        gose_6m = self.df_summary['Subject.GOSE6monthEndpointDerived'][index]
        gose4cls_6m = self.df_summary['gose_4cls'][index]
        marshall = self.df_summary['Imaging.MarshallCTClassification'][index]
        rotterdam = self.df_summary['Imaging.RotterdamCTScore'][index]
        helsinki = self.df_summary['Imaging.HelsinkiCTScore'][index]

        #Treatment process
        if self.treatment_csv is not None:
            if 's' in self.input:
                treatment = np.array(self.df_summary.loc[index, ['SurgeryDescCranial_'+str(i) for i in [2,7,5]] +['ExtraCranialSurgDone']], dtype=np.float64)
            else:
                treatment = np.array(self.df_summary.loc[index, ['SurgeryDescCranial_'+str(i) for i in range(1, 24)] +['SurgeryCranialDelay', 'ExtraCranialSurgDone']], dtype=np.float64)
                if np.sum(treatment) > 0:
                    treatment_bin = np.array(1.0, dtype=np.float64)
                else:
                    treatment_bin = np.array(0.0, dtype=np.float64)
        else:
            treatment = -1
            treatment_bin = -1

        #IMPACT 
        if self.impact_csv is not None:
            hypoxia = np.array(self.df_summary.loc[index, 'InjuryHx.EDComplEventHypoxia'], dtype=np.float64)
            hypotension = np.array(self.df_summary.loc[index, 'InjuryHx.EDComplEventHypotension'], dtype=np.float64)
            glucose = np.array(self.df_summary.loc[index, 'Labs.DLGlucosemmolL'], dtype=np.float64)
            hemoglobine = np.array(self.df_summary.loc[index, 'Labs.DLHemoglobingdL'], dtype=np.float64)
            tsah = np.array(self.df_summary.loc[index, 'tsah'], dtype=np.float64)
            edh = np.array(self.df_summary.loc[index, 'edh'], dtype=np.float64)
            impact_core = np.array(self.df_summary.loc[index, 'impact_core'], dtype=np.float64)
            impact_core_ct = np.array(self.df_summary.loc[index, 'impact_core_ct'], dtype=np.float64)
            impact_ct = np.array(self.df_summary.loc[index, 'impact_ct'], dtype=np.float64)
            impact_core_ct_lab = np.array(self.df_summary.loc[index, 'impact_core_ct_lab'], dtype=np.float64)
            impact_core_lab = np.array(self.df_summary.loc[index, 'impact_core_lab'], dtype=np.float64)
            
        else:
            hypoxia = -1
            hypotension = -1
            glucose = -1
            hemoglobine = -1
            tsah = -1
            edh = -1
            impact_core = -1
            impact_core_ct = -1
            impact_ct = -1
            impact_core_ct_lab = -1
            impact_core_lab = -1
            
        #prediction
        outcome = self.df_summary['Outcome'][index]
        mortality = self.df_summary['Mortality'][index]
        if self.disposition_csv is not None:
            disposition = self.df_summary['disposition'][index]
            home = self.df_summary['home'][index]
            deadhomenohome = self.df_summary['deadhomenohome'][index]
        else:
            disposition = -1
            home = -1
        
        return {'image':image, 'feature':feature, 'studyid':studyid, 'numslices':numslices,
                'age':age, 'pupil':pupil, 'gcs_motor':gcs_motor, 'gcs_score':gcs_score, 'gose_6m':gose_6m, 'gose4cls_6m':gose4cls_6m, 'marshall':marshall, 'rotterdam':rotterdam, 'helsinki':helsinki,
                'treatment':treatment,'treatment_bin':treatment_bin,
                'hypoxia': hypoxia, 'hypotension': hypotension, 'glucose': glucose, 'hemoglobine':hemoglobine,
                'tsah': tsah, 'edh': edh,
                'impact_core': impact_core, 'impact_core_ct': impact_core_ct, 'impact_ct':impact_ct, 'impact_core_ct_lab':impact_core_ct_lab, 'impact_core_lab': impact_core_lab,
                'tffeature': tffeature,
                'outcome':outcome, 'mortality':mortality, 'disposition': disposition, 'home':home, 'deadhomenohome':deadhomenohome} 
        #shape: feature: [batch, max_slicelen, feature dimension(2048etc)], numslices:[batch],...
        #       image: [batch, 3, slice_len, targetimgsize (h), targetimgsize (w)]
    def __len__(self):
        return len(self.df_summary)


if __name__ == "__main__":
    args = parameters.parser.parse_args()
    customdataset = dataset(input = 'core_tffeature', #'iv'
                            path_imaginginput='/mnt/c/work/tbidataset/center_tbi/04_npy_resize_maskimg'   ,#'/home/hibi/work/tbidataset/center_tbi/03_npy_feature_orgimg',
                            summary_csv='/mnt/c/work/tbidataset/center_tbi/datasummary_all.csv',
                            brainarea_csv='/mnt/c/work/tbidataset/center_tbi/brainarea.csv',
                            treatment_csv = None,
                            impact_csv='/mnt/c/work/tbidataset/center_tbi/20221228IMPACT_stratum.csv',
                            feature_extracted_model='TimeSformer_divST_96x4_224_K600_norm_minus1_plus1_dim768', #'vit_large_patch16_224_in21k'
                            slice_len = 96,
                            targetimgsize = 128,
                            maxbrainarearatiothr=0.15,
                            disposition_csv='/mnt/c/work/tbidataset/center_tbi/20230615Disposition.csv')

    customdataloader = DataLoader(dataset=customdataset, batch_size=11, pin_memory=False,
                              num_workers=5, shuffle=False)
    for batch in customdataloader:
        image = batch['image']
        feature = batch['feature']
        studyid = batch['studyid']
        numslices = batch['numslices']
        age = batch['age']
        gose_6m = batch['gose_6m']
        gose4cls_6m = batch['gose4cls_6m']
        helsinki = batch['helsinki']
        treatment = batch['treatment']
        treatment_bin = batch['treatment_bin']
        hypoxia = batch['hypoxia']
        hypotension = batch['hypotension']
        glucose = batch['glucose']
        hemoglobine = batch['hemoglobine']
        outcome = batch['outcome']
        mortality = batch['mortality']
        disposition = batch['disposition']
        home = batch['home']
        deadhomenohome = batch['deadhomenohome']
        tsah = batch['tsah']
        edh = batch['edh']
        impact_core = batch['impact_core']
        impact_core_ct = batch['impact_core_ct']
        impact_ct = batch['impact_ct']
        impact_core_ct_lab = batch['impact_core_ct_lab']
        impact_core_lab = batch['impact_core_lab']
        tffeature = batch['tffeature']

        print('Study ID:{}'.format(studyid))
        print('Number of Slices:{}'.format(numslices))
        print('Shape of image ({})'.format(image.shape)) #[batch, 3, slice_len, targetimgsize (h), targetimgsize (w)]
        print('Shape of feature ({})'.format(feature.shape)) #[batch_size, slicelen, feature dimenstion]
        print('Shape of treatment ({})'.format(treatment.shape)) #[batch_size, slicelen, feature dimenstion]
        print('Shape of treatment_bin ({})'.format(treatment_bin.shape)) #[batch_size]
        print('Shape of hypoxia ({})'.format(hypoxia.shape)) #[batch_size]
        print('Shape of hypotension ({})'.format(hypotension.shape)) #[batch_size]
        print('hypotension ({})'.format(hypotension)) 
        print('Shape of glucose ({})'.format(glucose.shape)) #[batch_size]
        print('glucose ({})'.format(glucose)) 
        print('Shape of hemoglobine ({})'.format(hemoglobine.shape)) #[batch_size]
        print('Shape of tSAH ({})'.format(tsah.shape)) #[batch_size]
        print('tSAH ({})'.format(tsah)) 
        print('Shape of EDH ({})'.format(edh.shape)) #[batch_size]
        print('Shape of impact_core ({})'.format(impact_core.shape)) #[batch_size]
        print('impact_core ({})'.format(impact_core)) 
        print('Shape of impact_core_ct ({})'.format(impact_core_ct.shape)) #[batch_size]
        print('impact_core_ct ({})'.format(impact_core_ct)) 
        print('Shape of impact_ct ({})'.format(impact_ct.shape)) #[batch_size]
        print('impact_ct ({})'.format(impact_ct)) 
        print('Shape of impact_core_ct_lab ({})'.format(impact_core_ct_lab.shape)) #[batch_size]
        print('impact_core_ct_lab ({})'.format(impact_core_ct_lab)) 
        print('Shape of impact_core_lab ({})'.format(impact_core_lab.shape)) #[batch_size]
        print('impact_core_lab ({})'.format(impact_core_lab)) 
        print('Shape of tffeature ({})'.format(tffeature.shape)) #[batch_size, tffeature demenction]
        
        
        
        print(f'Age: {age}')
        print(f'GOSe 6month: {gose_6m}')
        print(f'GOS4cls 6month: {gose4cls_6m}')
        print(f'Helsinki score: {helsinki}')
        print(f'Favorable/unfavorable outcome 6month: {outcome}')
        print(f'Mortality 6month: {mortality}')
        print(f'Disposition: {disposition}')
        print(f'Home: {home}')
        print(f'Dead-Home-Nohome: {deadhomenohome}')
        print('------')
        