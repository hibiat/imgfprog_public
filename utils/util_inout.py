import torch
import numpy as np
import pandas as pd
import os
from glob import glob
import json
import math

#Select studyid based on the existence of surgical treatment variables
def selectstudyid_treatment(df_summary_s, treatment_csv):
    pd.set_option('mode.chained_assignment', None)
    df = df_summary_s
    df_treat = pd.read_csv(treatment_csv)
    df.loc[:, 'treatment_valid'] = 0
    df.loc[:, ['SurgeryDescCranial_'+str(i) for i in range(1, 24)]] = 0
    df.loc[:, 'SurgeryCranialDelay'] = 0 #1 if there was a delay
    df.loc[:, 'ExtraCranialSurgDone'] = 0
    
    for j in range(len(df)):
        subjectid = df.at[j, 'subjectId']
        if sum(df_treat['subjectId'].isin([subjectid])) > 0:
            q = 'subjectId=="'+subjectid+'"'
            idx_list = df_treat.query(q).index
            if len(idx_list) > 0:
                if not df_treat.loc[idx_list]['Surgeries.CranialSurgDone'].isnull().any() and not df_treat.loc[idx_list]['Surgeries.ExtraCranialSurgDone'].isnull().any():
                    if sum(df_treat.loc[idx_list]['Surgeries.CranialSurgDone']==88) == 0 and sum(df_treat.loc[idx_list]['Surgeries.ExtraCranialSurgDone']==88) == 0:
                        df.at[j, 'treatment_valid'] = 1
                        df.at[j, 'ExtraCranialSurgDone'] = max(df_treat.loc[idx_list, 'Surgeries.ExtraCranialSurgDone'])
                        delay_tmp = 0
                        for d in idx_list:
                            delay = df_treat.at[d, 'SurgeriesCranial.SurgeryCranialDelay']
                            if delay == 1 or delay == 2 or delay == 3 or delay == 4 or delay == 99:
                                delay_tmp = 1
                                break
                        df.at[j, 'SurgeryCranialDelay'] = delay_tmp
                        for i in idx_list:
                            if not np.isnan(df_treat.at[i, 'SurgeriesCranial.SurgeryDescCranial']):
                                desc = int(df_treat.at[i, 'SurgeriesCranial.SurgeryDescCranial'])
                                if desc == 71:
                                    desc = 22
                                elif desc == 72:
                                    desc = 23
                                assert desc >= 1 and desc <= 23, f'desc is expected to be 1-23, but is {desc}'
                                df.at[j, 'SurgeryDescCranial_'+str(desc)] = 1
                            
    df_out = df[df['treatment_valid'] > 0]
    df_out.reset_index(drop=True, inplace=True)
    return df_out

# Select Study ID based on IMPACT eligibility 
# (1)Existence of Hypoxia measurement (2)Existence of Hypotension measurement (3)"Initial" (not second or later) glucose and hemoglobin
def selectstudyid_impact(df_summary_s, impact_csv):
    pd.set_option('mode.chained_assignment', None)
    df = df_summary_s
    df_impact = pd.read_csv(impact_csv)
    #Initialization for added columns
    df.loc[:, 'impact_valid'] = 0 #0: invalid 1: valid
    df.loc[:, 'Subject.PatientType'] = 0 #1:ER, 2:Admission 3:ICU
    df.loc[:, 'InjuryHx.EDComplEventHypoxia'] = 0 # 0:No, 1,2=Yes
    df.loc[:, 'InjuryHx.EDComplEventHypotension'] = 0 #0:No, 1,2=Yes
    df.loc[:, 'Labs.DLGlucosemmolL'] = 0 
    df.loc[:, 'Labs.DLHemoglobingdL'] = 0 

    for j in range(len(df)):
        subjectid = df.at[j, 'subjectId']
        if sum(df_impact['subjectId'].isin([subjectid])) > 0:
            df_t = df_impact[df_impact['subjectId']==subjectid]
            #remove rows with "nan"
            df_temp = df_t.dropna(subset=['InjuryHx.EDComplEventHypoxia', 'InjuryHx.EDComplEventHypotension', 'Labs.DLDate', 'Labs.DLTime', 'Labs.DLGlucosemmolL', 'Labs.DLHemoglobingdL'])
            #sort by date and time
            if len(df_temp) > 0:
                df_temp['Labs.DLTime_rev']=df_temp['Labs.DLTime'].str.replace(':','').astype(int)
                df_subject = df_temp.sort_values(['Labs.DLDate','Labs.DLTime_rev'])
                df_subject.reset_index(inplace=True, drop=True)
                for k in range(len(df_subject)):
                    if (df_subject.loc[k]['InjuryHx.EDComplEventHypoxia'].astype(int) in [0, 1, 2] and
                        df_subject.loc[k]['InjuryHx.EDComplEventHypotension'].astype(int) in [0, 1, 2]):
                        df.at[j, 'impact_valid'] = 1
                        df.at[j, 'Subject.PatientType'] = df_subject.loc[k]['Subject.PatientType']
                        df.at[j, 'InjuryHx.EDComplEventHypoxia'] = df_subject.loc[k]['InjuryHx.EDComplEventHypoxia'] 
                        df.at[j, 'InjuryHx.EDComplEventHypotension'] = df_subject.loc[k]['InjuryHx.EDComplEventHypotension']
                        df.at[j, 'Labs.DLGlucosemmolL'] = df_subject.loc[k]['Labs.DLGlucosemmolL']
                        df.at[j, 'Labs.DLHemoglobingdL'] = df_subject.loc[k]['Labs.DLHemoglobingdL']
                        break
 
    df_out = df[df['impact_valid'] > 0]
    df_out.reset_index(drop=True, inplace=True)
    return df_out

#Age>=14
def selectstudyid_adult(df_summary_ss):
    df_out = df_summary_ss[df_summary_ss['Subject.Age']>=14]
    df_out.reset_index(drop=True, inplace=True)
    return df_out
    
#Add columns of tSAH(Traumatic subarachnoid hemorrhage) and EDH (epidural hematoma)
def add_tsah_edh(df_summary_sss):
    pd.set_option('mode.chained_assignment', None)
    #Initialization for added columns
    df_summary_sss.loc[:, 'tsah'] = 0 
    df_summary_sss.loc[:, 'edh'] = 0 
    for i in range(len(df_summary_sss)):
        lesion_dict = json.loads(df_summary_sss.loc[i]['Imaging.LesionData']) #json string => dict
        dict_keys = list(lesion_dict.keys())
        if 'tsah' in dict_keys:
            df_summary_sss.at[i, 'tsah'] = 1
        if 'epidural_hematoma' in dict_keys:
            df_summary_sss.at[i, 'edh'] = 1

    df_summary_sss.reset_index(drop=True, inplace=True)
    return df_summary_sss

 # Add columns of impact scores
def add_impact_scores(df_summary):
    pd.set_option('mode.chained_assignment', None)
    #Initialization for added columns
    df_summary.loc[:, 'impact_core'] = 0 
    df_summary.loc[:, 'impact_core_ct'] = 0 
    df_summary.loc[:, 'impact_core_ct_lab'] = 0 
    df_summary.loc[:, 'impact_core_lab'] = 0 

    for index in range(len(df_summary)):
        age = np.array(df_summary['Subject.Age'][index], dtype=np.float64)
        gcs_motor = np.array(df_summary['InjuryHx.GCSMotorBaselineDerived'][index], dtype=np.float64)
        pupil = np.array(df_summary['InjuryHx.PupilsBaselineDerived'][index], dtype=np.float64)
        marshall = np.array(df_summary['Imaging.MarshallCTClassification'][index], dtype=np.float64)

        # core
        if age < 30:
            age_core = 0
        elif age >= 30 and age <= 39:
            age_core = 1
        elif age >= 40 and age <= 49:
            age_core = 2
        elif age >= 50 and age <= 59:
            age_core = 3
        elif age >= 60 and age <= 69:
            age_core = 4
        elif age >=70:
            age_core = 5

        if gcs_motor == 1 or gcs_motor == 2:
            gcs_motor_core = 6
        elif gcs_motor == 3:
            gcs_motor_core = 4
        elif gcs_motor == 4:
            gcs_motor_core = 2
        elif gcs_motor == 5 or gcs_motor == 6:
            gcs_motor_core = 0
    
        if pupil == 0:
            pupil_core = 0
        elif pupil == 1:
            pupil_core = 2
        elif pupil == 2:
            pupil_core = 4

        impact_core = age_core + gcs_motor_core + pupil_core
        # CT
        hypoxia = np.array(df_summary.loc[index, 'InjuryHx.EDComplEventHypoxia'], dtype=np.float64)
        hypotension = np.array(df_summary.loc[index, 'InjuryHx.EDComplEventHypotension'], dtype=np.float64)
        tsah = np.array(df_summary.loc[index, 'tsah'], dtype=np.float64)
        edh = np.array(df_summary.loc[index, 'edh'], dtype=np.float64)

        hypoxia_core = 1 if hypoxia > 0 else 0
        hypotension_core = 2 if hypotension > 0 else 0
        if marshall == 1:
            marshall_core = -2
        elif marshall == 2:
            marshall_core = 0
        elif marshall >= 3:
            marshall_core = 2
        tsah_core = 2 if tsah > 0 else 0
        edh_core = -2 if edh > 0 else 0
        impact_ct = hypoxia_core + hypotension_core + marshall_core + tsah_core + edh_core
        impact_core_ct = impact_core + impact_ct
        
        # Lab 
        glucose = np.array(df_summary.loc[index, 'Labs.DLGlucosemmolL'], dtype=np.float64)
        hemoglobine = np.array(df_summary.loc[index, 'Labs.DLHemoglobingdL'], dtype=np.float64)

        if glucose < 6:
            glucose_core = 0
        elif glucose >= 6 and glucose < 9:
            glucose_core = 1
        elif glucose >= 9 and glucose < 12:
            glucose_core = 2
        elif glucose >= 12 and glucose < 15:
            glucose_core = 3
        elif glucose >= 15:
            glucose_core = 4
        
        if hemoglobine < 9:
            hemoglobine_core = 3
        elif hemoglobine >=9 and hemoglobine < 12:
            hemoglobine_core  = 2
        elif hemoglobine >= 12 and hemoglobine < 15:
            hemoglobine_core = 1
        elif hemoglobine >= 15:
            hemoglobine_core = 0
            
        impact_core_ct_lab = impact_core_ct + glucose_core + hemoglobine_core
        impact_core_lab = impact_core + glucose_core + hemoglobine_core

        df_summary.at[index, 'impact_core'] = impact_core
        df_summary.at[index, 'impact_ct'] = impact_ct
        df_summary.at[index, 'impact_core_ct'] = impact_core_ct
        df_summary.at[index, 'impact_core_ct_lab'] = impact_core_ct_lab
        df_summary.at[index, 'impact_core_lab'] = impact_core_lab

    df_summary.reset_index(drop=True, inplace=True)
    return df_summary

#Add columns of surgical intervention
def selectsurgicalintervention(df_summary, surgical_csv):
    pd.set_option('mode.chained_assignment', None)
    df_summary.loc[:, 'surgicalintervention'] = -1 
    #-1: Unknown 
    #0: No intracranial procedure (InjuryHx.EmergSurgInterventionsIntraCran == blank) 
    #1: Craniotomy (InjuryHx.EmergSurgInterventionsIntraCran == 1)
    #2: Craniectomy (InjuryHx.EmergSurgInterventionsIntraCran == 2)
    #3: Ohter intracranial procedure (InjuryHx.EmergSurgInterventionsIntraCran == 3 or 99)
    df_surgical = pd.read_csv(surgical_csv)
    for j in range(len(df_summary)):
        subjectid = df_summary.at[j, 'subjectId']
        if sum(df_surgical['subjectId'].isin([subjectid])) > 0:
            q = 'subjectId=="'+subjectid+'"'
            idx_list = df_surgical.query(q).index
            if len(idx_list) > 0:
                assert len(idx_list) == 1, 'there are overlapping studyId'
                surgint_yesno = df_surgical.loc[idx_list[0]]['InjuryHx.EmergSurgInterventionsIntraCran']
                surgint = df_surgical.loc[idx_list[0]]['InjuryHx.EmergSurgInterventionsIntraCranYes']
                if surgint_yesno == 0:
                    df_summary.at[j, 'surgicalintervention'] = 0
                elif surgint_yesno == 1 and surgint == 1:
                    df_summary.at[j, 'surgicalintervention'] = 1
                elif surgint_yesno == 1 and surgint == 2:
                    df_summary.at[j, 'surgicalintervention'] = 2
                elif surgint_yesno == 1 and surgint == 3:
                    df_summary.at[j, 'surgicalintervention'] = 3
                elif surgint_yesno == 1 and surgint == 99:
                    df_summary.at[j, 'surgicalintervention'] = 3
                else:
                    df_summary.at[j, 'surgicalintervention'] = -1

    df_out = df_summary[df_summary['surgicalintervention'] >= 0]
    df_out.reset_index(drop=True, inplace=True)
    return df_out
         
#Add columns of disposition
def selectdisposition(df_summary, disposition_csv, exclude_dead=False):
    pd.set_option('mode.chained_assignment', None)
    df_summary.loc[:, 'disposition'] = -1 
    #-1) Unknown 
    # 0) Dead (Hospital.DischargeStatus == 0)
    # 1) Other hospital (Hopital.DispHosp == 1)
    # 2) Rehab unit (Hopital.DispHosp == 2)
    # 3) Nursing home (Hopital.DispHosp == 3)
    # 4) Home (Hopital.DispHosp == 4)
    df_disp = pd.read_csv(disposition_csv)
    for j in range(len(df_summary)):
        subjectid = df_summary.at[j, 'subjectId']
        if sum(df_disp['subjectId'].isin([subjectid])) > 0:
            q = 'subjectId=="'+subjectid+'"'
            idx_list = df_disp.query(q).index
            if len(idx_list) > 0:
                assert len(idx_list) == 1, 'there are overlapping studyId'
                alivedead = df_disp.loc[idx_list[0]]['Hospital.DischargeStatus'] #dead = 0, alive=1, unknown=88 or NaN(blank)
                dispwhere = df_disp.loc[idx_list[0]]['Hospital.DispHosp'] #other hospital=1, rehab unit=2, nursing home=3, home=5, unknown=88, other=99 or Nan (blank)
                if alivedead == 0:
                    df_summary.at[j, 'disposition'] = 0 #dead
                elif alivedead == 1 and dispwhere == 1:
                    df_summary.at[j, 'disposition'] = 1 #other hospital
                elif alivedead == 1 and dispwhere == 2:
                    df_summary.at[j, 'disposition'] = 2 #rehab unit
                elif alivedead == 1 and dispwhere == 3:
                    df_summary.at[j, 'disposition'] = 3 #nursing home
                elif alivedead == 1 and dispwhere == 5:
                    df_summary.at[j, 'disposition'] = 4 #home
                else:
                    df_summary.at[j, 'disposition'] = -1

    df_summary['home'] = np.where(df_summary['disposition']==4, 1, 0) #home=1, non-home=0

    def dead_home_nohome(row):
        if row['disposition'] == 0: #dead
            return 0
        if row['disposition'] == 4: #alive-home
            return 1
        if row['disposition'] == 1 or row['disposition'] == 2 or row['disposition'] == 3: #alive-no-home
            return 2
        if row['disposition'] == -1:
            return -1
        
    df_summary['deadhomenohome'] = df_summary.apply(dead_home_nohome, axis=1)  #dead=0, alive-home=1, alive-non-home=2
    
    if exclude_dead:
        df_out = df_summary[df_summary['disposition'] > 0]
        print('Dead cases exluded')
    else:
        df_out = df_summary[df_summary['disposition'] >= 0]
    df_out.reset_index(drop=True, inplace=True)
    return df_out

#Select studyid based on 'MaxBrainAreaRatio' > maxbrainarearatiothr
def selectstudyid(summary_csv, brainarea_csv, treatment_csv, impact_csv, maxbrainarearatiothr, disposition_csv=None, exclude_dead=False):
    df_brain = pd.read_csv(brainarea_csv)
    print('{} subjects in the original dataset'.format(len(df_brain)))
    df_brain_s = df_brain[df_brain['MaxBrainAreaRatio'] > maxbrainarearatiothr]
    df_brain_s.reset_index(drop=True, inplace=True)
    print('{} subjects are selected based on maximum brain area'.format(len(df_brain_s)))
    df_summary = pd.read_csv(summary_csv)
    df_summary_s = df_summary[df_brain['MaxBrainAreaRatio'] > maxbrainarearatiothr]
    df_summary_s.reset_index(drop=True, inplace=True)
    if treatment_csv is not None and impact_csv is None:
        df_summary_ss = selectstudyid_treatment(df_summary_s, treatment_csv)    
        print('{} subjects are selected based on surgical treatment process'.format(len(df_summary_ss)))
        return df_summary_ss, df_brain_s
    elif treatment_csv is None and impact_csv is not None:
        df_summary_ss = selectstudyid_impact(df_summary_s, impact_csv)    
        print('{} subjects are selected based on IMPACT eligiblity'.format(len(df_summary_ss)))
        df_summary_sss = selectstudyid_adult(df_summary_ss)    
        print('{} subjects are selected based on Age>=14'.format(len(df_summary_sss)))
        df_summary_ct = add_tsah_edh(df_summary_sss) #Add columns of tSAH and EDH
        df_summary_out = add_impact_scores(df_summary_ct) # Add columns of impact scores
        if disposition_csv is not None:
            df_summary_out = selectdisposition(df_summary_out, disposition_csv, exclude_dead)
            print('{} subjects are selected based on disposition'.format(len(df_summary_out)))
        return df_summary_out, df_brain_s
    elif treatment_csv is None and impact_csv is None:
        return df_summary_s, df_brain_s
    else:
        raise NotImplementedError('Error in path to treatment and/or impact csv')

#Label mapping of GOSe 6month => dichotomization for outcome (favorable[GOESE:5-8]/unfavorable[GOSE:1-4]) and mortality (death[GOESE:1]/not death)
def labelmap_gose(df_summary):
    df_summary.replace({'Subject.GOSE6monthEndpointDerived': {'1':0, '2_or_3':1, '4':2, '5':3, '6':4, '7':5, '8':6}}, inplace=True)
    df_summary['Subject.GOSE6monthEndpointDerived'] = pd.to_numeric(df_summary['Subject.GOSE6monthEndpointDerived'])
    gose = np.array(df_summary['Subject.GOSE6monthEndpointDerived'])
    df_summary['gose_4cls'] = df_summary.replace({'Subject.GOSE6monthEndpointDerived': {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3}},inplace=False)['Subject.GOSE6monthEndpointDerived'] #GOSe1:class 0, GOSe2-4:class 1, GOSe5-6= class 2, GOSe7-8= class 3   
    outcome = np.where(gose<=2, 1, 0)
    mortality = np.where(gose==0, 1, 0)
    df_summary['Outcome'] = outcome
    df_summary['Mortality'] = mortality
    return df_summary

# Converting one-hot vector into discributed labels based on Label dictribution learning
def labeldist(gt, classnum, device, sigma=0.5):
    batch_size = len(gt)
    x = torch.arange(0, classnum).repeat(batch_size, 1).to(device)
    n = torch.exp(-torch.square(x - gt.unsqueeze(1)) / (2 * sigma**2))
    d = torch.sqrt(torch.tensor(2 * math.pi * sigma**2))
    return n / d


def getthickness(nii_dir, studyid):
    import nibabel as nib
    niifiles = glob(os.path.join(nii_dir, studyid, '*nii.gz'))
    if len(list(niifiles)) > 1:
        maxsize = 0
        for f in niifiles:
            fsize = os.path.getsize(f)
            if fsize > maxsize:
                niifile = f
                maxsize = fsize
    else:
        niifile = list(niifiles)[0]
    
    niidata = nib.load(niifile)
    imgdim = niidata.header['dim']
    imgdim_sortidx = np.argsort(imgdim)
    thickness_axis = imgdim_sortidx[5]
    assert imgdim[thickness_axis] != 1, f'imgdim[thickness_axis] = {imgdim[thickness_axis]}'
    thickness = niidata.header['pixdim'][thickness_axis]
    assert thickness > 0, f'thickness is not positive ({thickness})'
    return thickness

def checkfeaturesize(feature_extracted_model):
    if feature_extracted_model == 'resnet50':
        featuresize = 2048
    elif feature_extracted_model == 'vit_large_patch16_224_in21k':
        featuresize = 1024
    elif feature_extracted_model == 'TimeSformer_divST_96x4_224_K600_norm_minus1_plus1_dim768':
        featuresize = 768
    else:
        raise NotImplementedError('{} model has not been implremented yet'.format(feature_extracted_model))
    return featuresize

def getinputsize(args):
    #IMPACT-based
    if args.input == 'core' or args.input == 'core_ct' or args.input == 'ct' or args.input == 'core_ct_lab' or args.input == 'core_lab' or args.input == 'm' or args.input == 'r' or args.input == 'h':
        input_size = 1
    elif args.input == 'core_m' or args.input == 'core_r' or args.input == 'core_h' or args.input == 'core_m_lab' or args.input == 'core_r_lab' or args.input == 'core_h_lab':
        input_size = 2
    elif args.input == 'image':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 0]
    elif args.input == 'core_image' or args.input == 'core_image_lab':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 1] 
    elif args.input =='tffeature':
        input_size = checkfeaturesize(args.feature_extracted_model)
    elif args.input =='core_tffeature' or args.input =='core_tffeature_lab':
        input_size = checkfeaturesize(args.feature_extracted_model) + 1
    elif args.input =='core_tffeaturemid' or args.input =='core_tffeaturemid_lab':
        input_size = [checkfeaturesize(args.feature_extracted_model), 1]
    elif args.input.startswith('core_tffeature_vinflated'):
        infratio = int(args.input.split('core_tffeature_vinflated')[1])
        input_size = checkfeaturesize(args.feature_extracted_model) + infratio
    elif args.input.startswith('core_tffeature_lab_vinflated'):
        infratio = int(args.input.split('core_tffeature_lab_vinflated')[1])
        input_size = checkfeaturesize(args.feature_extracted_model) + infratio

    
    #Non-IMPACT-based
    elif args.input == 'v':
        input_size = 3
    elif args.input == 'f':
        input_size = [args.slice_len, checkfeaturesize(args.feature_extracted_model), 0] #[height of img, width of img, length of clinical variables]
    elif args.input == 'fv':
        input_size = [args.slice_len, checkfeaturesize(args.feature_extracted_model), 3]
    elif args.input =='i':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 0] #[height of 3D resized image, width of 3D resized image, depth(=slice length) of 3D image, length of clinical variables]
    elif args.input =='iv':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 3] 
    elif args.input =='it':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 25] 
    elif args.input =='ivt':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 28]
    elif args.input =='ib':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 1] 
    elif args.input =='ivb':
        input_size = [args.targetimgsize, args.targetimgsize, args.slice_len, 4]
    elif args.input == 'm' or args.input == 'r' or args.input == 'h':
        input_size = 1
    elif args.input =='vt':
        input_size = 28    
    elif args.input =='vb':
        input_size = 4    
    elif args.input =='vs':
        input_size = 7      
    elif args.input == 'mv' or args.input == 'rv' or args.input == 'hv':
        input_size = 4
    elif args.input == 'mt' or args.input == 'rt' or args.input == 'ht':
        input_size = 26
    elif args.input == 'mvt' or args.input == 'rvt' or args.input == 'hvt':
        input_size = 29
    elif args.input == 'mb' or args.input == 'rb' or args.input == 'hb':
        input_size = 2
    elif args.input == 'mvb' or args.input == 'rvb' or args.input == 'hvb':
        input_size = 5
    else:
        raise NotImplementedError(f'{args.input} not implemented')
    return input_size

def getinout(data_batch, inp, pred, slice_len, device):
    #Input
    img = data_batch['image'].float().to(device)
    feature = data_batch['feature'].float().to(device)
    studyid = data_batch['studyid']
    numslices= data_batch['numslices']
    age = data_batch['age'].float().to(device) / 100.0
    pupil = data_batch['pupil'].float().to(device) / 2.0
    gcs_motor = data_batch['gcs_motor'].float().to(device)/ 6.0
    gcs_score = data_batch['gcs_score'].float().to(device)/15.0
    marshall = data_batch['marshall'].float().to(device)/ 6.0
    rotterdam = data_batch['rotterdam'].float().to(device)/ 6.0
    helsinki = (data_batch['helsinki'].float().to(device) + 3.0)/17.0
    treatment = data_batch['treatment'].float().to(device)
    treatment_bin = data_batch['treatment_bin'].float().to(device)
    hypoxia = data_batch['hypoxia'].float().to(device)
    hypotension = data_batch['hypotension'].float().to(device)
    glucose = data_batch['glucose'].float().to(device)
    hemoglobine = data_batch['hemoglobine'].float().to(device)
    tsah = data_batch['tsah'].float().to(device)
    edh = data_batch['edh'].float().to(device)
    impact_core = data_batch['impact_core'].float().to(device)/15.0 # min 0 max 15
    impact_core_ct = (data_batch['impact_core_ct'].float().to(device) + 4.0)/26.0 #min -4 max 22 
    impact_ct = (data_batch['impact_ct'].float().to(device) + 4.0)/11.0 #min -4 max 7
    impact_core_ct_lab = (data_batch['impact_core_ct_lab'].float().to(device) + 4.0)/33.0 # min -4 max 29
    impact_core_lab = data_batch['impact_core_lab'].float().to(device) /22.0 # min 0 max 22
    tffeature = data_batch['tffeature'].float().to(device)
    
    in_v = torch.cat((torch.unsqueeze(age, 1), torch.unsqueeze(gcs_motor, 1), torch.unsqueeze(pupil, 1)), dim=1) #shape: [batch, 3]
    
    # IMPACT_based:
    #shape: [batch, 1]
    if inp == 'core':
        input = torch.unsqueeze(impact_core, 1) 
    elif inp == 'core_ct':
        input = torch.unsqueeze(impact_core_ct, 1)
    elif inp == 'ct':
        input = torch.unsqueeze(impact_ct, 1)    
    elif inp == 'core_ct_lab':
        input = torch.unsqueeze(impact_core_ct_lab, 1)
    elif inp == 'core_lab':
        input = torch.unsqueeze(impact_core_lab, 1)
    elif inp == 'm':
        input = torch.unsqueeze(marshall, 1) 
    elif inp == 'r':
        input = torch.unsqueeze(rotterdam, 1) 
    elif inp == 'h':
        input = torch.unsqueeze(helsinki, 1) 
    #shape: [batch, 2]
    elif inp == 'core_m':
        input = torch.cat((torch.unsqueeze(impact_core, 1), torch.unsqueeze(marshall, 1)), dim=1) 
    elif inp == 'core_r':
        input = torch.cat((torch.unsqueeze(impact_core, 1), torch.unsqueeze(rotterdam, 1)), dim=1) 
    elif inp == 'core_h':
        input = torch.cat((torch.unsqueeze(impact_core, 1), torch.unsqueeze(helsinki, 1)), dim=1) 
    elif inp == 'core_m_lab':
        input = torch.cat((torch.unsqueeze(impact_core_lab, 1), torch.unsqueeze(marshall, 1)), dim=1) 
    elif inp == 'core_r_lab':
        input = torch.cat((torch.unsqueeze(impact_core_lab, 1), torch.unsqueeze(rotterdam, 1)), dim=1) 
    elif inp == 'core_h_lab':
        input = torch.cat((torch.unsqueeze(impact_core_lab, 1), torch.unsqueeze(helsinki, 1)), dim=1) 
    #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)]
    elif inp == 'image':
        input = img
    #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 1]
    elif inp == 'core_image':
        input = [img, torch.unsqueeze(impact_core, 1)]  
    elif inp == 'core_image_lab':
        input = [img, torch.unsqueeze(impact_core_lab, 1)]
    #shape: [batch, 768]
    elif inp == 'tffeature':
        input = tffeature
    #shape: [batch, 769]
    elif inp == 'core_tffeature':
        input = torch.cat((torch.unsqueeze(impact_core, 1), tffeature), dim=1) 
    elif inp == 'core_tffeature_lab':
        input = torch.cat((torch.unsqueeze(impact_core_lab, 1), tffeature), dim=1) 
    #shape: [batch, 768], [batch,1]
    elif inp == 'core_tffeaturemid':
        input = [tffeature,  torch.unsqueeze(impact_core, 1)] 
    elif inp == 'core_tffeaturemid_lab':
        input = [tffeature,  torch.unsqueeze(impact_core_lab, 1)] 
    #shape: [batch, 768+ inflated ratio of variables]
    elif inp.startswith('core_tffeature_vinflated'):
        infratio = int(inp.split('core_tffeature_vinflated')[1])
        impact_core_inf = torch.unsqueeze(impact_core, 1) * torch.ones((1,infratio), device=device)
        input = torch.cat((impact_core_inf, tffeature), dim=1) 
    elif inp.startswith('core_tffeature_lab_vinflated'):
        infratio = int(inp.split('core_tffeature_lab_vinflated')[1])
        impact_core_lab_inf = torch.unsqueeze(impact_core_lab, 1) * torch.ones((1,infratio), device=device)
        input = torch.cat((impact_core_lab_inf, tffeature), dim=1) 

    # Non-IMPACT_based (Previous trial)
    elif inp == 'v':
        input = in_v
    elif inp == 'f':
        input = torch.unsqueeze(feature, dim=1) #shape: [batch, 1, slice_len, feature dimension]
    elif inp == 'fv':
        f = torch.unsqueeze(feature, dim=1)
        input = [f, in_v]  #shape: [batch, 1, slice_len, feature dimension], [batch, 3]
    elif inp == 'i':
        input = img #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)]
    elif inp == 'iv':
        input = [img, in_v]  #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 3]
    elif inp =='it':
        input = [img, treatment]  #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 25]
    elif inp =='ivt':
        vt = torch.cat((in_v, treatment), dim=1)
        input = [img, vt] #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 28]
    elif inp =='ib':
        input = [img, torch.unsqueeze(treatment_bin, 1)]  #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 1]
    elif inp =='ivb':
        vt = torch.cat((in_v, torch.unsqueeze(treatment_bin, 1)), dim=1)
        input = [img, vt] #shape: [batch, 3, img_height, img_width, img_depth(=slice_len)], [batch, 4]
    elif inp =='vt':
        vt = torch.cat((in_v, treatment), dim=1)
        input = vt #shape: [batch, 28]
    elif inp =='vb':
        vt = torch.cat((in_v, torch.unsqueeze(treatment_bin, 1)), dim=1)
        input = vt #shape: [batch, 4]    
    elif inp =='vs':
        vt = torch.cat((in_v, treatment), dim=1)
        input = vt #shape: [batch, 7]
    elif inp == 'm':
        input = torch.unsqueeze(marshall, 1) #shape: [batch, 1]
    elif inp == 'r':
        input = torch.unsqueeze(rotterdam, 1) #shape: [batch, 1]
    elif inp == 'h':
        input = torch.unsqueeze(helsinki, 1) #shape: [batch, 1]  
    elif inp == 'mv':
        input = torch.cat((in_v, torch.unsqueeze(marshall, 1)), dim=1) #shape: [batch, 4]
    elif inp == 'rv':
        input = torch.cat((in_v, torch.unsqueeze(rotterdam, 1)), dim=1) #shape: [batch, 4]
    elif inp == 'hv':
        input = torch.cat((in_v, torch.unsqueeze(helsinki, 1)), dim=1) #shape: [batch, 4]
    elif inp == 'mt':
        input = torch.cat((treatment, torch.unsqueeze(marshall, 1)), dim=1) #shape: [batch, 26]
    elif inp == 'rt':
        input = torch.cat((treatment, torch.unsqueeze(rotterdam, 1)), dim=1) #shape: [batch, 26]
    elif inp == 'ht':
        input = torch.cat((treatment, torch.unsqueeze(helsinki, 1)), dim=1) #shape: [batch, 26]
    elif inp == 'mvt':
        input = torch.cat((in_v, treatment, torch.unsqueeze(marshall, 1)), dim=1) #shape: [batch, 29]
    elif inp == 'rvt':
        input = torch.cat((in_v, treatment, torch.unsqueeze(rotterdam, 1)), dim=1) #shape: [batch, 29]
    elif inp == 'hvt':
        input = torch.cat((in_v, treatment, torch.unsqueeze(helsinki, 1)), dim=1) #shape: [batch, 29]
    elif inp == 'mb':
        input = torch.cat((torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(marshall, 1)), dim=1) #shape: [batch, 2]
    elif inp == 'rb':
        input = torch.cat((torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(rotterdam, 1)), dim=1) #shape: [batch, 2]
    elif inp == 'hb':
        input = torch.cat((torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(helsinki, 1)), dim=1) #shape: [batch, 2]
    elif inp == 'mvb':
        input = torch.cat((in_v, torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(marshall, 1)), dim=1) #shape: [batch, 5]
    elif inp == 'rvb':
        input = torch.cat((in_v, torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(rotterdam, 1)), dim=1) #shape: [batch, 5]
    elif inp == 'hvb':
        input = torch.cat((in_v, torch.unsqueeze(treatment_bin, 1), torch.unsqueeze(helsinki, 1)), dim=1) #shape: [batch, 5]
    else:
        raise NotImplementedError(f'{inp} not implemented')

    #Output
    if pred == 'outcome':
        gt = torch.unsqueeze(data_batch['outcome'], 1).float().to(device) #Unfavorable[GOSE:1-4] =1, Favorable[GOESE:5-8] = 0
    elif pred == 'mortality':
        gt = torch.unsqueeze(data_batch['mortality'], 1).float().to(device) #Death[GOESE:1] = 1, Alive [GOSE>1] = 0
    elif pred == 'gose' or pred == 'goseldl' or pred == 'goseCDBloss':
        gt = data_batch['gose_6m'].long().to(device) #Shape is [batch]. Already mapped as {'1':0, '2_or_3':1, '4':2, '5':3, '6':4, '7':5, '8':6} in labelmap_gose function
    elif pred == 'gose4cls' or pred == 'gose4clsldl' or pred == 'gose4clsCDBloss':
        gt = data_batch['gose4cls_6m'].long().to(device) #Shape is [batch]. Already mapped as {'1':0, '2_or_3':1, '4':1, '5':2, '6':2, '7':3, '8':3} in labelmap_gose function   
    elif pred == 'disposition_5cls': 
        gt = data_batch['disposition'].long().to(device) # 0: Dead  1: Other hospital 2: Rehab unit 3: Nursing home 4: Home
    elif pred == 'home': #home or non-home disposition
        gt = torch.unsqueeze(data_batch['home'], 1).float().to(device) #home=1, non-home=0
    elif pred == 'deadhomenohome': #dead or home or non-home disposition
        gt = data_batch['deadhomenohome'].long().to(device) #dead=0, home=1, non-home=2
        
    else:
        raise NotImplementedError(f'{pred} not implemented')
        
    
    return input, gt

def getdatainfo(data_batch):
    studyid = data_batch['studyid'][0]
    numslices= data_batch['numslices'].item()
    return studyid, numslices

# def selectslice(brainarea_array, numslices, tracelengthratio, gravitycalcthr):
#     argmaxbrain = np.argmax(brainarea_array)
#     brainarea_till_max = brainarea_array[0:argmaxbrain]
#     brainarea_after_max = brainarea_array[argmaxbrain:]
#     if len(brainarea_till_max) == 0:
#         grav_peak = 0
#     else:
#         grav_start = np.where(brainarea_till_max < np.max(brainarea_array) * gravitycalcthr)[0][-1]
#         grav_end = argmaxbrain + np.where(brainarea_after_max < np.max(brainarea_array) * gravitycalcthr)[0][0]
#         grav_peak = np.dot(brainarea_array[grav_start:grav_end], np.arange(grav_start, grav_end))/np.sum(brainarea_array[grav_start:grav_end]) #float
    
#     slice_end = min(argmaxbrain + np.where(brainarea_after_max==0)[0][0], numslices - 1) #index most closer to maximum point (slice whose brain area bocomes zero for the 1st time after max point)
#     length_max_end = slice_end - grav_peak
#     slice_start = max(0, int(grav_peak - length_max_end * tracelengthratio))
#     assert slice_start < slice_end, print('slice start {} is greater than slice end {}'.format(slice_start, slice_end))
#     return slice_start, slice_end