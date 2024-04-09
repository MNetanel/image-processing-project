import os
import numpy as np
import pandas as pd
import h5py
import cv2
import numpy as np
import optuna

from sklearn.metrics import mean_absolute_percentage_error as mape
from skimage import exposure
from numpy.random import shuffle
from pathlib import Path
from preprocessing import preprocess_videos
from unsupervised_models import ICA_POH
from postprocessing import ppg2hr_by_window


# %% Define parameters and constants
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR'))
PREPROCESSED_DATA_DIR = Path(os.getenv('PREPROCESSED_DATA_DIR'))
OUTPUT_DATA_DIR = Path(os.getenv('OUTPUT_DATA_DIR'))
MAX_KERNEL_SIZE = 10
GT_DATA_DIR = os.getenv('GT_DATA_DIR')
foreheads_filepath = str(PREPROCESSED_DATA_DIR / 'foreheads_all.h5')

#%% Preprocessing
preprocess_videos(RAW_DATA_DIR, PREPROCESSED_DATA_DIR)

# %% split subjects 

test_subjects=['subject25','subject8','subject23','subject24','subject16',
               'subject12','subject38']


subjects=  ['subject49', 'subject15', 'subject36', 'subject10', 'subject48',
            'subject44', 'subject39', 'subject37', 'subject14', 'subject40',
            'subject17', 'subject46', 'subject34', 'subject31', 'subject11',
            'subject13', 'subject45', 'subject43', 'subject22', 'subject4',
            'subject33', 'subject18', 'subject47', 'subject32', 'subject41',
            'subject42', 'subject20', 'subject35', 'subject3', 'subject30',
            'subject1', 'subject9', 'subject26', 'subject27','subject5']

val_size=int(len(subjects)/5)

cross_val_5=[ {'train': x[: val_size*y ]+ x[ val_size*(y+1): ], 'validation':x[ val_size*y:val_size*(y+1) ] , 'fold': y}   for (x,y) in zip([subjects]*5,np.arange(5))]

# verification
[cross_val_5[x]['validation'] in cross_val_5[x]['train'] for x in np.arange(5)]



# %% reference frame for hist matching

with h5py.File(foreheads_filepath) as f:
    frames = np.array(f['subject49'])

reference_frame=frames[0]

cropped_reference_frame = reference_frame[30:40,45:115]


# %% Using optuna, create 5 folds and an objective function for each
    
fold_results = []

for fold in cross_val_5:
    
    subjects = fold['train']
    results=[]
    
    # 1. Define an objective function to be maximized.
    def objective(trial):
        metrics=[]
        histogram = trial.suggest_categorical('histogram', [None, 'equalization',
                                                   'matching'])
        if histogram =='matching':
           hist_crop = trial.suggest_categorical('hist_crop',[True,False])
        # 2. Suggest values for the hyperparameters using a trial object.
        color_space = trial.suggest_categorical('color_space', ['RGB', 
                                                                'Lab','Luv',
                                                                'YUV'])
        blur = trial.suggest_categorical('blur', [None, 'box', 'median', 'gaussian'])
    
        if blur is not None:
            kernel_size = trial.suggest_categorical('kernel_size',np.arange(3,MAX_KERNEL_SIZE+1,2).tolist())
            times = trial.suggest_categorical('times',(np.arange(5)+1).tolist())
           
        high_pass = trial.suggest_categorical('high_pass', [None, 'our_custom'])
        
        if high_pass is not None:  
            k = trial.suggest_categorical('kernel_size_high_pass',np.arange(3,MAX_KERNEL_SIZE+1,2).tolist())
                        
            if high_pass == 'our_custom':
                base = trial.suggest_float('base',-10,10)
                scale = trial.suggest_float('scale',-2,2)
                midline_increment = trial.suggest_float('midline_increment',-5,5)
                flip = trial.suggest_categorical('flip',[True,False])
                sharpen = trial.suggest_categorical('sharpen',[True,False])
        
        # ICA model parameters
        ica_lambda = trial.suggest_float('lambda',50,200)
        ica_LPF = trial.suggest_float('LPF',0,5)
        ica_HPF = trial.suggest_float('HPF',0,5)
        
        shuffle(subjects)
        for i_subject,subject in enumerate(subjects):

            with h5py.File(foreheads_filepath) as f:
                frames = np.array(f[subject])
                
                
            def transform(frames):
                if histogram is not None:
                    if histogram =='equalization':
                        for i, frame in enumerate(frames):
                            new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                            new_frame[:,:,2] = cv2.equalizeHist(new_frame[:,:,2]) 
                            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_HSV2RGB)              
                            frames[i] = new_frame  
                 
                    if histogram =='matching':
                       if hist_crop:
                           # croped_reference_frame
                           for i,frame in enumerate(frames):
                               
                               new_frame = exposure.match_histograms(frame, cropped_reference_frame, 
                                                                         channel_axis=-1)
                               frames[i] = new_frame 
                       else:
                           # reference_frame
                            for i,frame in enumerate(frames):
                                
                                new_frame = exposure.match_histograms(frame, reference_frame, 
                                                                          channel_axis=-1)
                                frames[i] = new_frame 
                # color spaces
                if color_space == 'RGB': # default input
                    pass
                    
                if color_space == 'Lab':
                    for i, frame in enumerate(frames):
                        new_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2Lab)
                        frames[i] = new_frame    
                    
                if color_space == 'Luv':
                    for i, frame in enumerate(frames):
                        new_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2Luv)
                        frames[i] = new_frame    
          
                if color_space == 'YUV':
                    for i, frame in enumerate(frames):
                        new_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YUV)
                        frames[i] = new_frame 
                        
                # blur types
                if blur is not None:
                    for t in np.arange(times):
        
                        if blur == 'box':
                            for i, frame in enumerate(frames):
                                new_frame = cv2.blur(frame,(kernel_size,kernel_size))
                                frames[i] = new_frame 
                                
                        if blur == 'median':
                            for i, frame in enumerate(frames):
                                new_frame = cv2.medianBlur(frame,kernel_size)
                                frames[i] = new_frame 
                                
                        if blur == 'gaussian':
                            for i, frame in enumerate(frames):
                                new_frame = cv2.GaussianBlur(frame,(kernel_size,kernel_size),0)
                                frames[i] = new_frame 
                            
                    
                if high_pass is not None:     
                    if high_pass == 'our_custom':            
                        custom_filter=np.zeros([k,k])
                        axis_2=int((k-1)/2)
                        quarter = np.zeros([axis_2,axis_2])
                        for i in np.arange(axis_2):
                            for j in np.arange(axis_2):
                                quarter[j,i]= -(base+(i+j)*scale)
                        midline =  quarter[axis_2-1] + midline_increment
                        custom_filter[:axis_2,:axis_2] = quarter
                        custom_filter[axis_2+1:,:axis_2] = np.flip(quarter,axis=0)
                        custom_filter[axis_2+1:,axis_2+1:] =  -1*np.flip(np.flip(quarter,axis=0),axis=1)
                        custom_filter[:axis_2,axis_2+1:] = -1*np.flip(quarter,axis=1)
                        custom_filter[axis_2,axis_2+1:] =  -1* np.flip(midline)
                        custom_filter[axis_2,:axis_2] = midline
                        
                        if sharpen:
                            custom_filter[axis_2+1,axis_2+1] = 1
                        
                        if flip:
                            custom_filter=custom_filter.T
                            
                        for i, frame in enumerate(frames):
                            new_frame=cv2.filter2D(src=frame, ddepth=-1, kernel=custom_filter)
                
                return frames
            
            # Transform the frames
            frames = transform(frames)
            
            yhat = ICA_POH(frames, 30,ica_LPF,ica_LPF + ica_HPF, ica_lambda)
          
            with open(GT_DATA_DIR / f'{subject}.txt', 'r') as f:
                y_raw = f.read()
            y_true = np.array([float(yy.strip("'")) for yy in y_raw.split('  ')[1:]][:len(yhat)])
        
            # add post processing
            y_true = ppg2hr_by_window(y_true)
            yhat= ppg2hr_by_window(yhat)
            
            # Calculate MAPE
            metric=mape(y_true,yhat)
            
            results.append({'trial':trial.number,
                            'subset':'train',
                            'params':trial.params,
                            'subject':subject,
                            'y_true':y_true,
                            'y_hat':yhat,
                            'mape':metric,}) # add params
            
            metrics.append(metric)
                        
            if(i_subject % 10==0):
                trial.report(np.mean(metrics), int(i_subject/10))
                
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # validate
        for i_subject,subject in enumerate(fold['validation']):
            with h5py.File(foreheads_filepath) as f:
                frames = np.array(f[subject])
                frames = transform(frames)
                yhat = ICA_POH(frames, 30,ica_LPF,ica_LPF + ica_HPF, ica_lambda)
            
                
                with open(GT_DATA_DIR / f'{subject}.txt', 'r') as f:
                    y_raw = f.read()
                y_true = np.array([float(yy.strip("'")) for yy in y_raw.split('  ')[1:]][:len(yhat)])
            
                # add post processing
                y_true= ppg2hr_by_window(y_true)
                yhat= ppg2hr_by_window(yhat)
                metric=mape(y_true,yhat)
                
                results.append({'trial':trial.number,
                                'subset':'validation',
                                'params':trial.params,
                                'subject':subject,
                                'y_true':y_true,
                                'y_hat':yhat,
                                'mape':metric,}) # add params
                
        return np.mean(metrics)     # mape(y_trues,yhats)
    

    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(direction='minimize',study_name=f"rPPG_fold_{fold['fold']}",
                                pruner=optuna.pruners.HyperbandPruner()) 
    study.optimize(objective, n_trials=250)
    fold_results.append({'fold_data': fold,
                         'results':results,
                         'study':study})

# Export results
pd.to_pickle(fold_results, OUTPUT_DATA_DIR / 'rPPG_Study_cv_results.pkl')
pd.to_pickle(results, OUTPUT_DATA_DIR / 'rPPG_results.pkl')

# %% testing 


# %% _____________ best config

def best_transform(frames):
    # histogram equalization
    for i, frame in enumerate(frames):
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        new_frame[:,:,2] = cv2.equalizeHist(new_frame[:,:,2]) 
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_HSV2RGB)              
        frames[i] = new_frame  
    
    return frames

best_ica_LPF=1.0349
best_ica_HPF=3.88798+best_ica_LPF
best_ica_lambda= 140.473

# %% _____________ get test result for best config

test_results=[]
# test set
for i_subject,subject in enumerate(test_subjects):
    
    # load test set
    with h5py.File(foreheads_filepath) as f:
        frames = np.array(f[subject])
        
        frames = best_transform(frames)
        yhat = ICA_POH(frames, 30,best_ica_LPF,best_ica_HPF, best_ica_lambda)
    
        
        with open(GT_DATA_DIR / f'{subject}.txt', 'r') as f:
            y_raw = f.read()
        y_true = np.array([float(yy.strip("'")) for yy in y_raw.split('  ')[1:]][:len(yhat)])
    
    
        # add post processing
        y_true = ppg2hr_by_window(y_true)
        yhat = ppg2hr_by_window(yhat)

        metric=mape(y_true,yhat)
        test_results.append({
                        'subset':'test',
                        'subject':subject,
                        'y_true':y_true,
                        'y_hat':yhat,
                        'mape':metric}) # add params
        
        
 
# %% _____________ look at test results

test_results_df=pd.DataFrame(test_results)
       
test_results_df['mape'].mean()


# %% _____________ baseline

from time import time

start=time()
baseline_metrics=[]
 
for subject in subjects:
    with h5py.File(foreheads_filepath) as f:
        frames = np.array(f[subject])
        
    yhat = ICA_POH(frames, 30)
  
    with open(GT_DATA_DIR / f'{subject}.txt', 'r') as f:
        y_raw = f.read()

    y_true = np.array([float(yy.strip("'")) for yy in y_raw.split('  ')[1:]][:len(yhat)])
  
  
    # add post processing   
    metric=mape(ppg2hr_by_window(y_true),ppg2hr_by_window(yhat))
    baseline_metrics.append(metric)
    


baseline_mape=np.mean(baseline_metrics)
print(f'baseline mape: {baseline_mape}') # 0.1688703979401937
print(f'time: {time()-start}')

