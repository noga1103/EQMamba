#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
last update: 05/27/2021

"""

from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import h5py
import time
import shutil
from .EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from .EqT_utils import generate_arrays_from_file, picker
from .EqT_utils import DataGeneratorTest, PreLoadGeneratorTest
#from .EqT_utils import ResidualBlock, ModelArgs
np.warnings.filterwarnings('ignore')
import datetime
from tqdm import tqdm
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def tester(input_hdf5=None,
           input_testset=None,
           input_model=None,
           output_name=None,
           detection_threshold=0.20,                
           P_threshold=0.1,
           S_threshold=0.1, 
           number_of_plots=100,
           estimate_uncertainty=True, 
           number_of_sampling=5,
           loss_weights=[0.05, 0.40, 0.55],
           loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
           input_dimention=(6000, 3),
           normalization_mode='std',
           mode='generator',
           batch_size=500,
           gpuid=None,
           gpu_limit=None):

    """
    
    Applies a trained model to a windowed waveform to perform both detection and picking at the same time.  


    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of "data" with NumPy arrays containing 3 component waveforms each 1 min long.

    input_testset: npy, default=None
        Path to a NumPy file (automaticaly generated by the trainer) containing a list of trace names.        

    input_model: str, default=None
        Path to a trained model.
        
    output_dir: str, default=None
        Output directory that will be generated. 
        
    output_probabilities: bool, default=False
        If True, it will output probabilities and estimated uncertainties for each trace into an HDF file. 
        
    detection_threshold : float, default=0.3
        A value in which the detection probabilities above it will be considered as an event.
          
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.

    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
               
    number_of_plots: float, default=10
        The number of plots for detected events outputed for each station data.
        
    estimate_uncertainty: bool, default=False
        If True uncertainties in the output probabilities will be estimated.  
        
    number_of_sampling: int, default=5
        Number of sampling for the uncertainty estimation. 
               
    loss_weights: list, default=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.
             
    loss_types: list, default=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'] 
        Loss types for detection, P picking, and S picking respectively.
        
    input_dimention: tuple, default=(6000, 3)
        Loss types for detection, P picking, and S picking respectively.          

    normalization_mode: str, default='std' 
        Mode of normalization for data preprocessing, 'max', maximum amplitude among three components, 'std', standard deviation.

    mode: str, default='generator'
        Mode of running. 'pre_load_generator' or 'generator'.
                      
    batch_size: int, default=500 
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommanded.

    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None.
         
    gpu_limit: int, default=None
        Set the maximum percentage of memory usage for the GPU.
        
      
    Returns
    -------- 
    ./output_name/X_test_results.csv: A table containing all the detection, and picking results. Duplicated events are already removed.      
        
    ./output_name/X_report.txt: A summary of the parameters used for prediction and performance.
        
    ./output_name/figures: A folder containing plots detected events and picked arrival times. 
    

    Notes
    --------
    Estimating the uncertainties requires multiple predictions and will increase the computational time. 
    
        
    """ 
              
         
    args = {
    "input_hdf5": input_hdf5,
    "input_testset": input_testset,
    "input_model": input_model,
    "output_name": output_name,
    "detection_threshold": detection_threshold,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "number_of_plots": number_of_plots,
    "estimate_uncertainty": estimate_uncertainty,
    "number_of_sampling": number_of_sampling,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "input_dimention": input_dimention,
    "normalization_mode": normalization_mode,
    "mode": mode,
    "batch_size": batch_size,
    "gpuid": gpuid,
    "gpu_limit": gpu_limit
    }  

    
    if args['gpuid']:           
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args['gpuid'])
        tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
        K.tensorflow_backend.set_session(tf.Session(config=config))
    
    save_dir = os.path.join(os.getcwd(), str(args['output_name'])+'_outputs')
    save_figs = os.path.join(save_dir, 'figures')
 
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  
    os.makedirs(save_figs) 
 
    test = np.load(args['input_testset'])
    
    print('Loading the model ...', flush=True)        
    model = load_model(args['input_model'], custom_objects={
    'SeqSelfAttention': SeqSelfAttention, 
    'FeedForward': FeedForward,
    'LayerNormalization': LayerNormalization, 
    'f1': f1,
    'ResidualBlock': ResidualBlock,
    'ModelArgs': ModelArgs})
                
    model.compile(loss = args['loss_types'],
                  loss_weights =  args['loss_weights'],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
                      
    
    print('Loading is complete!', flush=True)  
    print('Testing ...', flush=True)    
    print('Writting results into: " ' + str(args['output_name'])+'_outputs'+' "', flush=True)
    
    start_training = time.time()          

    csvTst = open(os.path.join(save_dir,'X_test_results.csv'), 'w')          
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow(['network_code', 
                          'ID', 
                          'earthquake_distance_km', 
                          'snr_db', 
                          'trace_name', 
                          'trace_category', 
                          'trace_start_time', 
                          'source_magnitude', 
                          'p_arrival_sample',
                          'p_status', 
                          'p_weight',
                          's_arrival_sample', 
                          's_status', 
                          's_weight', 
                          'receiver_type',
                          
                          'number_of_detections',
                          'detection_probability',
                          'detection_uncertainty',
                          
                          'P_pick', 
                          'P_probability',
                          'P_uncertainty',
                          'P_error',
                          
                          'S_pick',
                          'S_probability',
                          'S_uncertainty', 
                          'S_error'
                          ])  
    csvTst.flush()        
        
    plt_n = 0
    list_generator = generate_arrays_from_file(test, args['batch_size']) 
    
    pbar_test = tqdm(total= int(np.ceil(len(test)/args['batch_size'])))            
    for _ in range(int(np.ceil(len(test) / args['batch_size']))):
        pbar_test.update()
        new_list = next(list_generator)

        if args['mode'].lower() == 'pre_load_generator':                
            params_test = {'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}  
            test_set={}
            fl = h5py.File(args['input_hdf5'], 'r')
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/'+str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/'+str(ID))
                test_set.update( {str(ID) : dataset}) 
                
            test_generator = PreLoadGeneratorTest(new_list, test_set, **params_test)
            
            if args['estimate_uncertainty']:
                pred_DD = []
                pred_PP = []
                pred_SS = []
                for mc in range(args['number_of_sampling']):                    
                    predD, predP, predS = model.predict_generator(test_generator)                    
                    pred_DD.append(predD)
                    pred_PP.append(predP)               
                    pred_SS.append(predS)
                    
                pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_DD_mean = pred_DD.mean(axis=0)
                pred_DD_std = pred_DD.std(axis=0)  
                
                pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_PP_mean = pred_PP.mean(axis=0)
                pred_PP_std = pred_PP.std(axis=0)      
                
                pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_SS_mean = pred_SS.mean(axis=0)
                pred_SS_std = pred_SS.std(axis=0) 
                    
            else:
                pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(test_generator)
                pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
                pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
                pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 
                
                pred_DD_std = np.zeros((pred_DD_mean.shape))
                pred_PP_std = np.zeros((pred_PP_mean.shape))
                pred_SS_std = np.zeros((pred_SS_mean.shape))  
                
            for ts in range(pred_DD_mean.shape[0]): 
                evi =  new_list[ts] 
                dataset = test_set[evi]  
                
                try:
                    spt = int(dataset.attrs['p_arrival_sample']);
                except Exception:     
                    spt = None
                    
                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:     
                    sst = None
                    
                matches, pick_errors, yh3 =  picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                                    pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], spt, sst) 
                                              
                _output_writter_test(args, dataset, evi, test_writer, csvTst, matches, pick_errors)
                                            
                if plt_n < args['number_of_plots']:                   
                    
                    _plotter(ts, 
                            dataset,
                            evi,
                            args, 
                            save_figs, 
                            pred_DD_mean[ts], 
                            pred_PP_mean[ts],
                            pred_SS_mean[ts],
                            pred_DD_std[ts],
                            pred_PP_std[ts], 
                            pred_SS_std[ts],
                            matches)
    
                plt_n += 1

        
        else:       
            params_test = {'file_name': str(args['input_hdf5']), 
                           'dim': args['input_dimention'][0],
                           'batch_size': len(new_list),
                           'n_channels': args['input_dimention'][-1],
                           'norm_mode': args['normalization_mode']}     
    
            test_generator = DataGeneratorTest(new_list, **params_test)
            
            if args['estimate_uncertainty']:
                pred_DD = []
                pred_PP = []
                pred_SS = []          
                for mc in range(args['number_of_sampling']):
                    predD, predP, predS = model.predict_generator(generator=test_generator)
                    pred_DD.append(predD)
                    pred_PP.append(predP)               
                    pred_SS.append(predS)
                        
                pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_DD_mean = pred_DD.mean(axis=0)
                pred_DD_std = pred_DD.std(axis=0)  
                
                pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_PP_mean = pred_PP.mean(axis=0)
                pred_PP_std = pred_PP.std(axis=0)      
                
                pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(new_list), params_test['dim'])
                pred_SS_mean = pred_SS.mean(axis=0)
                pred_SS_std = pred_SS.std(axis=0) 
                    
            else:          
                pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(generator=test_generator)
                pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
                pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
                pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 
                
                pred_DD_std = np.zeros((pred_DD_mean.shape))
                pred_PP_std = np.zeros((pred_PP_mean.shape))
                pred_SS_std = np.zeros((pred_SS_mean.shape))           
                
            test_set={}
            fl = h5py.File(args['input_hdf5'], 'r')
            for ID in new_list:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('data/'+str(ID))
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('data/'+str(ID))
                test_set.update( {str(ID) : dataset})                 
            
            for ts in range(pred_DD_mean.shape[0]): 
                evi =  new_list[ts] 
                dataset = test_set[evi]  
                
                try:
                    spt = int(dataset.attrs['p_arrival_sample']);
                except Exception:     
                    spt = None
                    
                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:     
                    sst = None
                
                matches, pick_errors, yh3=picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                                       pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], spt, sst) 
               
                _output_writter_test(args,dataset, evi, test_writer, csvTst, matches, pick_errors)
                        
                if plt_n < args['number_of_plots']:  
                                            
                    _plotter(dataset,
                                evi,
                                args, 
                                save_figs, 
                                pred_DD_mean[ts], 
                                pred_PP_mean[ts],
                                pred_SS_mean[ts],
                                pred_DD_std[ts],
                                pred_PP_std[ts], 
                                pred_SS_std[ts],
                                matches)
    
                plt_n += 1
    end_training = time.time()  
    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta     
                    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')         
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
        the_file.write('input_testset: '+str(args['input_testset'])+'\n')
        the_file.write('input_model: '+str(args['input_model'])+'\n')
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Testing Parameters ======================='+'\n')  
        the_file.write('mode: '+str(args['mode'])+'\n')  
        the_file.write('finished the test in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2))) 
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('total number of tests '+str(len(test))+'\n')
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('================== Other Parameters ========================='+'\n')            
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('estimate uncertainty: '+str(args['estimate_uncertainty'])+'\n')
        the_file.write('number of Monte Carlo sampling: '+str(args['number_of_sampling'])+'\n')             
        the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
        the_file.write('P_threshold: '+str(args['P_threshold'])+'\n')
        the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')
        the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')                        

    

    
    
def _output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        matches, 
                        pick_errors,
                        ):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.    
 
    dataset: hdf5 obj
        Dataset object of the trace.

    evi: str
        Trace name.    
              
    output_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvfile: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        Contains the information for the detected and picked event.  
      
    pick_errors: dic
        Contains prediction errors for P and S picks.          
        
    Returns
    --------  
    X_test_results.csv  
    
        
    """        
    
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
        D_unc = matches[list(matches)[0]][2]

        P_arrival = matches[list(matches)[0]][3]
        P_prob = matches[list(matches)[0]][4] 
        P_unc = matches[list(matches)[0]][5] 
        P_error = pick_errors[list(matches)[0]][0]
        
        S_arrival = matches[list(matches)[0]][6] 
        S_prob = matches[list(matches)[0]][7] 
        S_unc = matches[list(matches)[0]][8]
        S_error = pick_errors[list(matches)[0]][1]  
        
    else: 
        D_prob = None
        D_unc = None 

        P_arrival = None
        P_prob = None
        P_unc = None
        P_error = None
        
        S_arrival = None
        S_prob = None 
        S_unc = None
        S_error = None
    
    if evi.split('_')[-1] == 'EV':                                     
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']  
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = dataset.attrs['trace_start_time'] 
        source_magnitude = dataset.attrs['source_magnitude'] 
        p_arrival_sample = dataset.attrs['p_arrival_sample'] 
        p_status = dataset.attrs['p_status'] 
        p_weight = dataset.attrs['p_weight'] 
        s_arrival_sample = dataset.attrs['s_arrival_sample'] 
        s_status = dataset.attrs['s_status'] 
        s_weight = dataset.attrs['s_weight'] 
        receiver_type = dataset.attrs['receiver_type']  
                   
    elif evi.split('_')[-1] == 'NO':               
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type'] 
        
    if P_unc:
        P_unc = round(P_unc, 3)


    output_writer.writerow([network_code, 
                            source_id, 
                            source_distance_km, 
                            snr_db, 
                            trace_name, 
                            trace_category, 
                            trace_start_time, 
                            source_magnitude,
                            p_arrival_sample, 
                            p_status, 
                            p_weight, 
                            s_arrival_sample, 
                            s_status,
                            s_weight,
                            receiver_type, 
                            
                            numberOFdetections,
                            D_prob,
                            D_unc,    
                            
                            P_arrival, 
                            P_prob,
                            P_unc,                             
                            P_error,
                            
                            S_arrival, 
                            S_prob,
                            S_unc,
                            S_error,
                            
                            ]) 
    
    csvfile.flush()   
    
    
    


def _plotter(dataset, evi, args, save_figs, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, matches):
    

    """ 
    
    Generates plots.

    Parameters
    ----------
    dataset: obj
        The hdf5 obj containing a NumPy array of 3 component data and associated attributes.

    evi: str
        Trace name.  

    args: dic
        A dictionary containing all of the input parameters. 

    save_figs: str
        Path to the folder for saving the plots. 

    yh1: 1D array
        Detection probabilities. 

    yh2: 1D array
        P arrival probabilities.   
      
    yh3: 1D array
        S arrival probabilities.  

    yh1_std: 1D array
        Detection standard deviations. 

    yh2_std: 1D array
        P arrival standard deviations.   
      
    yh3_std: 1D array
        S arrival standard deviations. 

    matches: dic
        Contains the information for the detected and picked event.  
          
        
    """ 
    
    
    try:
        spt = int(dataset.attrs['p_arrival_sample']);
    except Exception:     
        spt = None
                    
    try:
        sst = int(dataset.attrs['s_arrival_sample']);
    except Exception:     
        sst = None

    predicted_P = []
    predicted_S = []
    if len(matches) >=1:
        for match, match_value in matches.items():
            if match_value[3]: 
                predicted_P.append(match_value[3])
            else:
                predicted_P.append(None)
                
            if match_value[6]:
                predicted_S.append(match_value[6])
            else:
                predicted_S.append(None)

    
    data = np.array(dataset)
    
    fig = plt.figure()
    ax = fig.add_subplot(411)         
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}  
    plt.title(str(evi))
    plt.tight_layout()
    ymin, ymax = ax.get_ylim() 
    pl = None
    sl = None       
    ppl = None
    ssl = None  
    
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)     
                            
    ax = fig.add_subplot(412)   
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()                
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)    

    ax = fig.add_subplot(413) 
    plt.plot(data[:, 2], 'k')   
    plt.tight_layout()                
    if len(predicted_P) > 0:
        ymin, ymax = ax.get_ylim()
        for pt in predicted_P:
            if pt:
                ppl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Predicted_P_Arrival')
    if len(predicted_S) > 0:  
        for st in predicted_S: 
            if st:
                ssl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Predicted_S_Arrival')
                
    if ppl or ssl:    
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 

                
    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
    if args['estimate_uncertainty']:                               
        plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
        lowerD = yh1-yh1_std
        upperD = yh1+yh1_std
        plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')            
                            
        plt.plot(x, yh2, 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
        lowerP = yh2-yh2_std
        upperP = yh2+yh2_std
        plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')  
                                     
        plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
        lowerS = yh3-yh3_std
        upperS = yh3+yh3_std
        plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.ylim((-0.1, 1.1))
        plt.tight_layout()                
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    else:
        plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
        plt.plot(x, yh2, 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
        plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
        plt.tight_layout()       
        plt.ylim((-0.1, 1.1))
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.png')) 


    
    
