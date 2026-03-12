import sys
import time
import torch
from setuptools.wheel import unpack
import numpy as  np
import os
import Utils.EEGDataset as EEGDataset
from Utils.test import test
from Utils.saveresult import write_to_excel
from Utils.dataprocess import data_preprocess,fix_random_seed
from etc.global_config import config


def run():
    torch.set_num_threads(5)
    devicesid= config['train_param']['cuda']
    if devicesid != 'cpu':
        devices = f"cuda:{devicesid}" if torch.cuda.is_available() else "cpu"
    else :
        devices ='cpu'
    print(devices)
    
    datasetid = config["train_param"]['datasets']
    ratio=config['train_param']['ratio']
    Ns = config[f"data_param{datasetid}"]['Ns']
    Fs = config[f"data_param{datasetid}"]['Fs']
    Nc = config[f"data_param{datasetid}"]['Nc']
    Nm = config["model_param"]['Nm']
    ws = config['train_param']['ws']
    Nt = int(ws*Fs)
    testmethods = config['train_param']['testmethods']
    
    augtype=config['train_param']['augtype']
    
    start_time = time.time()
    
    if datasetid == 2 :
        Kf = 6
    elif datasetid == 1:
        Kf = 5
    else:
        Kf = 4
    for i in range (Kf):
        fix_random_seed(i)
        for Subject in range(1, Ns + 1):
            ACC=[]
            print(f'Kf:{i}')
            print(f'subject:{Subject}')
            info=[i,datasetid,Subject,devices]
            Data_Train, Data_Test = EEGDataset.getSSVEPIntra(subject=Subject, train_ratio=ratio,kfold=i,info=info)[:]
            EEGData_Train, AUGData_Train,EEGLabel_Train,AUGLabel_Train,EEGData_Test, EEGLabel_Test= data_preprocess(Data_Train,Data_Test)
            for testmethod in testmethods:
                acc1,acc2=test(testmethod,config,devices, EEGData_Train, AUGData_Train, EEGLabel_Train ,AUGLabel_Train, EEGData_Test,EEGLabel_Test)
                ACC.append(acc1)
                ACC.append(acc2)
            filename=f"results/set{datasetid}-{config['train_param']['ratio']}-{config['train_param']['savefilename']}"
            #sheetname=f"Sheet-{noise_ratio}-{snr_db}-{testmethod}"
            sheetname=f"Sheet-{augtype}"
            write_to_excel(i, Subject, ACC,filename=filename, sheetname=sheetname)
                

    end_time = time.time()

    print("cost_time:", end_time - start_time)

    # 3、Plot Result
    



if __name__ == '__main__':
    os.chdir(r'D:\Project\SSVEP_AUG')
    run()
